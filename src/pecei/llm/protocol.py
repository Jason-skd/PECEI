"""LLM author protocol: blind script-author + post-run feedback (gradient-free).

One ``decide()`` = author ONE complete script (Program). The author is **blind**
to live game state — there is no per-round reactive loop. A script runs from the
start until it stops; only then is :class:`Feedback` (the observations the script
chose to ``beat(YIELD)`` + the stop-report) returned and fed into the next cycle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from pecei.action import CompileError, Program
from pecei.infra import FailureSnapshot, Result


class Directive(str, Enum):
    PLAN = "PLAN"        # author a script that runs from start to stop
    # forward-compat (triple-loop memory ops; structured-field contract lands later):
    REFLECT = "REFLECT"  # verbal reflection on a failure (§5.2 Reflexion)
    COMPRESS = "COMPRESS"  # distill snowball into durable principles (§5.4)
    STORE = "STORE"      # persist an atom/principle to shared memory (§5.3)


@dataclass
class Feedback:
    """Outcome of the PREVIOUS cycle, fed back to the author. ``None`` on cycle 1."""

    stop_reason: Result                                      # SUCCESS | COMPILE_ERROR | ROUND_LIMIT_EXCEED | ENERGY_RUN_OUT | SCRIPT_ENDED
    rounds_used: int
    script: str = ""                                         # pretty(program) the author wrote this cycle ("" if none / parse failed)
    yielded: list[dict] = field(default_factory=list)        # observations the script beat(YIELD)'d
    failure_snapshot: FailureSnapshot | None = None
    compile_error: str | None = None                         # set iff stop_reason is COMPILE_ERROR
    extra: str | None = None


@dataclass
class TurnInput:
    directive: Directive = Directive.PLAN
    instructions: str | None = None                          # authoritative, author-immutable (e.g. experiment k/N, role)
    seed_observation: dict = field(default_factory=dict)     # PARTIAL start-pose view (90° cone), NOT the full map; the rest is learned via yields
    feedback: Feedback | None = None                         # previous cycle's outcome (None on first cycle)
    snowball: list[dict] = field(default_factory=list)       # prior cycles: {index, script, stop_reason, rounds, ...}
    extra: str | None = None


@dataclass
class TurnOutput:
    program: Program | None = None
    reflection: str | None = None
    error: str | None = None          # compile error text if the tool-call didn't validate
    raw_request: dict | None = None   # feeds the trace's llm_request slot
    raw_response: dict | None = None  # feeds the trace's llm_response slot


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def decide(self, turn: TurnInput) -> TurnOutput: ...


def parse_program(payload: dict | str) -> Program:
    """Validate an LLM tool-call ``payload`` into a Program.

    The payload is first run through a LENIENT normaliser that repairs the shape
    errors a model most often makes when writing the nested AST by hand
    (see :func:`_normalize`); the strict pydantic AST then validates the
    canonical form. Raises :class:`CompileError` (carrying a human-readable
    location) if it is still malformed. The caller (provider) surfaces this as
    ``TurnOutput.error`` so the runner records the cycle as ``COMPILE_ERROR`` and
    feeds the message back to the author.
    """
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        if isinstance(data, dict):
            data = _normalize(data)
        return Program.model_validate(data)
    except (ValidationError, ValueError) as e:  # ValueError covers malformed JSON
        raise CompileError(_format_payload_error(e)) from e


# ---- lenient normalisation (schema aligns to the taught surface syntax) ----

# Surface form -> canonical statement kind. The model writes `{"assign": {...}}`
# or omits the `kind` tag entirely; the AST is a discriminated union that needs it.
_STMT_KEYS = ("assign", "if", "while", "for", "expr")
# Node kind inferred from a dict's own field(s), when the model omits `kind`.
_EXPR_KIND_BY_KEY = {
    "value": "lit", "name": "var", "attr": "attr", "at": "at", "op": None,
    "action": "act", "left": "compare", "operands": "boolop", "obj": None,
}
_BOOL_OPS = {"and", "or", "not"}
_BEAT_OPS = {"OBSERVE", "YIELD"}


def _normalize(node: Any, *, in_body: bool = False) -> Any:
    """Recursively repair common LLM AST-shape mistakes. Pure on dicts/lists.

    ``in_body`` marks that this node is a direct element of a statement list
    (Program.body, or the body/then/orelse of a compound statement) — the only
    place where a bare act/beat must be wrapped in an ExprStmt.
    """
    if isinstance(node, list):
        return [_normalize(n, in_body=True) for n in node]
    if not isinstance(node, dict):
        return node

    # Single-key surface-statement wrapper: {"assign": {...}} -> {"kind":"assign", ...}.
    # Only at statement position (in_body): in an expression position a dict like
    # {"beat": {...}} is an expression wrapper handled by _infer_expr_kind, not a
    # statement wrapper. The wrapper's value may itself be wrapper-style (e.g.
    # {"assign": {"expr": {"beat": {...}}}}), so recurse into the merged result.
    if in_body:
        wrapped = None
        for sk in _STMT_KEYS:
            if sk in node and "kind" not in node:
                inner = node[sk]
                wrapped = {"kind": sk}
                if sk == "expr":
                    # ExprStmt wrapper: {"expr": <expression>} -> {"kind":"expr","expr":...}
                    wrapped["expr"] = inner
                elif isinstance(inner, dict):
                    wrapped.update(inner)
                break
        if wrapped is not None:
            return _normalize(wrapped, in_body=True)

    if "body" in node and isinstance(node["body"], list):
        node = {**node, "body": [_normalize(s, in_body=True) for s in node["body"]]}

    # Fill a missing kind from the field signature. _infer_expr_kind mutates
    # `node` in place (it promotes surface-wrapper inner fields up), so call it
    # before spreading — otherwise the in-place promotion is lost on the copy.
    if "kind" not in node:
        if "test" in node and ("then" in node or "orelse" in node):
            node["kind"] = "if"
        elif "test" in node and "body" in node:
            node["kind"] = "while"
        elif "count" in node:
            node["kind"] = "for"
        elif "name" in node and "expr" in node:
            node["kind"] = "assign"
        else:
            _infer_expr_kind(node)

    kind = node.get("kind")

    # A bare act/beat written directly in a statement list -> wrap in ExprStmt.
    # (Only at body level: an act/beat that is the value of an `expr` field is
    # already in the right place and must NOT be re-wrapped.)
    if in_body and kind in ("act", "beat"):
        node = {"kind": "expr", "expr": {k: v for k, v in node.items()}}

    # Fill a missing beat op from its shape (YIELD takes a value, OBSERVE does not).
    if kind == "beat" and "op" not in node:
        node = {**node, "op": "YIELD" if node.get("value") is not None else "OBSERVE"}

    # A condition given as {"kind":"var","name":x} -> the bare name string the
    # schema wants for If.test / While.test.
    if kind in ("if", "while") and isinstance(node.get("test"), dict) and node["test"].get("kind") == "var":
        node = {**node, "test": node["test"]["name"]}

    # Recurse into sub-expression / sub-statement fields (NOT in_body: these are
    # expression positions, so a bare act/beat here stays as an expression).
    out = dict(node)
    for f in ("obj", "dx", "dy", "left", "right", "expr", "value", "count"):
        if f in out and isinstance(out[f], (dict, list)):
            out[f] = _normalize(out[f], in_body=False)
    for f in ("operands",):
        if f in out and isinstance(out[f], list):
            out[f] = [_normalize(n, in_body=False) for n in out[f]]
    for f in ("then", "orelse"):
        if f in out and isinstance(out[f], list):
            out[f] = [_normalize(n, in_body=True) for n in out[f]]
    return out


def _infer_expr_kind(node: dict) -> str:
    """Guess an expression's kind from its field signature when `kind` is missing.

    Handles two shapes: field-signature dicts ({op, attr, value, ...}) and
    surface wrappers ({beat: {...}}, {at: {...}}, ...). The wrapper case promotes
    the inner dict's fields up so the node becomes canonical.
    """
    # Surface wrappers: {"beat": {...}} / {"at": {...}} / {"attr": {...}} etc.
    for key, kind in (("beat", "beat"), ("at", "at"), ("attr", "attr"),
                      ("act", "act"), ("boolop", "boolop"), ("compare", "compare"),
                      ("var", "var"), ("lit", "lit")):
        if key in node and isinstance(node[key], dict):
            inner = node.pop(key)
            node.update(inner)
            node["kind"] = kind
            return kind
    if "action" in node:
        return "act"
    if "op" in node:
        op = node.get("op")
        if op in _BEAT_OPS:
            return "beat"
        if op in _BOOL_OPS:
            return "boolop"
    if "left" in node or "right" in node:
        return "compare"
    if "operands" in node:
        return "boolop"
    if "value" in node:
        return "lit"
    if "name" in node:
        return "var"
    return "expr"


def _format_payload_error(e: Exception) -> str:
    if isinstance(e, ValidationError) and e.errors():
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        msg = first.get("msg", "invalid program")
        bad = first.get("input")
        # echo the offending scalar (e.g. action 'YIELD') so the author sees what it did wrong
        got = f" (got {bad!r})" if isinstance(bad, (str, int, bool)) else ""
        return f"at {loc or '<root>'}: {msg}{got}"
    return str(e) or "invalid program"
