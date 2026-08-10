"""LLM author protocol: blind script-author + post-run feedback (gradient-free).

One ``decide()`` = author ONE complete script (Program). The author is **blind**
to live game state — there is no per-round reactive loop. A script runs from the
start until it stops; only then is :class:`Feedback` (the observations the script
chose to ``beat(YIELD)`` + the stop-report) returned and fed into the next cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

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

    Raises :class:`CompileError` (carrying a human-readable location) if the AST
    is malformed — e.g. ``act(YIELD)`` (conflating ``beat(YIELD, ...)`` with
    ``act(FORWARD|...)``), a missing field, or a bad discriminator. The caller
    (provider) catches this and surfaces it as ``TurnOutput.error`` so the runner
    records the cycle as ``COMPILE_ERROR`` and feeds the message back to the
    author. ``payload`` is a dict (Anthropic ``block.input``) or a JSON string
    (OpenAI ``tool_call.function.arguments``).
    """
    try:
        if isinstance(payload, str):
            return Program.model_validate_json(payload)
        return Program.model_validate(payload)
    except (ValidationError, ValueError) as e:  # ValueError covers malformed JSON
        raise CompileError(_format_payload_error(e)) from e


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
