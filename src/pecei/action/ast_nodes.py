"""Policy-script AST nodes (pydantic). Canonical program form = this AST.

The LLM emits this AST via tool-use against ``Program.model_json_schema()``; the
text DSL (``pretty.py``) is only a renderer over it. Key structural rule: an
``If.test`` is a *variable name* (a string), never an expression — so the schema
itself forbids ``if <expr>``. ``typecheck.type_check`` additionally guarantees
the named variable is bool-typed.

(Node class is ``Lit``, not ``Literal``, to avoid shadowing ``typing.Literal``.)
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator

from pecei.world.actions import ActionType


# ---------------- expressions ----------------

class Lit(BaseModel):
    kind: Literal["lit"] = "lit"
    value: bool | int | str | None = None


class Var(BaseModel):
    kind: Literal["var"] = "var"
    name: str


class Attr(BaseModel):
    kind: Literal["attr"] = "attr"
    obj: "Expr"
    attr: str

    @field_validator("obj", mode="before")
    @classmethod
    def _coerce_obj(cls, v: Any) -> Any:
        # Surface syntax `c.is_goal` writes the object as a bare variable name.
        if isinstance(v, str):
            return {"kind": "var", "name": v}
        return v


class At(BaseModel):
    # Cell lookup in the egocentric camera frame: ``ob.at(dx, dy)`` -> a cell,
    # whose ``is_X`` predicates / ``ctype`` can then be read. dx/dy are int
    # expressions in canonical offsets (+x = the observer's gaze; (0,0) = self).
    # Unseen offsets yield an empty cell, so every predicate is safe.
    #
    # Schema aligns to the taught surface syntax: the model naturally writes
    # ``ob.at(5, 0)`` with a bare int offset and a bare ``ob`` variable name, so
    # a bare int offset is read as ``Lit(int)`` and a bare string ``obj`` is read
    # as ``Var(str)`` — no nested ``{"kind":...}`` wrappers required.
    kind: Literal["at"] = "at"
    obj: "Expr"
    dx: "Expr"
    dy: "Expr"

    @field_validator("obj", mode="before")
    @classmethod
    def _coerce_obj(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"kind": "var", "name": v}
        return v

    @field_validator("dx", "dy", mode="before")
    @classmethod
    def _coerce_offset(cls, v: Any) -> Any:
        if isinstance(v, int) and not isinstance(v, bool):
            return {"kind": "lit", "value": v}
        return v


class Compare(BaseModel):
    kind: Literal["compare"] = "compare"
    op: Literal["==", "!=", "<", ">", "<=", ">="]
    left: "Expr"
    right: "Expr"


class BoolOp(BaseModel):
    kind: Literal["boolop"] = "boolop"
    op: Literal["and", "or", "not"]
    operands: list["Expr"]

    @field_validator("operands", mode="before")
    @classmethod
    def _coerce_operands(cls, v: Any) -> Any:
        # `not p` / `a and b` may be written with bare variable-name operands.
        if isinstance(v, list):
            return [{"kind": "var", "name": o} if isinstance(o, str) else o for o in v]
        return v


class Act(BaseModel):
    kind: Literal["act"] = "act"
    action: ActionType
    args: dict[str, Any] | None = None  # reserved (e.g. forward*2 later)


class BeatOp(str, Enum):
    """The two senses of the single ``beat`` verb (OBSERVE / YIELD)."""

    OBSERVE = "OBSERVE"
    YIELD = "YIELD"


class Beat(BaseModel):
    # The taught surface syntax is one verb, ``beat``: ``beat(OBSERVE)`` senses
    # (and must be assigned), ``beat(YIELD, obs)`` reports an observation. The
    # discriminator ``kind`` is therefore ``"beat"`` (not observe/yield), so the
    # schema the LLM emits against matches the language it is taught.
    kind: Literal["beat"] = "beat"
    op: BeatOp
    value: "Expr | None" = None  # the observation variable; required when op is YIELD


Expr = Annotated[
    Union[Lit, Var, Attr, At, Compare, BoolOp, Act, Beat],
    Field(discriminator="kind"),
]

for _m in (Lit, Var, Attr, At, Compare, BoolOp, Act, Beat):
    _m.model_rebuild()


# ---------------- statements ----------------

class Assign(BaseModel):
    kind: Literal["assign"] = "assign"
    name: str
    expr: Expr


class If(BaseModel):
    # ``test`` is a bool *variable name* (string), never an expression node.
    kind: Literal["if"] = "if"
    test: str
    then: list["Stmt"] = []
    orelse: list["Stmt"] = []


class While(BaseModel):
    # Same constraint as ``If``: ``test`` is a bool *variable name* (string),
    # never an expression node — so the schema itself forbids ``while <expr>``.
    # The interpreter re-reads the named variable before each iteration, so a
    # loop that re-senses (``beat(OBSERVE)``) and re-assigns the predicate each
    # pass runs blind until the world or the step budget stops it.
    kind: Literal["while"] = "while"
    test: str
    body: list["Stmt"] = []


class For(BaseModel):
    # Bounded repetition: ``count`` is an int-valued expression evaluated ONCE
    # before the loop (the iteration bound cannot change mid-loop). ``var``, if
    # given, binds the 0-based iteration index (an int) inside the body —
    # flat-scoped like the rest of the language (it leaks after the loop, same
    # as a variable assigned inside an ``If``). The step budget caps the total.
    kind: Literal["for"] = "for"
    count: "Expr"
    var: str | None = None
    body: list["Stmt"] = []


class ExprStmt(BaseModel):
    kind: Literal["expr"] = "expr"
    expr: Expr


Stmt = Annotated[Union[Assign, If, While, For, ExprStmt], Field(discriminator="kind")]

for _m in (Assign, If, While, For, ExprStmt):
    _m.model_rebuild()


class Program(BaseModel):
    body: list[Stmt]
