"""Policy-script AST nodes (pydantic). Canonical program form = this AST.

The LLM emits this AST via tool-use against ``Program.model_json_schema()``; the
text DSL (``pretty.py``) is only a renderer over it. Key structural rule: an
``If.test`` is a *variable name* (a string), never an expression — so the schema
itself forbids ``if <expr>``. ``typecheck.type_check`` additionally guarantees
the named variable is bool-typed.

(Node class is ``Lit``, not ``Literal``, to avoid shadowing ``typing.Literal``.)
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

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


class Compare(BaseModel):
    kind: Literal["compare"] = "compare"
    op: Literal["==", "!=", "<", ">", "<=", ">="]
    left: "Expr"
    right: "Expr"


class BoolOp(BaseModel):
    kind: Literal["boolop"] = "boolop"
    op: Literal["and", "or"]
    operands: list["Expr"]


class Act(BaseModel):
    kind: Literal["act"] = "act"
    action: ActionType
    args: dict[str, Any] | None = None  # reserved (e.g. forward*2 later)


class Observe(BaseModel):
    kind: Literal["observe"] = "observe"


class Yield(BaseModel):
    kind: Literal["yield"] = "yield"
    value: "Expr"  # must be an observation variable


Expr = Annotated[
    Union[Lit, Var, Attr, Compare, BoolOp, Act, Observe, Yield],
    Field(discriminator="kind"),
]

for _m in (Lit, Var, Attr, Compare, BoolOp, Act, Observe, Yield):
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


class ExprStmt(BaseModel):
    kind: Literal["expr"] = "expr"
    expr: Expr


Stmt = Annotated[Union[Assign, If, ExprStmt], Field(discriminator="kind")]

for _m in (Assign, If, ExprStmt):
    _m.model_rebuild()


class Program(BaseModel):
    body: list[Stmt]
