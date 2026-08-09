"""Compile-time type check for the policy AST.

Infers a type tag per variable (bool / obs / cell / str / int / void) and enforces:
  * ``If.test`` names a defined **bool** variable (the "if only accepts a bool
    variable" rule — the schema already made it a bare name, not an expression);
  * ``beat(OBSERVE)`` is only legal as the direct RHS of an Assign
    ("observe must be caught in a variable");
  * attribute navigation matches the object type (obs.front -> cell; cell.is_fire -> bool; ...).
"""
from __future__ import annotations

from .ast_nodes import (
    Act,
    Assign,
    Attr,
    BoolOp,
    Beat,
    BeatOp,
    Compare,
    ExprStmt,
    If,
    Lit,
    Program,
    Var,
)


class CompileError(Exception):
    """A program failed compile-time type/structure checking."""


OBS_DIRS = {"front", "left", "right", "back", "here"}
CELL_BOOLS = {
    "is_fire", "is_water", "is_stone", "is_wood", "is_metal",
    "is_wheel", "is_brain", "is_empty", "is_blocked", "is_goal",
}
CELL_STR = {"ctype"}


def type_check(program: Program) -> None:
    env: dict[str, str] = {}
    for stmt in program.body:
        _stmt(stmt, env)


def _stmt(s, env: dict[str, str]) -> None:
    if isinstance(s, Assign):
        if isinstance(s.expr, Beat) and s.expr.op is BeatOp.OBSERVE:
            env[s.name] = "obs"
        else:
            env[s.name] = _expr(s.expr, env)
    elif isinstance(s, If):
        if s.test not in env:
            raise CompileError(f"if condition references undefined variable {s.test!r}")
        if env[s.test] != "bool":
            raise CompileError(f"if condition must be a bool variable; {s.test!r} is {env[s.test]}")
        for c in s.then:
            _stmt(c, env)
        for c in s.orelse:
            _stmt(c, env)
    elif isinstance(s, ExprStmt):
        _expr(s.expr, env)


def _expr(e, env: dict[str, str]) -> str:
    if isinstance(e, Lit):
        v = e.value
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, str):
            return "str"
        return "void"
    if isinstance(e, Var):
        if e.name not in env:
            raise CompileError(f"undefined variable {e.name!r}")
        return env[e.name]
    if isinstance(e, Beat):
        if e.op is BeatOp.OBSERVE:
            raise CompileError("beat(OBSERVE) must be assigned to a variable, not used inline")
        if e.value is None or _expr(e.value, env) != "obs":
            raise CompileError("beat(YIELD, ...) expects an observation variable")
        return "void"
    if isinstance(e, Act):
        return "bool"
    if isinstance(e, Attr):
        ot = _expr(e.obj, env)
        if ot == "obs" and e.attr in OBS_DIRS:
            return "cell"
        if ot == "cell" and e.attr in CELL_BOOLS:
            return "bool"
        if ot == "cell" and e.attr in CELL_STR:
            return "str"
        raise CompileError(f"no attribute {e.attr!r} on {ot}")
    if isinstance(e, Compare):
        _expr(e.left, env)
        _expr(e.right, env)
        return "bool"
    if isinstance(e, BoolOp):
        for o in e.operands:
            if _expr(o, env) != "bool":
                raise CompileError("bool operator requires bool operands")
        return "bool"
    raise CompileError(f"unknown expression node {e!r}")
