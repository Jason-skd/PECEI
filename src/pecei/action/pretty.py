"""Pretty-print a Program AST as the Python-flavored text DSL.

This is the human-readable serialisation of the canonical AST (used by the
preview/report/replay). Parsing text -> AST is NOT needed for the MVP (the LLM
emits the AST directly via tool-use); this printer makes programs legible.
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


def pretty(program: Program) -> str:
    return "\n".join(_stmt(s, 0) for s in program.body)


def _indent(n: int) -> str:
    return "    " * n


def _stmt(s, n: int) -> str:
    pad = _indent(n)
    if isinstance(s, Assign):
        return f"{pad}{s.name} = {_expr(s.expr)}"
    if isinstance(s, If):
        out = [f"{pad}if {s.test}:"]
        out.extend([_stmt(c, n + 1) for c in s.then] or [f"{_indent(n + 1)}pass"])
        if s.orelse:
            out.append(f"{pad}else:")
            out.extend([_stmt(c, n + 1) for c in s.orelse] or [f"{_indent(n + 1)}pass"])
        return "\n".join(out)
    if isinstance(s, ExprStmt):
        return f"{pad}{_expr(s.expr)}"
    return f"{pad}?"


def _expr(e) -> str:
    if isinstance(e, Lit):
        return repr(e.value)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Attr):
        return f"{_expr(e.obj)}.{e.attr}"
    if isinstance(e, Compare):
        return f"{_expr(e.left)} {e.op} {_expr(e.right)}"
    if isinstance(e, BoolOp):
        joiner = " and " if e.op == "and" else " or "
        return joiner.join(_expr(o) for o in e.operands)
    if isinstance(e, Act):
        return f"act({e.action.value})"
    if isinstance(e, Beat):
        if e.op is BeatOp.OBSERVE:
            return "beat(OBSERVE)"
        return f"beat(YIELD, {_expr(e.value)})"
    return "?"
