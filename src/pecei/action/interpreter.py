"""AST interpreter: walks a type-checked Program with act/beat host callbacks.

``act``/``beat`` are NOT imported from the engine — they are injected via
:class:`Host` (dependency inversion), so this module depends on world/observation
types only, never on engine. A step budget bounds execution (defense in depth,
even though the minimal DSL has no loops).
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable

from pecei.world.actions import ActionType

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
from .typecheck import type_check
from .views import NavObs

_OPS = {
    "==": operator.eq, "!=": operator.ne,
    "<": operator.lt, ">": operator.gt,
    "<=": operator.le, ">=": operator.ge,
}


class BudgetExceeded(Exception):
    """The interpreter exceeded its computation-step budget."""


@dataclass
class Host:
    """Injected runtime callbacks (built by the engine/orchestrator)."""
    act: Callable[[ActionType], bool]          # returns whether it moved
    observe: Callable[[], NavObs]              # beat(OBSERVE) -> navigable obs
    yield_: Callable[[NavObs], None]           # beat(YIELD, obs) -> report sink


class Interpreter:
    def __init__(self, host: Host, step_budget: int = 10000) -> None:
        self.host = host
        self.step_budget = step_budget
        self._steps = 0

    def run(self, program: Program) -> None:
        type_check(program)
        env: dict[str, Any] = {}
        for s in program.body:
            self._exec(s, env)

    def _tick(self) -> None:
        self._steps += 1
        if self._steps > self.step_budget:
            raise BudgetExceeded(f"step budget {self.step_budget} exceeded")

    def _exec(self, s, env: dict[str, Any]) -> None:
        self._tick()
        if isinstance(s, Assign):
            env[s.name] = self._eval(s.expr, env)
        elif isinstance(s, If):
            for c in (s.then if env.get(s.test) else s.orelse):
                self._exec(c, env)
        elif isinstance(s, ExprStmt):
            self._eval(s.expr, env)

    def _eval(self, e, env: dict[str, Any]) -> Any:
        self._tick()
        if isinstance(e, Lit):
            return e.value
        if isinstance(e, Var):
            return env[e.name]
        if isinstance(e, Beat):
            if e.op is BeatOp.OBSERVE:
                return self.host.observe()
            self.host.yield_(self._eval(e.value, env))  # YIELD
            return None
        if isinstance(e, Act):
            return self.host.act(e.action)
        if isinstance(e, Attr):
            return getattr(self._eval(e.obj, env), e.attr)
        if isinstance(e, Compare):
            return _OPS[e.op](self._eval(e.left, env), self._eval(e.right, env))
        if isinstance(e, BoolOp):
            if e.op == "and":
                for o in e.operands:
                    if not self._eval(o, env):
                        return False
                return True
            for o in e.operands:
                if self._eval(o, env):
                    return True
            return False
        raise TypeError(f"unknown expression node {e!r}")
