"""AST interpreter: walks a type-checked Program with act/beat host callbacks.

``act``/``beat`` are NOT imported from the engine — they are injected via
:class:`Host` (dependency inversion), so this module depends on world/observation
types only, never on engine. A step budget bounds execution (defense in depth —
the backstop that lets ``while``/``for`` loop without risking an infinite loop).
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from pecei.world.actions import ActionType

from .ast_nodes import (
    Act,
    Assign,
    Attr,
    At,
    BoolOp,
    Beat,
    BeatOp,
    Compare,
    ExprStmt,
    For,
    If,
    Lit,
    Program,
    Var,
    While,
)
from .typecheck import type_check

if TYPE_CHECKING:
    from pecei.observation import Observation

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
    act: Callable[[ActionType], bool]              # returns whether it moved
    observe: Callable[[], "Observation"]           # beat(OBSERVE) -> egocentric obs
    yield_: Callable[["Observation"], None]        # beat(YIELD, obs) -> report sink


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
        elif isinstance(s, While):
            # Re-read the named bool variable before every iteration: a body that
            # re-senses (beat(OBSERVE)) and re-assigns the predicate advances the
            # blind run. A per-iteration _tick bounds the total even when the
            # body is empty (so an unchanging predicate cannot hang forever).
            while env.get(s.test):
                self._tick()
                for c in s.body:
                    self._exec(c, env)
        elif isinstance(s, For):
            n = self._eval(s.count, env)  # type-checked int; evaluated once
            for i in range(n):
                self._tick()  # bounds a huge count even with an empty body
                if s.var is not None:
                    env[s.var] = i
                for c in s.body:
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
        if isinstance(e, At):
            obs = self._eval(e.obj, env)
            return obs.at(self._eval(e.dx, env), self._eval(e.dy, env))
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
