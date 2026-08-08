"""Mock provider: deterministic program, no network. Enables offline closed-loop/CI.

The default program is open-loop (one decide per cycle): observe, yield the
observation, then advance three times — reaches example02's goal in one cycle.
"""
from __future__ import annotations

from pecei.action import Act, Assign, ExprStmt, Observe, Program, Var, Yield
from pecei.world.actions import ActionType

from ..protocol import Directive, TurnInput, TurnOutput


def default_program() -> Program:
    """ob = beat(OBSERVE); beat(YIELD, ob); act(FORWARD) x3."""
    return Program(body=[
        Assign(name="ob", expr=Observe()),
        ExprStmt(expr=Yield(value=Var(name="ob"))),
        ExprStmt(expr=Act(action=ActionType.FORWARD)),
        ExprStmt(expr=Act(action=ActionType.FORWARD)),
        ExprStmt(expr=Act(action=ActionType.FORWARD)),
    ])


class MockProvider:
    name = "mock"

    def __init__(self, program: Program | None = None) -> None:
        self._program = program or default_program()

    def decide(self, turn: TurnInput) -> TurnOutput:
        return TurnOutput(
            program=self._program if turn.directive is Directive.PLAN else None,
            reflection=None,
            raw_request={"provider": "mock", "directive": turn.directive.value},
            raw_response={"provider": "mock"},
        )
