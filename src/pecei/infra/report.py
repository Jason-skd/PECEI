"""Match report: the fixed outcome + optional failure snapshot + yielded views.

Per the design:
    result: SUCCESS | COMPILE_ERROR | ROUND_LIMIT_EXCEED | ENERGY_RUN_OUT | SCRIPT_ENDED
    (ENERGY deferred)
    failure_snapshot (nullable): pos, current_state (ego entity graph), complexity
    yielded: list of observation snapshots the actor chose to report
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from pecei.world.world import World

from .complexity import complexity


class Result(str, Enum):
    SUCCESS = "SUCCESS"
    COMPILE_ERROR = "COMPILE_ERROR"    # script didn't compile (bad AST or type error)
    ROUND_LIMIT_EXCEED = "ROUND_LIMIT_EXCEED"
    ENERGY_RUN_OUT = "ENERGY_RUN_OUT"  # reserved (energy budget deferred)
    SCRIPT_ENDED = "SCRIPT_ENDED"      # body fully executed, budget remaining, goal not reached


class FailureSnapshot(BaseModel):
    pos: tuple[int, int]
    current_state: dict           # ego entity graph (Entity.to_dict)
    complexity: float | None      # entity-aware path cost to the goal


class MatchReport(BaseModel):
    result: Result
    round: int
    round_budget: int
    failure_snapshot: FailureSnapshot | None = None
    yielded: list[dict] = Field(default_factory=list)


def build_report(
    world: World,
    result: Result,
    *,
    goal: tuple[int, int] | None,
    yielded: list[dict],
    round: int,
    round_budget: int,
) -> MatchReport:
    """Assemble a MatchReport. On non-success, attach a failure snapshot
    (ego position + entity graph + complexity to goal)."""
    snapshot: FailureSnapshot | None = None
    if result is not Result.SUCCESS:
        ego = world.ego
        comp = complexity(world, ego.eid, goal) if (goal and ego is not None) else None
        snapshot = FailureSnapshot(
            pos=ego.anchor,
            current_state=ego.to_dict(),
            complexity=comp,
        )
    return MatchReport(
        result=result,
        round=round,
        round_budget=round_budget,
        failure_snapshot=snapshot,
        yielded=list(yielded),
    )
