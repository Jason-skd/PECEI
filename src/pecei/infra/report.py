"""Match report: the fixed outcome + optional failure snapshot + yielded views.

Per the design:
    result: SUCCESS | COMPILE_ERROR | ROUND_LIMIT_EXCEED | ENERGY_RUN_OUT | SCRIPT_ENDED
            | BRITTLE_FAILURE
    (ENERGY deferred)
    failure_snapshot (nullable): pos, current_state (ego entity graph), complexity,
            and — when the ego is detected spinning in place — a ``stuck`` note the
            author can act on next cycle.
    yielded: list of observation snapshots the actor chose to report
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from pecei.world.world import World

from .complexity import complexity
from .trace import Trace


class Result(str, Enum):
    SUCCESS = "SUCCESS"
    COMPILE_ERROR = "COMPILE_ERROR"    # script didn't compile (bad AST or type error)
    ROUND_LIMIT_EXCEED = "ROUND_LIMIT_EXCEED"
    ENERGY_RUN_OUT = "ENERGY_RUN_OUT"  # reserved (energy budget deferred)
    SCRIPT_ENDED = "SCRIPT_ENDED"      # body fully executed, budget remaining, goal not reached
    BRITTLE_FAILURE = "BRITTLE_FAILURE"  # brittle ego touched a metal cell (fatal)


class FailureSnapshot(BaseModel):
    pos: tuple[int, int]
    current_state: dict           # ego entity graph (Entity.to_dict)
    complexity: float | None      # entity-aware path cost to the goal
    stuck: str | None = None      # human-readable "spinning/stuck" note, None when moving freely


class MatchReport(BaseModel):
    result: Result
    round: int
    round_budget: int
    failure_snapshot: FailureSnapshot | None = None
    yielded: list[dict] = Field(default_factory=list)


# A recent window of distinct visited cells smaller than this (while running many
# rounds) reads as "going nowhere": the author is hugging a wall / spinning.
_STUCK_WINDOW = 6


def detect_stuck(trace: Trace) -> str | None:
    """Detect that the ego stopped making progress (spinning / hugging a wall).

    Looks at the distinct ``(anchor, orientation)`` poses over the trace's recent
    tail. A wandering ego visits many cells; a stuck one keeps revisiting a tiny
    set (e.g. the corner loop: blocked -> TURNRIGHT -> blocked -> ...). Returns a
    short author-facing note, or ``None`` when the run moved freely or is too
    short to judge.
    """
    poses = [
        (ev.anchor_after, ev.orientation_after)
        for ev in trace.events
        if ev.anchor_after is not None
    ]
    tail = poses[-12:]
    if len(tail) < _STUCK_WINDOW:
        return None
    distinct = len(set(tail))
    if distinct < _STUCK_WINDOW:
        cell = tail[-1][0]
        return (
            f"stuck: you kept revisiting the same few cells (only {distinct} distinct "
            f"poses over the last {len(tail)} rounds, near {cell}). Your movement rule "
            f"loops in place — pick a turn that actually reaches an unvisited cell."
        )
    return None


def build_report(
    world: World,
    result: Result,
    *,
    goal: tuple[int, int] | None,
    yielded: list[dict],
    round: int,
    round_budget: int,
    trace: Trace | None = None,
) -> MatchReport:
    """Assemble a MatchReport. On non-success, attach a failure snapshot
    (ego position + entity graph + complexity to goal + a stuck note if the ego
    was detected spinning, derived from ``trace`` when given)."""
    snapshot: FailureSnapshot | None = None
    if result is not Result.SUCCESS:
        ego = world.ego
        comp = complexity(world, ego.eid, goal) if (goal and ego is not None) else None
        stuck = detect_stuck(trace) if trace is not None else None
        snapshot = FailureSnapshot(
            pos=ego.anchor,
            current_state=ego.to_dict(),
            complexity=comp,
            stuck=stuck,
        )
    return MatchReport(
        result=result,
        round=round,
        round_budget=round_budget,
        failure_snapshot=snapshot,
        yielded=list(yielded),
    )
