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

from pecei.action import Program
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

    stop_reason: Result                                      # SUCCESS | ROUND_LIMIT_EXCEED | ENERGY_RUN_OUT | SCRIPT_ENDED
    rounds_used: int
    yielded: list[dict] = field(default_factory=list)        # observations the script beat(YIELD)'d
    failure_snapshot: FailureSnapshot | None = None
    extra: str | None = None


@dataclass
class TurnInput:
    directive: Directive = Directive.PLAN
    instructions: str | None = None                          # authoritative, author-immutable (e.g. experiment k/N, role)
    map_desc: dict = field(default_factory=dict)             # static puzzle: size, goal, ego pose, entities
    feedback: Feedback | None = None                         # previous cycle's outcome (None on first cycle)
    snowball: list[dict] = field(default_factory=list)       # prior cycles: {index, script, stop_reason, rounds, ...}
    extra: str | None = None


@dataclass
class TurnOutput:
    program: Program | None = None
    reflection: str | None = None
    raw_request: dict | None = None   # feeds the trace's llm_request slot
    raw_response: dict | None = None  # feeds the trace's llm_response slot


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def decide(self, turn: TurnInput) -> TurnOutput: ...
