"""Round engine: drives one action per round + environment tick + goal/budget.

M3 exercises it with handwritten action sequences; the policy interpreter (M4)
will map ``act`` onto :meth:`RoundEngine.apply`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pecei.world.actions import ActionResult, ActionType, apply_action
from pecei.world.world import World


class BrittleFailure(RuntimeError):
    """Raised when a brittle ego touches a metal cell (fatal interaction).

    The run must stop with Result.BRITTLE_FAILURE rather than continue.
    """


@dataclass
class RoundEngine:
    world: World
    round_budget: int = 100
    round: int = 0
    # optional per-round hook (round, eid, action, result) for tracing/preview
    on_round: Callable[[int, str, ActionType, ActionResult], None] | None = None

    def apply(self, eid: str, action: ActionType) -> ActionResult:
        """Apply one action = one round (action resolves, then environment ticks).

        A fatal interaction (brittle ego touching metal) raises
        :class:`BrittleFailure`; active statuses (burning/soaked/brittle) cost
        extra steps, which are added to the round counter.
        """
        if self.time_exceeded:
            raise RuntimeError(f"round budget {self.round_budget} exhausted")
        res = apply_action(self.world, eid, action)
        if res.failed:
            raise BrittleFailure(
                f"brittle ego touched metal on {action.value} (round {self.round + 1})"
            )
        extra = self.world.tick_environment()
        self.round += 1 + extra
        if self.on_round is not None:
            self.on_round(self.round, eid, action, res)
        return res

    @property
    def time_exceeded(self) -> bool:
        return self.round >= self.round_budget

    def at_goal(self, eid: str) -> bool:
        """True if any cell of ``eid`` occupies the goal cell."""
        if self.world.goal is None:
            return False
        gx, gy = self.world.goal
        if not self.world.grid.in_bounds(gx, gy):
            return False
        return any(o.eid == eid for o in self.world.grid.occupants(gx, gy))
