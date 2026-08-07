"""Round engine: drives one action per round + environment tick + goal/budget.

M3 exercises it with handwritten action sequences; the policy interpreter (M4)
will map ``act`` onto :meth:`RoundEngine.apply`.
"""
from __future__ import annotations

from dataclasses import dataclass

from pecei.world.actions import ActionResult, ActionType, apply_action
from pecei.world.world import World


@dataclass
class RoundEngine:
    world: World
    round_budget: int = 100
    round: int = 0

    def apply(self, eid: str, action: ActionType) -> ActionResult:
        """Apply one action = one round (action resolves, then environment ticks)."""
        if self.time_exceeded:
            raise RuntimeError(f"round budget {self.round_budget} exhausted")
        res = apply_action(self.world, eid, action)
        self.world.tick_environment()
        self.round += 1
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
