"""Ego status effects: burning / soaked / brittle + environment dynamics.

The ego's material state is driven by the terrain it stands on:

* ``fire`` cell  -> ignites the ego (unless soaked). A burning ego destroys
  the wood it touches (wood cells become passable) and pays extra steps.
* ``water`` cell -> soaks the ego and quenches burning. Soaked metal makes the
  ego brittle: touching a ``metal`` cell then FAILS the run outright
  (``Result.BRITTLE_FAILURE``).
* ``metal`` cell -> harmless to a normal ego, lethal to a brittle one.
* ``wood`` cell  -> a solid obstacle, but a burning ego burns straight through.

Statuses persist for a short duration after leaving the triggering terrain;
every active status costs +1 extra step per round (paid by the round engine).
"""
from __future__ import annotations

from dataclasses import dataclass

BURN_DURATION = 3      # rounds a burning ego stays lit after leaving the fire
SOAK_DURATION = 3      # rounds an ego stays wet after leaving the water
BRITTLE_DURATION = 3   # rounds a soaked-metal ego stays brittle

EXTRA_STEP_BURNING = 1
EXTRA_STEP_SOAKED = 1
EXTRA_STEP_BRITTLE = 1


@dataclass
class EgoStatus:
    """Per-round material state of the ego (single source for effects)."""

    burning: bool = False
    soaked: bool = False
    brittle: bool = False
    burning_left: int = 0
    soaked_left: int = 0
    brittle_left: int = 0

    def extra_steps(self) -> int:
        """Sum of extra step costs imposed by currently active statuses."""
        return (
            (EXTRA_STEP_BURNING if self.burning else 0)
            + (EXTRA_STEP_SOAKED if self.soaked else 0)
            + (EXTRA_STEP_BRITTLE if self.brittle else 0)
        )

    def to_dict(self) -> dict:
        return {
            "burning": self.burning,
            "soaked": self.soaked,
            "brittle": self.brittle,
            "burning_left": self.burning_left,
            "soaked_left": self.soaked_left,
            "brittle_left": self.brittle_left,
        }
