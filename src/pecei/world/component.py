"""Component: the smallest typed material piece in the world.

Every cell occupant is a Component belonging to some Entity. A Component
carries a fixed :class:`ComponentType` and a bag of numeric/boolean attributes
(weight, buoyancy, burn, wet, fireproof, ...). Defaults are seeded from the
type and may be overridden per-instance (e.g. a fireproofed piece of wood).

Capability aggregation across a whole Entity (does it float? does it burn?)
lives on the capability layer (M3); this module only carries per-component
state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComponentType(str, Enum):
    STONE = "stone"
    WOOD = "wood"
    FIRE = "fire"
    WATER = "water"
    WHEEL = "wheel"
    BRAIN = "brain"
    METAL = "metal"
    GOAL = "goal"
    BOUNDARY = "boundary"


# Per-type default attributes. Keys are the loose trait/state names from the
# design ("burn" = flammable trait, "wet" = currently wet). These seed every
# component of a given type; map authors override via Component.of(...).
_DEFAULT_ATTRS: dict[ComponentType, dict[str, Any]] = {
    ComponentType.STONE: {"weight": 3.0, "buoyancy": 0.0, "burn": False, "wet": False, "fireproof": True},
    ComponentType.WOOD: {"weight": 1.0, "buoyancy": 0.5, "burn": True, "wet": False, "fireproof": False},
    ComponentType.FIRE: {"weight": 0.0, "buoyancy": 0.0, "burn": False, "wet": False, "fireproof": True},
    ComponentType.WATER: {"weight": 0.0, "buoyancy": 0.0, "burn": False, "wet": True, "fireproof": True},
    ComponentType.WHEEL: {"weight": 1.0, "buoyancy": 0.0, "burn": False, "wet": False, "fireproof": False},
    ComponentType.BRAIN: {"weight": 0.5, "buoyancy": 0.0, "burn": False, "wet": False, "fireproof": False},
    ComponentType.METAL: {"weight": 2.0, "buoyancy": 0.0, "burn": False, "wet": False, "fireproof": True},
    # GOAL is a non-physical marker: it occupies the goal cell so the observer
    # can perceive it through the same ctype channel as any other component, but
    # it has no weight and never blocks movement (it is NOT in Grid._SOLID).
    ComponentType.GOAL: {"weight": 0.0, "buoyancy": 0.0, "burn": False, "wet": False, "fireproof": True},
    ComponentType.BOUNDARY: {"weight": 0.0, "buoyancy": 0.0, "burn": False, "wet": False, "fireproof": True},
}


# Component types that physically block movement. The single source of truth —
# shared by Grid.is_blocked (collision) and CellView.is_blocked (perception) so
# the agent's predicate can never disagree with the world. Liquids (water) and
# fire are non-blocking terrain; GOAL is a non-physical marker. BOUNDARY is the
# perceived map edge (it never occupies a real cell — only the observation layer
# emits it), and it blocks: the agent must treat the edge as impassable.
SOLID: frozenset[ComponentType] = frozenset({
    ComponentType.STONE,
    ComponentType.METAL,
    ComponentType.WOOD,
    ComponentType.WHEEL,
    ComponentType.BRAIN,
    ComponentType.BOUNDARY,
})


@dataclass
class Component:
    ctype: ComponentType
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, ctype: "ComponentType | str", **overrides: Any) -> "Component":
        """Create a Component, seeding defaults from its type, then applying overrides."""
        ct = ctype if isinstance(ctype, ComponentType) else ComponentType(ctype)
        merged: dict[str, Any] = dict(_DEFAULT_ATTRS[ct])
        merged.update(overrides)
        return cls(ctype=ct, attrs=merged)

    def get(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)
