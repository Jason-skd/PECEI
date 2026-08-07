"""World: the canonical mutable game state.

Owns the Entity objects (single source of truth for anchors/orientation) and a
Grid (spatial index kept in sync). The World is the only layer that holds
mutable placement state; primitive actions mutate it, observation/engine read or
drive it. Environment dynamics tick here (M3 stub).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .entity import Entity
from .grid import Grid


@dataclass
class World:
    grid: Grid
    entities: dict[str, Entity] = field(default_factory=dict)
    goal: tuple[int, int] | None = None

    @classmethod
    def empty(cls, width: int, height: int, goal: tuple[int, int] | None = None) -> "World":
        return cls(grid=Grid(width, height), goal=goal)

    def add(self, entity: Entity) -> None:
        if entity.anchor is None:
            raise ValueError(f"entity {entity.eid!r} has no anchor; cannot add to world")
        self.entities[entity.eid] = entity
        self.grid.place(entity)

    def entity(self, eid: str) -> Entity:
        return self.entities[eid]

    @property
    def ego(self) -> Entity | None:
        for e in self.entities.values():
            if e.is_ego:
                return e
        return None

    def occupies(self, eid: str, x: int, y: int) -> bool:
        return any(o.eid == eid for o in self.grid.occupants(x, y))

    def tick_environment(self) -> None:
        """Advance non-agent dynamics by one round.

        M3 stub: fire spread / water flow / material state changes land later.
        Kept as a no-op hook so the round engine's tick semantics are in place.
        """
        return None
