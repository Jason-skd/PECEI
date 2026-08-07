"""Entity: a rigid, connected assembly of Components on a local frame.

Components live at local (col, row) offsets. Anchoring the Entity at a world
cell with an orientation produces absolute cells via :meth:`placements`.
Movement (M3) translates/rotates the whole assembly rigidly; capability
aggregation (e.g. buoyancy) is a pluggable policy on the capability layer.
Only a BRAIN component can later execute a policy script; the Entity itself is
pure structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .component import Component, ComponentType


_ORDER = ("NORTH", "EAST", "SOUTH", "WEST")
_DELTAS = {
    "NORTH": (0, -1),
    "EAST": (1, 0),
    "SOUTH": (0, 1),
    "WEST": (-1, 0),
}


class Direction(str, Enum):
    NORTH = "NORTH"
    EAST = "EAST"
    SOUTH = "SOUTH"
    WEST = "WEST"

    def ordinal(self) -> int:
        return _ORDER.index(self.value)

    def rotate(self, quarter_turns_cw: int) -> "Direction":
        return Direction(_ORDER[(self.ordinal() + quarter_turns_cw) % 4])

    @property
    def delta(self) -> tuple[int, int]:
        """Forward (dx, dy) in grid coords (col, row; row grows downward)."""
        return _DELTAS[self.value]


def _rotate(offset: tuple[int, int], steps_cw: int) -> tuple[int, int]:
    """Rotate a local offset by ``steps_cw`` quarter-turns, clockwise on screen."""
    x, y = offset
    for _ in range(steps_cw % 4):
        x, y = -y, x
    return (x, y)


@dataclass
class Entity:
    eid: str
    components: dict[tuple[int, int], Component] = field(default_factory=dict)
    anchor: tuple[int, int] | None = None
    orientation: Direction = Direction.NORTH
    is_ego: bool = False

    def footprint(self) -> set[tuple[int, int]]:
        """Local offsets occupied by this entity."""
        return set(self.components)

    def placements(self) -> dict[tuple[int, int], tuple[tuple[int, int], Component]]:
        """Absolute cell -> (local offset, Component), applying anchor + orientation.

        Requires the entity to be anchored.
        """
        if self.anchor is None:
            raise ValueError(f"entity {self.eid!r} has no anchor; cannot place")
        ax, ay = self.anchor
        steps = self.orientation.ordinal()
        out: dict[tuple[int, int], tuple[tuple[int, int], Component]] = {}
        for local, comp in self.components.items():
            rx, ry = _rotate(local, steps)
            out[(ax + rx, ay + ry)] = (local, comp)
        return out

    def abs_cells(self) -> dict[tuple[int, int], Component]:
        """Absolute cell -> Component (convenience view of :meth:`placements`)."""
        return {cell: comp for cell, (_local, comp) in self.placements().items()}

    @property
    def has_brain(self) -> bool:
        return any(c.ctype is ComponentType.BRAIN for c in self.components.values())

    def brain_local(self) -> tuple[int, int] | None:
        """Local offset of the brain component, if any."""
        for off, comp in self.components.items():
            if comp.ctype is ComponentType.BRAIN:
                return off
        return None
