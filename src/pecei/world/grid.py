"""Grid: the ground-truth spatial store of the world.

Each cell holds a list of :class:`Occupant` (stacking allowed, e.g. a wood
bridge over water). The grid knows nothing about observation, actions, or
LLMs; FOV/Observation is a derived layer (M3).

Coordinate convention: (x=col, y=row), row grows downward. _cells[y][x].
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .component import Component, ComponentType
from .entity import Entity


_GLYPH: dict[ComponentType, str] = {
    ComponentType.STONE: "#",
    ComponentType.WOOD: "w",
    ComponentType.FIRE: "*",
    ComponentType.WATER: "~",
    ComponentType.WHEEL: "o",
    ComponentType.BRAIN: "@",
    ComponentType.METAL: "m",
}

# Component types that physically block movement. Liquids (water) and fire are
# non-blocking terrain: they affect an actor through attributes, not collision.
# This set is tunable as behavior is fleshed out (M3).
_SOLID: frozenset[ComponentType] = frozenset({
    ComponentType.STONE,
    ComponentType.METAL,
    ComponentType.WOOD,
    ComponentType.WHEEL,
    ComponentType.BRAIN,
})


@dataclass
class Occupant:
    """A component of a specific Entity occupying one cell."""
    eid: str
    local: tuple[int, int]
    component: Component


@dataclass
class Grid:
    width: int
    height: int
    _cells: list[list[list[Occupant]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cells = [[[] for _ in range(self.width)] for _ in range(self.height)]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _check(self, x: int, y: int) -> None:
        if not self.in_bounds(x, y):
            raise IndexError(f"cell ({x},{y}) out of bounds {self.width}x{self.height}")

    def occupants(self, x: int, y: int) -> list[Occupant]:
        self._check(x, y)
        return self._cells[y][x]

    def is_blocked(self, x: int, y: int) -> bool:
        """True if any occupant is solid."""
        return any(o.component.ctype in _SOLID for o in self.occupants(x, y))

    def place(self, entity: Entity) -> None:
        """Write an Entity's components into their absolute cells (stacking)."""
        for (x, y), (local, comp) in entity.placements().items():
            self._check(x, y)
            self._cells[y][x].append(Occupant(entity.eid, local, comp))

    def remove(self, eid: str) -> None:
        """Remove all occupants of entity ``eid`` from every cell."""
        for y in range(self.height):
            for cell in self._cells[y]:
                if any(o.eid == eid for o in cell):
                    cell[:] = [o for o in cell if o.eid != eid]

    def cells(self):
        """Yield ``(x, y, occupants)`` for every cell (occupants is the live list)."""
        for y in range(self.height):
            for x in range(self.width):
                yield x, y, self._cells[y][x]

    def ascii(self) -> str:
        """Debug dump: one glyph per cell (top occupant wins; '.' = empty)."""
        rows: list[str] = []
        for y in range(self.height):
            chars: list[str] = []
            for x in range(self.width):
                occs = self._cells[y][x]
                chars.append(_GLYPH[occs[-1].component.ctype] if occs else ".")
            rows.append(" ".join(chars))
        return "\n".join(rows)
