"""Complexity: an entity-aware grid path cost (the report's "loss function").

Shortest-path cost from an entity's anchor to a target, where each cell's entry
cost depends on terrain modulated by the entity's capabilities (fireproof,
floats). Doubles as the LLM planning heuristic later (``obs.goal.complexity``).

Treats the entity as a point at its anchor — a deliberate heuristic, not full
multi-cell pathfinding (the complexity surrogate doesn't need it).
"""
from __future__ import annotations

import heapq

from pecei.world.capability import capability, floats
from pecei.world.component import ComponentType
from pecei.world.world import World

BASE_STEP = 1.0
FIRE_PENALTY = 8.0
WATER_PENALTY = 8.0

_SOLID = {
    ComponentType.STONE,
    ComponentType.METAL,
    ComponentType.WOOD,
    ComponentType.WHEEL,
    ComponentType.BRAIN,
}


def complexity(world: World, eid: str, target: tuple[int, int]) -> float | None:
    """Shortest entry-cost from entity ``eid``'s anchor to ``target`` (None if unreachable)."""
    ent = world.entity(eid)
    start = ent.anchor
    if start == target:
        return 0.0
    fireproof = bool(capability(ent, "fireproof"))
    does_float = bool(floats(ent))
    grid = world.grid

    def cell_cost(x: int, y: int) -> float | None:
        if not grid.in_bounds(x, y):
            return None
        occs = grid.occupants(x, y)
        # impassable if a solid occupant belongs to a different entity
        if any(o.component.ctype in _SOLID and o.eid != eid for o in occs):
            return None
        types = {o.component.ctype for o in occs}
        cost = BASE_STEP
        if ComponentType.FIRE in types and not fireproof:
            cost += FIRE_PENALTY
        if ComponentType.WATER in types and not does_float:
            cost += WATER_PENALTY
        return cost

    dist: dict[tuple[int, int], float] = {start: 0.0}
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    while pq:
        d, cell = heapq.heappop(pq)
        if cell == target:
            return d
        if d > dist.get(cell, float("inf")):
            continue
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            c = cell_cost(nx, ny)
            if c is None:
                continue
            nd = d + c
            if nd < dist.get((nx, ny), float("inf")):
                dist[(nx, ny)] = nd
                heapq.heappush(pq, (nd, (nx, ny)))
    return None
