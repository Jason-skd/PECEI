"""Observation: a derived, leak-free view of the world from one observer.

Conical FOV: cells within ``range`` (Chebyshev) and ``half_angle`` (degrees) of
the observer's facing, with line-of-sight occlusion by solid cells (Bresenham).
The observer always perceives its own body. All data is copied into frozen
snapshots, so an Observation never leaks live ground-truth references.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pecei.world.component import ComponentType
from pecei.world.entity import Direction, Entity
from pecei.world.grid import Grid
from pecei.world.world import World

DEFAULT_RANGE = 5
DEFAULT_HALF_ANGLE = 45.0


@dataclass(frozen=True)
class CellView:
    """A frozen snapshot of one perceived cell (may stack multiple occupants)."""
    ctypes: tuple[ComponentType, ...]
    eids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"types": [t.value for t in self.ctypes], "eids": list(self.eids)}


@dataclass(frozen=True)
class Observation:
    # relative (dx, dy) from the observer's anchor -> perceived cell
    cells: dict[tuple[int, int], CellView]
    anchor: tuple[int, int]
    orientation: Direction
    vision_range: int
    half_angle: float

    def at(self, dx: int, dy: int) -> CellView | None:
        """Perceived cell at relative offset (dx, dy), or None if unseen."""
        return self.cells.get((dx, dy))

    def to_dict(self) -> dict:
        return {
            "anchor": list(self.anchor),
            "orientation": self.orientation.value,
            "vision_range": self.vision_range,
            "half_angle": self.half_angle,
            "cells": {f"{dx},{dy}": cv.to_dict() for (dx, dy), cv in self.cells.items()},
        }


def observe(
    world: World,
    observer_eid: str,
    *,
    vision_range: int | None = None,
    half_angle: float | None = None,
) -> Observation:
    """Build an Observation for ``observer_eid`` (conical FOV from its anchor)."""
    ent = world.entity(observer_eid)
    ox, oy = ent.anchor
    rng = vision_range or ent.vision_range or DEFAULT_RANGE
    ha = (
        half_angle
        if half_angle is not None
        else (ent.half_angle if ent.half_angle is not None else DEFAULT_HALF_ANGLE)
    )

    grid = world.grid
    visible = _visible_cells(grid, (ox, oy), ent.orientation, rng, ha)

    cells: dict[tuple[int, int], CellView] = {}
    for (x, y) in visible:
        cells[(x - ox, y - oy)] = _view(grid, x, y)
    # the observer always perceives its own body, even outside the cone
    for (x, y) in ent.placements():
        cells.setdefault((x - ox, y - oy), _view(grid, x, y))

    return Observation(
        cells=cells,
        anchor=(ox, oy),
        orientation=ent.orientation,
        vision_range=rng,
        half_angle=ha,
    )


def _view(grid: Grid, x: int, y: int) -> CellView:
    occs = grid.occupants(x, y)
    return CellView(
        ctypes=tuple(o.component.ctype for o in occs),
        eids=tuple(o.eid for o in occs),
    )


def _visible_cells(
    grid: Grid, origin: tuple[int, int], facing: Direction, rng: int, half_angle: float
) -> set[tuple[int, int]]:
    ox, oy = origin
    out: set[tuple[int, int]] = {origin}
    for (x, y, _occs) in grid.cells():
        if (x, y) == origin:
            continue
        dx, dy = x - ox, y - oy
        if max(abs(dx), abs(dy)) > rng:
            continue
        if not _in_cone((dx, dy), facing, half_angle):
            continue
        if _line_of_sight(grid, origin, (x, y)):
            out.add((x, y))
    return out


def _in_cone(v: tuple[int, int], facing: Direction, half_angle: float) -> bool:
    fx, fy = facing.delta
    dx, dy = v
    cross = dx * fy - dy * fx
    dot = dx * fx + dy * fy
    angle = abs(math.degrees(math.atan2(cross, dot)))
    return angle <= half_angle + 1e-9


def _line_of_sight(grid: Grid, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Bresenham LOS: True iff no solid cell strictly between ``a`` and ``b``."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        if (x, y) == (x1, y1):
            break
        if grid.in_bounds(x, y) and grid.is_blocked(x, y):
            return False
    return True
