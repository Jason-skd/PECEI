"""Observation: a derived, leak-free, egocentric view from one observer.

The view is a **canonical "camera frame"**: every cell offset is rotated so the
observer's gaze is always +x (as-if-facing east). The agent therefore never sees
its world orientation or absolute position — only a self-centred field of cells,
each carrying the component types it contains (including a ``GOAL`` marker when
the goal cell is in view). Turning changes which world cells land in the frame;
it never relabels the frame's axes.

Conical FOV: cells within ``range`` (Chebyshev) and ``half_angle`` (degrees) of
the observer's facing, with line-of-sight occlusion by solid cells (Bresenham).
The observer always perceives its own body. All data is copied into frozen
snapshots, so an Observation never leaks live ground-truth references.

Same-source contract: ``CellView.is_X`` predicates are pure functions of
``ctypes``, and ``to_dict`` serialises that same ``ctypes``. So the boolean the
interpreter computes at runtime and the types the LLM reads in a yielded
observation are one and the same information — they can never diverge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pecei.world.component import SOLID, ComponentType
from pecei.world.entity import Direction, Entity, _rotate
from pecei.world.grid import Grid
from pecei.world.world import World

DEFAULT_RANGE = 5
DEFAULT_HALF_ANGLE = 45.0

# Rotation (in quarter-turns CW) that maps the observer's facing onto +x (east),
# canonicalising the frame. steps = (1 - ordinal) % 4 (see entity.Direction).
_CANONICAL_STEPS = {
    Direction.NORTH: 1,
    Direction.EAST: 0,
    Direction.SOUTH: 3,
    Direction.WEST: 2,
}

# Returned by ``at()`` for offsets outside the FOV: a safe, all-false cell so
# predicates never crash on unseen cells. An unseen cell reports is_empty=True
# and is_blocked=False (the agent may walk into an unseen wall, but feels the
# collision via ``act()`` returning moved=False; FOV range >> step size).
_EMPTY_CELL = None  # lazily built (CellView is defined below)


@dataclass(frozen=True)
class CellView:
    """A frozen snapshot of one perceived cell (may stack multiple occupants).

    ``ctypes`` is the single source; every ``is_X`` predicate and ``ctype`` is a
    pure derivation from it, so runtime and serialised views are identical.
    """
    ctypes: tuple[ComponentType, ...]
    eids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"types": [t.value for t in self.ctypes], "eids": list(self.eids)}

    def _has(self, t: ComponentType) -> bool:
        return t in self.ctypes

    @property
    def is_empty(self) -> bool:
        return not self.ctypes

    @property
    def is_blocked(self) -> bool:
        return any(t in SOLID for t in self.ctypes)

    is_fire = property(lambda self: self._has(ComponentType.FIRE))
    is_water = property(lambda self: self._has(ComponentType.WATER))
    is_stone = property(lambda self: self._has(ComponentType.STONE))
    is_wood = property(lambda self: self._has(ComponentType.WOOD))
    is_metal = property(lambda self: self._has(ComponentType.METAL))
    is_wheel = property(lambda self: self._has(ComponentType.WHEEL))
    is_brain = property(lambda self: self._has(ComponentType.BRAIN))
    is_goal = property(lambda self: self._has(ComponentType.GOAL))

    @property
    def ctype(self) -> str:
        if self.ctypes:
            return self.ctypes[-1].value
        return "empty"


@dataclass(frozen=True)
class Observation:
    # Canonical (dx, dy) from the observer's anchor -> perceived cell. The frame
    # is rotated so +x is always the observer's gaze; (0,0) is the anchor cell
    # itself (the leading cell). No anchor/orientation is exposed: the agent
    # works purely in its own camera frame.
    cells: dict[tuple[int, int], CellView]
    vision_range: int
    half_angle: float

    def at(self, dx: int, dy: int) -> CellView:
        """Perceived cell at canonical offset (dx, dy).

        Returns an empty CellView for unseen offsets, so predicates are always
        safe to evaluate.
        """
        return self.cells.get((dx, dy)) or _empty_cell()

    def to_dict(self) -> dict:
        return {
            "vision_range": self.vision_range,
            "half_angle": self.half_angle,
            "cells": {f"{dx},{dy}": cv.to_dict() for (dx, dy), cv in self.cells.items()},
        }


def _empty_cell() -> CellView:
    global _EMPTY_CELL
    if _EMPTY_CELL is None:
        _EMPTY_CELL = CellView(ctypes=(), eids=())
    return _EMPTY_CELL


def observe(
    world: World,
    observer_eid: str,
    *,
    vision_range: int | None = None,
    half_angle: float | None = None,
) -> Observation:
    """Build an egocentric Observation for ``observer_eid``.

    FOV cone is computed in world space (pointed along the entity's facing), then
    every visible offset is rotated into the canonical camera frame (+x = gaze).
    The observer always perceives its own body.
    """
    ent = world.entity(observer_eid)
    ox, oy = ent.anchor
    rng = vision_range or ent.vision_range or DEFAULT_RANGE
    ha = (
        half_angle
        if half_angle is not None
        else (ent.half_angle if ent.half_angle is not None else DEFAULT_HALF_ANGLE)
    )

    steps = _CANONICAL_STEPS[ent.orientation]
    grid = world.grid
    visible = _visible_cells(grid, (ox, oy), ent.orientation, rng, ha)

    cells: dict[tuple[int, int], CellView] = {}
    for (x, y) in visible:
        cells[_rotate((x - ox, y - oy), steps)] = _view(grid, x, y)
    # the observer always perceives its own body, even outside the cone
    for (x, y) in ent.placements():
        cells.setdefault(_rotate((x - ox, y - oy), steps), _view(grid, x, y))

    return Observation(cells=cells, vision_range=rng, half_angle=ha)


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
