"""Primitive world actions: the vocabulary of ``act(...)``.

World-layer mutations: move/turn a rigid Entity. The DSL interpreter (M4) calls
these via host callbacks injected by the engine; the engine drives one action
per round. FORWARD/BACKWARD translate the whole rigid body; TURNLEFT/TURNRIGHT
rotate orientation about the anchor. Collision (solid cell or out of bounds)
blocks the move and leaves the entity unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .entity import Direction, Entity
from .world import World


class ActionType(str, Enum):
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    TURNLEFT = "TURNLEFT"
    TURNRIGHT = "TURNRIGHT"


@dataclass(frozen=True)
class ActionResult:
    action: ActionType
    moved: bool    # the entity's placement changed
    blocked: bool  # a collision (solid cell or bounds) prevented the move


def apply_action(world: World, eid: str, action: ActionType) -> ActionResult:
    ent = world.entity(eid)
    ax, ay = ent.anchor
    if action is ActionType.FORWARD:
        dx, dy = ent.orientation.delta
        ok = _try_commit(world, ent, (ax + dx, ay + dy), ent.orientation)
    elif action is ActionType.BACKWARD:
        dx, dy = ent.orientation.delta
        ok = _try_commit(world, ent, (ax - dx, ay - dy), ent.orientation)
    elif action is ActionType.TURNLEFT:
        ok = _try_commit(world, ent, ent.anchor, ent.orientation.rotate(-1))
    elif action is ActionType.TURNRIGHT:
        ok = _try_commit(world, ent, ent.anchor, ent.orientation.rotate(1))
    else:  # pragma: no cover
        raise ValueError(f"unknown action {action!r}")
    return ActionResult(action=action, moved=ok, blocked=not ok)


def _try_commit(
    world: World, ent: Entity, new_anchor: tuple[int, int], new_orient: Direction
) -> bool:
    """Move ``ent`` to (new_anchor, new_orient) iff every cell is in-bounds and
    not blocked by another solid entity. Otherwise leave it unchanged."""
    grid = world.grid
    old_anchor, old_orient = ent.anchor, ent.orientation
    grid.remove(ent.eid)
    ent.anchor, ent.orientation = new_anchor, new_orient
    cells = ent.placements()
    ok = all(grid.in_bounds(x, y) and not grid.is_blocked(x, y) for (x, y) in cells)
    if not ok:
        ent.anchor, ent.orientation = old_anchor, old_orient
    grid.place(ent)
    return ok
