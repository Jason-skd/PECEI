"""Primitive world actions: the vocabulary of ``act(...)``.

World-layer mutations: move/turn a rigid Entity. The DSL interpreter (M4) calls
these via host callbacks injected by the engine; the engine drives one action
per round. FORWARD/BACKWARD translate the whole rigid body; TURNLEFT/TURNRIGHT
rotate orientation about the anchor. Collision (solid cell or out of bounds)
blocks the move and leaves the entity unchanged.

Interaction rules with terrain/effects:
  - a BRITTLE ego that would touch a ``metal`` cell fails outright
    (``failed=True`` -> Result.BRITTLE_FAILURE, the run stops);
  - a BURNING ego destroys the ``wood`` in the target cells before moving, so
    it can burn through wooden obstacles;
  - rotation is swept-path checked: the cells swept by the body between the old
    and new placements must be clear (no tunneling through a wall).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .component import ComponentType
from .entity import Direction, Entity, _rotate
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
    failed: bool = False  # a fatal interaction (brittle touching metal) stopped the run


def apply_action(world: World, eid: str, action: ActionType) -> ActionResult:
    ent = world.entity(eid)
    ax, ay = ent.anchor
    if action is ActionType.FORWARD:
        dx, dy = ent.orientation.delta
        new_anchor, new_orient = (ax + dx, ay + dy), ent.orientation
    elif action is ActionType.BACKWARD:
        dx, dy = ent.orientation.delta
        new_anchor, new_orient = (ax - dx, ay - dy), ent.orientation
    elif action is ActionType.TURNLEFT:
        new_anchor, new_orient = ent.anchor, ent.orientation.rotate(-1)
    elif action is ActionType.TURNRIGHT:
        new_anchor, new_orient = ent.anchor, ent.orientation.rotate(1)
    else:  # pragma: no cover
        raise ValueError(f"unknown action {action!r}")

    # brittle ego touching metal -> fatal failure (regardless of move success)
    if world.ego_status.brittle and _touches(world, ent, new_anchor, new_orient, ComponentType.METAL):
        return ActionResult(action=action, moved=False, blocked=False, failed=True)

    # burning ego burns through wooden obstacles in the target cells
    if world.ego_status.burning:
        _burn_target_wood(world, ent, new_anchor, new_orient)

    # swept-path check for rotations: the body must not tunnel through a wall
    if action in (ActionType.TURNLEFT, ActionType.TURNRIGHT) and _swept_blocked(
        world, ent, new_orient
    ):
        return ActionResult(action=action, moved=False, blocked=True)

    ok = _try_commit(world, ent, new_anchor, new_orient)
    return ActionResult(action=action, moved=ok, blocked=not ok, failed=False)


def _touches(
    world: World, ent: Entity, new_anchor: tuple[int, int], new_orient: Direction, ctype: ComponentType
) -> bool:
    """True if the entity, placed at (new_anchor, new_orient), overlaps ``ctype``."""
    ax, ay = new_anchor
    steps = new_orient.ordinal()
    for local in ent.components:
        rx, ry = _rotate(local, steps)
        x, y = ax + rx, ay + ry
        if not world.grid.in_bounds(x, y):
            continue
        if any(o.component.ctype is ctype for o in world.grid.occupants(x, y)):
            return True
    return False


def _burn_target_wood(
    world: World, ent: Entity, new_anchor: tuple[int, int], new_orient: Direction
) -> None:
    """Destroy wood components in the target cells before the move commits."""
    ax, ay = new_anchor
    steps = new_orient.ordinal()
    cells = [(ax + rx, ay + ry) for local in ent.components for (rx, ry) in [_rotate(local, steps)]]
    world._burn_wood(cells)


def _swept_blocked(world: World, ent: Entity, new_orient: Direction) -> bool:
    """Rotation swept-path check: midpoint cells between old and new placements
    must be in-bounds and not solid (prevents tunneling through a wall)."""
    old = ent.placements()  # absolute cell -> (local, component)
    delta_steps = (new_orient.ordinal() - ent.orientation.ordinal()) % 4
    for local in ent.components:
        old_cell = next((c for c, (l, _) in old.items() if l == local), None)
        if old_cell is None:
            continue
        rx, ry = _rotate(local, delta_steps)
        new_cell = (ent.anchor[0] + rx, ent.anchor[1] + ry)
        mid = ((old_cell[0] + new_cell[0]) // 2, (old_cell[1] + new_cell[1]) // 2)
        if mid == old_cell or mid == new_cell:
            continue
        if not world.grid.in_bounds(*mid) or world.grid.is_blocked(*mid):
            return True
    return False


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
