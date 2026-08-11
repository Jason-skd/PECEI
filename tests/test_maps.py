"""Map validity tests: the four game maps are solvable and non-trivial.

A state-space search over (anchor, orientation, burning_left, remaining-wood)
proves each map is winnable — including the fire+wood map, whose solution
requires the ego to ignite and burn through a wood wall (terrain is *not*
static, so the state must carry burning + the live wood set). A naive
"always go straight" probe must fail on the non-baseline maps (02+). If a map
regresses (becomes unsolvable or trivially solvable), these tests catch it.
"""
import collections
from pathlib import Path

from pecei.world.component import ComponentType
from pecei.world import ActionType, apply_action, load_world

REPO = Path(__file__).resolve().parents[1]
MAPS = REPO / "src" / "pecei" / "maps"

CORRIDOR = MAPS / "01_corridor.yaml"
ONE_WALL = MAPS / "02_one_wall.yaml"
MAZE = MAPS / "03_maze.yaml"
FIRE_WOOD = MAPS / "04_fire_wood.yaml"


def _at_goal(world, eid):
    return bool(world.goal) and any(c == tuple(world.goal) for c in world.entity(eid).abs_cells())


def _wood_cells(world) -> frozenset[tuple[int, int]]:
    """Live (x, y) cells holding a wood component."""
    out: set[tuple[int, int]] = set()
    for x, y, occs in world.grid.cells():
        for o in occs:
            if o.component.ctype is ComponentType.WOOD:
                out.add((x, y))
    return frozenset(out)


def _restore_state(path, anchor, orient, burning_left, remaining_wood):
    """Rebuild a world at an arbitrary search state.

    Re-seeds the ego's (anchor, orient), its burning countdown, and destroys any
    wood not in ``remaining_wood`` (already burned away earlier in this path).
    """
    w = load_world(path)
    e = w.ego
    w.grid.remove(e.eid)
    e.anchor, e.orientation = anchor, orient
    w.ego_status.burning = burning_left > 0
    w.ego_status.burning_left = burning_left
    init_wood = _wood_cells(load_world(path))
    for (wx, wy) in (init_wood - remaining_wood):
        for occ in list(w.grid.occupants(wx, wy)):
            if occ.component.ctype is not ComponentType.WOOD:
                continue
            ent = w.entities.get(occ.eid)
            if ent is None:
                continue
            del ent.components[occ.local]
            w.grid.remove(ent.eid)
            if ent.components:
                w.grid.place(ent)
            else:
                w.entities.pop(ent.eid, None)
    w.grid.place(e)
    return w


def _solve(path, max_depth=300):
    """BFS over ego state (anchor, orient, burning_left, remaining-wood)."""
    w0 = load_world(path)
    init_wood = _wood_cells(w0)
    start = (w0.ego.anchor, w0.ego.orientation, 0, init_wood)
    if _at_goal(w0, w0.ego.eid):
        return []
    q = collections.deque([(start, [])])
    seen = {start}
    while q:
        (anchor, orient, bl, rwood), seq = q.popleft()
        if len(seq) >= max_depth:
            continue
        for a in ActionType:
            w = _restore_state(path, anchor, orient, bl, rwood)
            eid = w.ego.eid
            res = apply_action(w, eid, a)
            if res.failed:  # brittle death, etc.
                continue
            w.tick_environment()
            new_bl = w.ego_status.burning_left if w.ego_status.burning else 0
            new_rwood = _wood_cells(w)
            nstate = (w.entity(eid).anchor, w.entity(eid).orientation, new_bl, new_rwood)
            nseq = seq + [a.value]
            if _at_goal(w, eid):
                return nseq
            changed = (
                nstate[:2] != (anchor, orient)
                or new_rwood != rwood
                or new_bl != bl
            )
            if changed and nstate not in seen:
                seen.add(nstate)
                q.append((nstate, nseq))
    return None


def _naive_straight(path):
    """Always FORWARD; True if it reaches the goal (i.e. trivially solvable)."""
    w = load_world(path)
    eid = w.ego.eid
    for _ in range(w.grid.width * w.grid.height + 5):
        if _at_goal(w, eid):
            return True
        apply_action(w, eid, ActionType.FORWARD)
        w.tick_environment()
    return _at_goal(w, eid)


def test_all_four_maps_are_solvable():
    for m in (CORRIDOR, ONE_WALL, MAZE, FIRE_WOOD):
        sol = _solve(str(m))
        assert sol is not None, f"{m.name} should be solvable"


def test_harder_maps_are_not_trivially_solvable():
    assert _naive_straight(str(CORRIDOR)) is True      # baseline: straight east works
    for m in (ONE_WALL, MAZE, FIRE_WOOD):
        assert _naive_straight(str(m)) is False, f"{m.name} should block naive-straight"


def test_maps_have_ego_and_goal():
    for m in (CORRIDOR, ONE_WALL, MAZE, FIRE_WOOD):
        w = load_world(str(m))
        assert w.ego is not None and w.goal is not None


def test_maps_use_three_cell_body():
    """All four game maps use the 3-cell rigid ego (wheel/metal/brain)."""
    for m in (CORRIDOR, ONE_WALL, MAZE, FIRE_WOOD):
        w = load_world(str(m))
        ctypes = {c.ctype for c in w.ego.components.values()}
        assert ctypes == {ComponentType.WHEEL, ComponentType.METAL, ComponentType.BRAIN}, (
            f"{m.name} ego should be wheel/metal/brain, got {ctypes}"
        )
