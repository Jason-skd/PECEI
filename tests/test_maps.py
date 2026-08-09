"""Map validity tests: the four MVP maps are solvable and non-trivial.

A BFS over (anchor, orientation) states proves each map is winnable; a naive
"always go straight" probe must fail on the non-baseline maps (02+). If a map
regresses (becomes unsolvable or trivially solvable), these tests catch it.
"""
import collections
from pathlib import Path

from pecei.world import ActionType, apply_action, load_world

REPO = Path(__file__).resolve().parents[1]
MAPS = REPO / "src" / "pecei" / "maps"

CORRIDOR = MAPS / "01_corridor.yaml"
ONE_WALL = MAPS / "02_one_wall.yaml"
WATER = MAPS / "03_water_obstacle.yaml"
MAZE = MAPS / "04_maze.yaml"


def _at_goal(world, eid):
    return bool(world.goal) and any(c == tuple(world.goal) for c in world.entity(eid).abs_cells())


def _set_state(path, anchor, orient):
    w = load_world(path)
    e = w.entity(w.ego.eid)
    w.grid.remove(e.eid)
    e.anchor, e.orientation = anchor, orient
    w.grid.place(e)
    return w


def _bfs_solve(path, max_depth=80):
    """BFS over ego (anchor, orientation) states; returns an action sequence or None."""
    w0 = load_world(path)
    start = (w0.ego.anchor, w0.ego.orientation)
    if _at_goal(w0, w0.ego.eid):
        return []
    q = collections.deque([(start, [])])
    seen = {start}
    while q:
        (anchor, orient), seq = q.popleft()
        if len(seq) >= max_depth:
            continue
        for a in ActionType:
            w = _set_state(path, anchor, orient)
            eid = w.ego.eid
            res = apply_action(w, eid, a)
            nstate = (w.entity(eid).anchor, w.entity(eid).orientation)
            nseq = seq + [a]
            if _at_goal(w, eid):
                return nseq
            if res.moved and nstate not in seen:
                seen.add(nstate)
                q.append((nstate, nseq))
    return None


def _naive_straight(path):
    """Always FORWARD; returns True if it reaches the goal (i.e. trivially solvable)."""
    w = load_world(path)
    eid = w.ego.eid
    for _ in range(w.grid.width * w.grid.height):
        if _at_goal(w, eid):
            return True
        apply_action(w, eid, ActionType.FORWARD)
    return False


def test_all_four_maps_are_solvable():
    for m in (CORRIDOR, ONE_WALL, WATER, MAZE):
        sol = _bfs_solve(str(m))
        assert sol is not None, f"{m.name} should be solvable"


def test_harder_maps_are_not_trivially_solvable():
    assert _naive_straight(str(CORRIDOR)) is True      # baseline: straight east works
    for m in (ONE_WALL, WATER, MAZE):
        assert _naive_straight(str(m)) is False, f"{m.name} should block naive-straight"


def test_maps_have_ego_and_goal():
    for m in (CORRIDOR, ONE_WALL, WATER, MAZE):
        w = load_world(str(m))
        assert w.ego is not None and w.goal is not None
