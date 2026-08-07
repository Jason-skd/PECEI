from pathlib import Path

import pytest

from pecei.engine import RoundEngine
from pecei.world import ActionType, load_world

REPO = Path(__file__).resolve().parents[1]
LEVEL = REPO / "src" / "pecei" / "maps" / "example02.yaml"


def test_action_sequence_reaches_goal():
    world = load_world(LEVEL)
    assert world.goal == (6, 2)
    ego = world.ego
    assert ego is not None
    eng = RoundEngine(world, round_budget=50)

    # ego faces EAST at anchor (3,2); goal at (6,2): 3 FORWARDs
    for _ in range(3):
        r = eng.apply(ego.eid, ActionType.FORWARD)
        assert r.moved and not r.blocked

    assert eng.at_goal(ego.eid)
    assert eng.round == 3


def test_blocked_move_at_boundary():
    world = load_world(LEVEL)
    ego = world.ego
    eng = RoundEngine(world, round_budget=50)

    for _ in range(3):
        eng.apply(ego.eid, ActionType.FORWARD)
    assert eng.at_goal(ego.eid)  # wheel cell now on the goal

    # one more EAST is out of bounds -> blocked, no movement, still on goal
    r = eng.apply(ego.eid, ActionType.FORWARD)
    assert r.blocked and not r.moved
    assert eng.at_goal(ego.eid)


def test_round_budget_exhausts():
    world = load_world(LEVEL)
    ego = world.ego
    eng = RoundEngine(world, round_budget=2)

    eng.apply(ego.eid, ActionType.TURNLEFT)   # EAST -> NORTH (body fits in rows)
    eng.apply(ego.eid, ActionType.TURNRIGHT)  # NORTH -> EAST
    assert eng.round == 2 and eng.time_exceeded

    with pytest.raises(RuntimeError):
        eng.apply(ego.eid, ActionType.FORWARD)
