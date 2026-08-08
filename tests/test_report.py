from pecei.infra import Result, build_report
from pecei.world import Component, Entity, World


def _world():
    w = World.empty(5, 1, goal=(4, 0))
    w.add(Entity(eid="ego", components={(0, 0): Component.of("brain")},
                 anchor=(1, 0), is_ego=True))
    return w


def test_success_report_has_no_snapshot():
    rep = build_report(_world(), Result.SUCCESS, goal=(4, 0),
                       yielded=[], round=4, round_budget=50)
    assert rep.result is Result.SUCCESS
    assert rep.failure_snapshot is None


def test_failure_report_has_snapshot_and_complexity():
    rep = build_report(_world(), Result.ROUND_LIMIT_EXCEED, goal=(4, 0),
                       yielded=[], round=50, round_budget=50)
    assert rep.result is Result.ROUND_LIMIT_EXCEED
    snap = rep.failure_snapshot
    assert snap is not None
    assert snap.pos == (1, 0)
    assert snap.complexity == 3.0  # ego at (1,0) -> goal (4,0)
    assert snap.current_state["eid"] == "ego"


def test_report_serializes_to_dict():
    rep = build_report(_world(), Result.ROUND_LIMIT_EXCEED, goal=(4, 0),
                       yielded=[{"anchor": [1, 0]}], round=50, round_budget=50)
    d = rep.model_dump()
    assert d["result"] == "ROUND_LIMIT_EXCEED"
    assert d["failure_snapshot"]["complexity"] == 3.0
    assert d["yielded"] == [{"anchor": [1, 0]}]
