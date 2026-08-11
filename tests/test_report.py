from pecei.infra import Result, Trace, TraceEvent, build_report
from pecei.infra.report import detect_stuck
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
    assert snap.stuck is None          # no trace given -> no stuck diagnosis


def test_report_serializes_to_dict():
    rep = build_report(_world(), Result.ROUND_LIMIT_EXCEED, goal=(4, 0),
                       yielded=[{"anchor": [1, 0]}], round=50, round_budget=50)
    d = rep.model_dump()
    assert d["result"] == "ROUND_LIMIT_EXCEED"
    assert d["failure_snapshot"]["complexity"] == 3.0
    assert d["yielded"] == [{"anchor": [1, 0]}]


def _spin_trace() -> Trace:
    """A corner-loop: the ego revisits the same two poses over and over."""
    t = Trace()
    for i in range(12):
        # alternates between two poses -> distinct count 2, far below the window
        pose = ((4, 2), "EAST") if i % 2 == 0 else ((4, 2), "SOUTH")
        t.append(TraceEvent(round=i + 1, actor="ego", action="TURNRIGHT",
                            moved=False, blocked=True,
                            anchor_after=pose[0], orientation_after=pose[1]))
    return t


def _marching_trace() -> Trace:
    """A freely-moving ego visits a new cell every round."""
    t = Trace()
    for i in range(12):
        t.append(TraceEvent(round=i + 1, actor="ego", action="FORWARD",
                            moved=True, blocked=False,
                            anchor_after=(i + 1, 0), orientation_after="EAST"))
    return t


def test_detect_stuck_flags_spinning():
    note = detect_stuck(_spin_trace())
    assert note is not None
    assert "stuck" in note


def test_detect_stuck_ignores_free_movement():
    assert detect_stuck(_marching_trace()) is None
    # too short to judge
    short = Trace()
    short.append(TraceEvent(round=1, actor="ego", action="FORWARD",
                            moved=True, blocked=False,
                            anchor_after=(1, 0), orientation_after="EAST"))
    assert detect_stuck(short) is None


def test_build_report_includes_stuck_when_trace_spinning():
    rep = build_report(_world(), Result.ROUND_LIMIT_EXCEED, goal=(4, 0),
                       yielded=[], round=12, round_budget=12, trace=_spin_trace())
    assert rep.failure_snapshot is not None
    assert rep.failure_snapshot.stuck is not None
    assert "stuck" in rep.failure_snapshot.stuck
