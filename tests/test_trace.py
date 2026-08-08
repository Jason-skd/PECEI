from pathlib import Path

from pecei.engine import RoundEngine
from pecei.infra import Result, Trace, TraceEvent, build_report
from pecei.observation import observe
from pecei.world import Component, Entity, World, load_world
from pecei.world.actions import ActionType

REPO = Path(__file__).resolve().parents[1]
EXAMPLE02 = REPO / "src" / "pecei" / "maps" / "example02.yaml"


def _world():
    w = World.empty(3, 1, goal=(2, 0))
    w.add(Entity(eid="ego", components={(0, 0): Component.of("brain")}, anchor=(0, 0), is_ego=True))
    return w


def test_trace_jsonl_roundtrip(tmp_path):
    w = _world()
    trace = Trace()
    trace.append(TraceEvent(round=1, actor="ego", action="FORWARD", moved=True, blocked=False,
                            anchor_after=(1, 0), orientation_after="EAST",
                            observation=observe(w, "ego").to_dict()))
    trace.append(TraceEvent(round=2, actor="ego", action="FORWARD", moved=True, blocked=False,
                            anchor_after=(2, 0), orientation_after="EAST",
                            observation=observe(w, "ego").to_dict()))
    out = tmp_path / "t.jsonl"
    trace.write(out)

    loaded = Trace.read(out)
    assert len(loaded.events) == 2
    assert loaded.events[0].action == "FORWARD"
    assert loaded.events[1].anchor_after == (2, 0)
    assert loaded.events[0].observation is not None


def test_trace_llm_slots_are_present_and_none_by_default():
    ev = TraceEvent(round=1)
    d = ev.model_dump()
    assert "llm_request" in d and d["llm_request"] is None
    assert "llm_response" in d and d["llm_response"] is None


def test_episode_produces_trace_and_success_report(tmp_path):
    world = load_world(EXAMPLE02)
    ego = world.ego
    eng = RoundEngine(world, round_budget=50)
    trace = Trace()

    def on_round(r, eid, action, res):
        e = world.entity(eid)
        trace.append(TraceEvent(
            round=r, actor=eid, action=action.value, moved=res.moved, blocked=res.blocked,
            anchor_after=e.anchor, orientation_after=e.orientation.value,
            observation=observe(world, eid).to_dict(),
        ))

    eng.on_round = on_round
    for _ in range(3):
        eng.apply(ego.eid, ActionType.FORWARD)
    assert eng.at_goal(ego.eid)

    out = tmp_path / "ep.jsonl"
    trace.write(out)
    loaded = Trace.read(out)
    assert len(loaded.events) == 3
    assert loaded.events[-1].anchor_after == (6, 2)

    rep = build_report(world, Result.SUCCESS, goal=world.goal, yielded=[],
                       round=eng.round, round_budget=eng.round_budget)
    assert rep.result is Result.SUCCESS
    assert rep.failure_snapshot is None
