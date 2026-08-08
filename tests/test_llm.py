"""LLM author-layer tests: protocol shape (feedback, no live obs) + rendering."""
import dataclasses

from pecei.infra import Result
from pecei.llm import Feedback, MockProvider
from pecei.llm.protocol import Directive, TurnInput
from pecei.llm.prompt import render_user


def test_mock_decide_returns_complete_script():
    out = MockProvider().decide(TurnInput())
    assert out.program is not None
    kinds = [s.kind for s in out.program.body]
    assert "assign" in kinds and "expr" in kinds   # observe + acts/yields


def test_turninput_shape_is_feedback_not_live_observation():
    fields = {f.name for f in dataclasses.fields(TurnInput)}
    assert {"map_desc", "feedback", "snowball"} <= fields
    assert "observation" not in fields              # live per-round observation removed


def test_render_user_shows_static_map_and_feedback_not_live_obs():
    turn = TurnInput(
        map_desc={"width": 6, "height": 5, "goal": [6, 2],
                  "ego": {"anchor": [3, 2], "orientation": "EAST"}, "entities": []},
        feedback=Feedback(stop_reason=Result.SCRIPT_ENDED, rounds_used=1),
    )
    out = render_user(turn)
    assert "DIRECTIVE: PLAN" in out
    assert "map:" in out and "goal [6, 2]" in out
    assert "last_cycle stopped: SCRIPT_ENDED" in out
    assert "Visible cells" not in out               # no live observation leak
