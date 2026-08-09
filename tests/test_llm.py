"""LLM author-layer tests: protocol shape (feedback, no live obs) + rendering."""
import dataclasses
import json

import pytest

from pecei.action import CompileError
from pecei.infra import Result
from pecei.llm import Feedback, MockProvider
from pecei.llm.protocol import Directive, TurnInput, TurnOutput, parse_program
from pecei.llm.prompt import render_user


def test_mock_decide_returns_complete_script():
    out = MockProvider().decide(TurnInput())
    assert out.program is not None
    kinds = [s.kind for s in out.program.body]
    assert "assign" in kinds and "expr" in kinds   # observe + acts/yields


def test_turninput_shape_is_feedback_not_live_observation():
    fields = {f.name for f in dataclasses.fields(TurnInput)}
    assert {"map_desc", "feedback", "snowball", "instructions"} <= fields
    assert "observation" not in fields              # live per-round observation removed


def test_render_user_shows_static_map_feedback_and_instructions():
    turn = TurnInput(
        instructions="experiment session 2 of 4: map 'maze'",
        map_desc={"width": 6, "height": 5, "goal": [6, 2],
                  "ego": {"anchor": [3, 2], "orientation": "EAST"}, "entities": []},
        feedback=Feedback(stop_reason=Result.SCRIPT_ENDED, rounds_used=1),
    )
    out = render_user(turn)
    assert "DIRECTIVE: PLAN" in out
    assert "INSTRUCTIONS" in out and "2 of 4" in out
    assert "map:" in out and "goal [6, 2]" in out
    assert "last_cycle stopped: SCRIPT_ENDED" in out
    assert "Visible cells" not in out               # no live observation leak


def test_parse_program_raises_on_malformed_tool_call():
    # Regression: the model once conflated beat(YIELD, ...) with act(...),
    # emitting act(YIELD). parse_program must raise CompileError (carrying the
    # bad location) so the provider/runner can surface it as COMPILE_ERROR.
    bad = {"body": [{"kind": "expr", "expr": {"kind": "act", "action": "YIELD"}}]}
    with pytest.raises(CompileError) as ei:
        parse_program(bad)
    assert "act.action" in str(ei.value) or "action" in str(ei.value)

    # valid movement is accepted; both dict and JSON-string payloads work
    good = {"body": [{"kind": "expr", "expr": {"kind": "act", "action": "FORWARD"}}]}
    assert parse_program(good) is not None
    assert parse_program(json.dumps(good)) is not None

    # malformed JSON string (OpenAI argument path) also raises CompileError
    with pytest.raises(CompileError):
        parse_program("{not json")


def test_render_user_shows_compile_error_in_feedback_and_snowball():
    # feedback path
    turn = TurnInput(
        feedback=Feedback(
            stop_reason=Result.COMPILE_ERROR, rounds_used=0,
            compile_error="at body.7.expr.expr.act.action: bad action 'YIELD'",
        ),
    )
    out = render_user(turn)
    assert "COMPILE_ERROR" in out
    assert "compile error:" in out and "YIELD" in out

    # snowball path
    turn = TurnInput(snowball=[{
        "index": 1, "stop_reason": "COMPILE_ERROR", "rounds": 0,
        "scripts": [], "error": "if condition must be a bool variable",
    }])
    out = render_user(turn)
    assert "compile error:" in out and "bool variable" in out
