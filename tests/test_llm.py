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


def test_turninput_shape_is_seed_observation_not_live_or_godview():
    fields = {f.name for f in dataclasses.fields(TurnInput)}
    assert {"seed_observation", "feedback", "snowball", "instructions"} <= fields
    assert "map_desc" not in fields                 # god-view map removed
    assert "observation" not in fields              # no LIVE per-round observation field


def test_render_user_shows_seed_observation_feedback_and_instructions():
    turn = TurnInput(
        instructions="experiment session 2 of 4: map 'maze'",
        seed_observation={
            "vision_range": 5, "half_angle": 45.0,
            "cells": {"1,0": {"types": ["stone"]}, "5,0": {"types": ["goal"]}, "1,1": {"types": []}},
        },
        feedback=Feedback(stop_reason=Result.SCRIPT_ENDED, rounds_used=1),
    )
    out = render_user(turn)
    assert "DIRECTIVE: PLAN" in out
    assert "INSTRUCTIONS" in out and "2 of 4" in out
    assert "goal:" not in out                              # no absolute goal coordinate
    assert "seed observation:" in out and "+x = your gaze" in out
    assert "1,0:stone" in out and "5,0:goal" in out        # canonical cells rendered (goal as a type)
    assert "layout" not in out and "map:" not in out       # god-view layout gone
    assert "facing" not in out and "EAST" not in out       # no orientation/compass leak
    assert "last_cycle stopped: SCRIPT_ENDED" in out


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
    # feedback path — the previous script rides along (B-layer: parsed OK, type error)
    turn = TurnInput(
        feedback=Feedback(
            stop_reason=Result.COMPILE_ERROR, rounds_used=0,
            script="act(YIELD)",
            compile_error="at body.7.expr.expr.act.action: bad action 'YIELD'",
        ),
    )
    out = render_user(turn)
    assert "COMPILE_ERROR" in out
    assert "compile error:" in out and "YIELD" in out
    assert "script:" in out and "act(YIELD)" in out    # last cycle's script in feedback

    # snowball path
    turn = TurnInput(snowball=[{
        "index": 1, "stop_reason": "COMPILE_ERROR", "rounds": 0,
        "scripts": [], "error": "if condition must be a bool variable",
    }])
    out = render_user(turn)
    assert "compile error:" in out and "bool variable" in out
