from pathlib import Path

import pytest
from pydantic import ValidationError

from pecei.action import (
    Act,
    Assign,
    Attr,
    Beat,
    BeatOp,
    BoolOp,
    Compare,
    CompileError,
    ExprStmt,
    Host,
    If,
    Interpreter,
    Lit,
    NavObs,
    Program,
    Var,
    pretty,
    type_check,
)
from pecei.engine import RoundEngine
from pecei.observation import observe
from pecei.world import Component, Direction, Entity, World
from pecei.world.actions import ActionType

REPO = Path(__file__).resolve().parents[1]
EXAMPLE02 = REPO / "src" / "pecei" / "maps" / "example02.yaml"


# ---------- fixtures ----------

def _eye_world_with_wall():
    """ego facing EAST at (2,2) with a wall directly east (front blocked)."""
    w = World.empty(5, 5)
    w.add(Entity(eid="ego", components={(0, 0): Component.of("wheel"),
                                        (0, 1): Component.of("metal"),
                                        (0, 2): Component.of("brain")},
                 anchor=(2, 2), orientation=Direction.EAST, is_ego=True))
    w.add(Entity(eid="wall", components={(0, 0): Component.of("stone")}, anchor=(3, 2)))
    return w


def _make_host(world, ego_eid, budget=50):
    eng = RoundEngine(world, round_budget=budget)
    yielded = []
    host = Host(
        act=lambda a: eng.apply(ego_eid, a).moved,
        observe=lambda: NavObs(observe(world, ego_eid), world.goal),
        yield_=lambda v: yielded.append(v),
    )
    return eng, host, yielded


def _sense_front_program():
    """ob = beat(OBSERVE); blocked = ob.front.is_blocked; if blocked: turn else: forward"""
    return Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        Assign(name="blocked", expr=Attr(obj=Attr(obj=Var(name="ob"), attr="front"), attr="is_blocked")),
        If(test="blocked",
           then=[ExprStmt(expr=Act(action=ActionType.TURNRIGHT))],
           orelse=[ExprStmt(expr=Act(action=ActionType.FORWARD))]),
    ])


# ---------- interpreter integration ----------

def test_program_turns_when_front_blocked():
    world = _eye_world_with_wall()
    eng, host, _ = _make_host(world, "ego")
    ego = world.ego
    assert ego.orientation is Direction.EAST
    Interpreter(host).run(_sense_front_program())
    # front was blocked -> TURNRIGHT (EAST -> SOUTH), anchor unchanged
    assert ego.orientation is Direction.SOUTH
    assert ego.anchor == (2, 2)


def test_program_advances_when_front_clear():
    from pecei.world import load_world
    world = load_world(EXAMPLE02)  # ego EAST at (3,2), front (4,2) clear
    eng, host, _ = _make_host(world, world.ego.eid)
    ego = world.ego
    anchor_before = ego.anchor
    Interpreter(host).run(_sense_front_program())
    assert ego.anchor == (anchor_before[0] + 1, anchor_before[1])  # moved EAST


def test_yield_writes_observation():
    world = _eye_world_with_wall()
    eng, host, yielded = _make_host(world, "ego")
    prog = Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        ExprStmt(expr=Beat(op=BeatOp.YIELD, value=Var(name="ob"))),
    ])
    Interpreter(host).run(prog)
    assert len(yielded) == 1
    assert isinstance(yielded[0], NavObs)


def test_act_round_budget_propagates():
    world = _eye_world_with_wall()
    eng, host, _ = _make_host(world, "ego", budget=1)
    prog = Program(body=[
        ExprStmt(expr=Act(action=ActionType.TURNRIGHT)),
        ExprStmt(expr=Act(action=ActionType.TURNRIGHT)),  # 2nd exhausts budget
    ])
    with pytest.raises(RuntimeError):
        Interpreter(host).run(prog)


# ---------- compile-time rules ----------

def test_if_requires_bool_variable():
    bad = Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        If(test="ob", then=[], orelse=[]),  # ob is obs, not bool
    ])
    with pytest.raises(CompileError):
        type_check(bad)


def test_if_test_is_structurally_a_name_not_expression():
    # The schema makes If.test a string; an object/expression there is rejected.
    with pytest.raises(ValidationError):
        If.model_validate({"test": {"kind": "var", "name": "x"}, "then": []})


def test_observe_must_be_assigned():
    bad = Program(body=[ExprStmt(expr=Beat(op=BeatOp.OBSERVE))])  # bare observe, not caught
    with pytest.raises(CompileError):
        type_check(bad)


def test_bool_op_and_compare_typecheck():
    prog = Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        Assign(name="b1", expr=Attr(obj=Attr(obj=Var(name="ob"), attr="front"), attr="is_fire")),
        Assign(name="b2", expr=Attr(obj=Attr(obj=Var(name="ob"), attr="left"), attr="is_water")),
        Assign(name="b", expr=BoolOp(op="or", operands=[Var(name="b1"), Var(name="b2")])),
        Assign(name="c", expr=Compare(op="==",
                                      left=Attr(obj=Attr(obj=Var(name="ob"), attr="here"), attr="ctype"),
                                      right=Lit(value="brain"))),
        If(test="b", then=[], orelse=[]),
    ])
    type_check(prog)  # should not raise


# ---------- schema + pretty ----------

def test_json_schema_generated():
    schema = Program.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema or "$defs" in schema


def test_pretty_renders_text_dsl():
    prog = Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        If(test="blocked", then=[ExprStmt(expr=Act(action=ActionType.FORWARD))], orelse=[]),
    ])
    text = pretty(prog)
    assert "ob = beat(OBSERVE)" in text
    assert "if blocked:" in text
    assert "act(FORWARD)" in text
