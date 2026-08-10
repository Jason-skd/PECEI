from pathlib import Path

import pytest
from pydantic import ValidationError

from pecei.action import (
    Act,
    Assign,
    At,
    Attr,
    Beat,
    BeatOp,
    BoolOp,
    BudgetExceeded,
    Compare,
    CompileError,
    ExprStmt,
    For,
    Host,
    If,
    Interpreter,
    Lit,
    Program,
    Var,
    While,
    pretty,
    type_check,
)
from pecei.engine import RoundEngine
from pecei.observation import Observation, observe
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


def _at(dx: int, dy: int, var: str = "ob") -> At:
    """Build an ``At`` node for the cell at canonical offset (dx, dy)."""
    return At(obj=Var(name=var), dx=Lit(value=dx), dy=Lit(value=dy))


def _make_host(world, ego_eid, budget=50):
    eng = RoundEngine(world, round_budget=budget)
    yielded = []
    host = Host(
        act=lambda a: eng.apply(ego_eid, a).moved,
        observe=lambda: observe(world, ego_eid),
        yield_=lambda v: yielded.append(v),
    )
    return eng, host, yielded


def _sense_front_program():
    """ob = beat(OBSERVE); blocked = ob.at(1,0).is_blocked; if blocked: turn else: forward.

    Canonical frame: +x is the gaze, so (1,0) is the cell directly ahead."""
    return Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        Assign(name="blocked", expr=Attr(obj=_at(1, 0), attr="is_blocked")),
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
    assert isinstance(yielded[0], Observation)


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
        Assign(name="b1", expr=Attr(obj=_at(1, 0), attr="is_fire")),
        Assign(name="b2", expr=Attr(obj=_at(0, -1), attr="is_water")),
        Assign(name="b", expr=BoolOp(op="or", operands=[Var(name="b1"), Var(name="b2")])),
        Assign(name="c", expr=Compare(op="==",
                                      left=Attr(obj=_at(0, 0), attr="ctype"),
                                      right=Lit(value="brain"))),
        If(test="b", then=[], orelse=[]),
    ])
    type_check(prog)  # should not raise


# ---------- schema + pretty ----------

def test_json_schema_generated():
    schema = Program.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema or "$defs" in schema


def test_at_coerces_bare_int_offsets_and_bare_string_obj():
    # The taught surface syntax is `ob.at(5, 0)`; the model often emits bare
    # ints/strings instead of nested {kind:lit/var} wrappers. At must coerce:
    # bare int offset -> Lit, bare string obj -> Var. Regression for the
    # 01_corridor run where these caused 3 COMPILE_ERRORs.
    from pecei.llm.protocol import parse_program
    prog = parse_program({"body": [
        {"kind": "assign", "name": "ob", "expr": {"kind": "beat", "op": "OBSERVE"}},
        {"kind": "assign", "name": "c", "expr": {"kind": "at", "obj": "ob", "dx": 5, "dy": 0}},
        {"kind": "assign", "name": "b", "expr": {"kind": "attr", "obj": {"kind": "var", "name": "c"}, "attr": "is_goal"}},
    ]})
    assert prog is not None
    at_node = prog.body[1].expr
    assert at_node.obj.name == "ob"           # bare "ob" coerced to Var
    assert at_node.dx.value == 5 and at_node.dx.kind == "lit"
    assert at_node.dy.value == 0 and at_node.dy.kind == "lit"
    type_check(prog)                            # coerced form type-checks (obs/int -> cell -> bool)


def test_pretty_renders_text_dsl():
    prog = Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        If(test="blocked", then=[ExprStmt(expr=Act(action=ActionType.FORWARD))], orelse=[]),
    ])
    text = pretty(prog)
    assert "ob = beat(OBSERVE)" in text
    assert "if blocked:" in text
    assert "act(FORWARD)" in text


# ---------- loops ----------

def _eye_east(width, anchor=(2, 2), wall=None, height=5):
    """ego (wheel/metal/brain) facing EAST at ``anchor`` in an empty width×height
    world; a stone wall is placed at ``wall`` when given. Anchor defaults to
    (2,2): the creature's local frame sits components two cells west of the
    anchor, so x>=2 keeps every cell in bounds."""
    w = World.empty(width, height)
    w.add(Entity(eid="ego",
                 components={(0, 0): Component.of("wheel"),
                             (0, 1): Component.of("metal"),
                             (0, 2): Component.of("brain")},
                 anchor=anchor, orientation=Direction.EAST, is_ego=True))
    if wall is not None:
        w.add(Entity(eid="wall", components={(0, 0): Component.of("stone")}, anchor=wall))
    return w


def _while_until_blocked_program():
    """clear = True; while clear: forward, re-observe, re-check the cell ahead.

    Walks forward until the cell one step ahead (canonical (1,0)) is not empty."""
    return Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        Assign(name="clear", expr=Attr(obj=_at(1, 0), attr="is_empty")),
        While(test="clear", body=[
            ExprStmt(expr=Act(action=ActionType.FORWARD)),
            Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
            Assign(name="clear", expr=Attr(obj=_at(1, 0), attr="is_empty")),
        ]),
    ])


def test_while_advances_until_front_blocked():
    world = _eye_east(8, wall=(7, 2))   # cells (3,2)..(6,2) clear, (7,2) wall
    eng, host, _ = _make_host(world, "ego")
    ego = world.ego
    Interpreter(host).run(_while_until_blocked_program())
    assert ego.anchor == (6, 2)         # stopped one short of the wall (4 forwards)


def test_while_skips_body_when_predicate_false():
    world = _eye_east(8, wall=(3, 2))   # wall immediately ahead of the anchor
    eng, host, _ = _make_host(world, "ego")
    ego = world.ego
    Interpreter(host).run(_while_until_blocked_program())
    assert ego.anchor == (2, 2)         # never entered the loop


def test_for_loop_runs_count_times():
    world = _eye_east(8)
    eng, host, _ = _make_host(world, "ego")
    ego = world.ego
    prog = Program(body=[
        For(count=Lit(value=3), body=[ExprStmt(expr=Act(action=ActionType.FORWARD))]),
    ])
    Interpreter(host).run(prog)
    assert ego.anchor == (5, 2)         # 2 + 3 forwards


def test_for_loop_binds_index_variable():
    # for i in range(2): forward  -> index leaks as int after the loop; we use it
    # in a compare to prove the var is bound and int-valued in (and after) the body.
    world = _eye_east(8)
    eng, host, _ = _make_host(world, "ego")
    ego = world.ego
    prog = Program(body=[
        For(var="i", count=Lit(value=2), body=[
            ExprStmt(expr=Act(action=ActionType.FORWARD)),
        ]),
        # after the loop i == 1 (last index); compare it to prove it survived as int
        Assign(name="done", expr=Compare(op="==", left=Var(name="i"), right=Lit(value=1))),
    ])
    Interpreter(host).run(prog)         # should not raise; ego advanced 2
    assert ego.anchor == (4, 2)


def test_while_test_must_be_bool_variable():
    bad = Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        While(test="ob", body=[]),      # ob is obs, not bool
    ])
    with pytest.raises(CompileError):
        type_check(bad)


def test_while_test_undefined_variable():
    with pytest.raises(CompileError):
        type_check(Program(body=[While(test="missing", body=[])]))


def test_for_count_must_be_int():
    bad = Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        # a bool predicate is not a valid count
        For(count=Attr(obj=_at(1, 0), attr="is_blocked"),
            body=[]),
    ])
    with pytest.raises(CompileError):
        type_check(bad)


def test_for_var_is_int_in_body():
    prog = Program(body=[
        For(var="i", count=Lit(value=3), body=[
            Assign(name="last", expr=Var(name="i")),
        ]),
    ])
    type_check(prog)                    # body sees i as int -> no error


def test_infinite_while_hits_step_budget():
    world = _eye_east(8)
    eng, host, _ = _make_host(world, "ego", budget=50)
    prog = Program(body=[
        Assign(name="on", expr=Lit(value=True)),
        While(test="on", body=[]),      # predicate never flips -> bounded by budget
    ])
    with pytest.raises(BudgetExceeded):
        Interpreter(host).run(prog)


def test_pretty_renders_loops():
    prog = Program(body=[
        Assign(name="ob", expr=Beat(op=BeatOp.OBSERVE)),
        While(test="clear", body=[ExprStmt(expr=Act(action=ActionType.FORWARD))]),
        For(var="i", count=Lit(value=4), body=[ExprStmt(expr=Act(action=ActionType.TURNLEFT))]),
        For(count=Lit(value=2), body=[]),  # no index var -> rendered as `_`
    ])
    text = pretty(prog)
    assert "while clear:" in text
    assert "for i in range(4):" in text
    assert "for _ in range(2):" in text
    assert "pass" in text                 # empty for body renders a pass
