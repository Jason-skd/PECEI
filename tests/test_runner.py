"""Script-runner tests: one decide() per cycle + stop-reason mapping + feedback."""
from pathlib import Path

from pecei.action import Act, Assign, ExprStmt, If, Observe, Program
from pecei.infra import Result
from pecei.llm import MockProvider
from pecei.llm.protocol import TurnInput, TurnOutput
from pecei.runner import run_script, world_at_step
from pecei.world.actions import ActionType

REPO = Path(__file__).resolve().parents[1]
EXAMPLE02 = REPO / "src" / "pecei" / "maps" / "example02.yaml"


class _Count:
    """Wraps MockProvider, counting decide() calls + inspecting the TurnInput."""
    name = "count"

    def __init__(self) -> None:
        self._m = MockProvider()
        self.n = 0
        self.last_turn: TurnInput | None = None

    def decide(self, turn: TurnInput) -> TurnOutput:
        self.n += 1
        self.last_turn = turn
        return self._m.decide(turn)


class _Fixed:
    """Provider that returns one fixed TurnOutput (drives COMPILE_ERROR-path tests)."""
    name = "fixed"

    def __init__(self, out: TurnOutput) -> None:
        self._out = out

    def decide(self, turn: TurnInput) -> TurnOutput:
        return self._out


def test_one_decide_per_cycle_and_success():
    p = _Count()
    r = run_script(str(EXAMPLE02), p, round_budget=50)
    assert p.n == 1                       # exactly ONE LLM request per cycle
    assert r.stop_reason is Result.SUCCESS
    assert r.rounds == 5                  # 3 east to goal + 2 blocked (east edge); blocked acts still cost a round
    assert len(r.yielded) == 1
    assert p.last_turn is not None and p.last_turn.feedback is None   # cycle 1 has no feedback


def test_round_limit_exceed_on_tiny_budget():
    r = run_script(str(EXAMPLE02), _Count(), round_budget=1)
    assert r.stop_reason is Result.ROUND_LIMIT_EXCEED


def test_script_ended_when_body_has_no_acts():
    inert = MockProvider(program=Program(body=[Assign(name="ob", expr=Observe())]))
    r = run_script(str(EXAMPLE02), inert, round_budget=50)
    assert r.stop_reason is Result.SCRIPT_ENDED
    assert r.rounds == 0
    assert r.compile_error is None          # body fully executed, no compile fault


def test_compile_error_on_malformed_ast():
    # A-layer: provider parsed a malformed tool-call (e.g. act(YIELD)) -> error set
    out = TurnOutput(program=None, error="at body.0.expr.act.action: bad action 'YIELD'")
    r = run_script(str(EXAMPLE02), _Fixed(out), round_budget=50)
    assert r.stop_reason is Result.COMPILE_ERROR
    assert r.rounds == 0
    assert r.program == ""                  # nothing ran
    assert "YIELD" in (r.compile_error or "")


def test_compile_error_on_typecheck_failure():
    # B-layer: well-formed AST but type_check rejects it (if on undefined var)
    bad = Program(body=[If(test="never_defined",
                           then=[ExprStmt(expr=Act(action=ActionType.FORWARD))])])
    r = run_script(str(EXAMPLE02), MockProvider(program=bad), round_budget=50)
    assert r.stop_reason is Result.COMPILE_ERROR
    assert r.rounds == 0
    assert "never_defined" in (r.compile_error or "")


def test_feedback_and_snowball_wired_into_next_cycle():
    r1 = run_script(str(EXAMPLE02), MockProvider(), round_budget=50)
    probe = _Count()
    run_script(
        str(EXAMPLE02), probe,
        feedback=r1.to_feedback(),
        snowball=[{"index": 1, "stop_reason": r1.stop_reason.value,
                   "rounds": r1.rounds, "scripts": [r1.program]}],
        round_budget=50,
    )
    assert probe.last_turn is not None
    assert probe.last_turn.feedback is not None
    assert probe.last_turn.feedback.stop_reason is Result.SUCCESS
    assert len(probe.last_turn.snowball) == 1


def test_reset_per_script_each_cycle_starts_from_initial_pose():
    r1 = run_script(str(EXAMPLE02), MockProvider(), round_budget=50)
    r2 = run_script(str(EXAMPLE02), MockProvider(), round_budget=50)
    # both cycles reload the fresh map -> identical starting anchor
    assert world_at_step(str(EXAMPLE02), r1.trace, 0).ego.anchor == (3, 2)
    assert world_at_step(str(EXAMPLE02), r2.trace, 0).ego.anchor == (3, 2)


def test_trace_carries_program():
    r = run_script(str(EXAMPLE02), MockProvider(), round_budget=50)
    assert r.trace.events
    assert r.trace.events[0].program and "act(FORWARD)" in r.trace.events[0].program
    assert world_at_step(str(EXAMPLE02), r.trace, 3).ego.anchor == (6, 2)
