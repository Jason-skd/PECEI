"""Session/cycle tests: one cycle per run_one_cycle, snowball grows, round-trip."""
from pathlib import Path

from pecei.infra import Result
from pecei.llm import MockProvider
from pecei.llm.protocol import TurnInput, TurnOutput
from pecei.session import CycleRecord, Session

REPO = Path(__file__).resolve().parents[1]
EXAMPLE02 = REPO / "src" / "pecei" / "maps" / "example02.yaml"


class _Count:
    name = "count"

    def __init__(self) -> None:
        self._m = MockProvider()
        self.n = 0
        self.snowball_len = 0
        self.has_feedback = False

    def decide(self, turn: TurnInput) -> TurnOutput:
        self.n += 1
        self.snowball_len = len(turn.snowball)
        self.has_feedback = turn.feedback is not None
        return self._m.decide(turn)


def test_session_roundtrip(tmp_path):
    s = Session(map="x.yaml")
    s.cycles.append(CycleRecord(index=1, script="act(FORWARD)",
                                stop_reason=Result.SUCCESS, rounds=3))
    out = tmp_path / "s.json"
    s.save(out)
    s2 = Session.load(out)
    assert s2.map == "x.yaml"
    assert len(s2.cycles) == 1
    assert s2.cycles[0].stop_reason is Result.SUCCESS


def test_run_one_cycle_is_one_request_and_snowball_lags():
    # The last cycle is fully represented in `feedback` (incl. its script), so the
    # snowball carries only the EARLIER cycles — it lags by one.
    s = Session(map=str(EXAMPLE02), provider="mock", round_budget=50)

    p1 = _Count()
    s.run_one_cycle(p1)
    assert p1.n == 1 and not p1.has_feedback          # cycle 1: no feedback, one request
    assert len(s.cycles) == 1 and s.cycles[0].stop_reason is Result.SUCCESS

    p2 = _Count()
    s.run_one_cycle(p2)
    assert p2.n == 1 and p2.has_feedback              # cycle 2: feedback present, one request
    assert p2.snowball_len == 0                        # the 1 prior cycle is IN feedback
    assert len(s.cycles) == 2

    p3 = _Count()
    s.run_one_cycle(p3)
    assert p3.snowball_len == 1                        # cycle 1 now in snowball; cycle 2 in feedback
    assert len(s.cycles) == 3


def test_run_one_cycle_records_prompt():
    s = Session(map=str(EXAMPLE02), provider="mock", round_budget=50)
    s.run_one_cycle(MockProvider())
    rec = s.cycles[0]
    assert rec.prompt and set(rec.prompt) == {"system", "user"}
    assert rec.prompt["system"].strip()                 # non-empty system prompt
    assert "DIRECTIVE: PLAN" in rec.prompt["user"]      # render_user output captured


def test_prompt_roundtrips_through_session_json(tmp_path):
    s = Session(map="x.yaml")
    s.cycles.append(CycleRecord(
        index=1, script="act(FORWARD)", stop_reason=Result.SUCCESS, rounds=1,
        prompt={"system": "SYS", "user": "USR"}))
    out = tmp_path / "s.json"
    s.save(out)
    s2 = Session.load(out)
    assert s2.cycles[0].prompt == {"system": "SYS", "user": "USR"}


def test_yielded_observations_flow_into_next_cycle_feedback(tmp_path):
    # End-to-end feedback channel: a cycle's beat(YIELD) observations must be
    # recoverable from its written trace by last_feedback(), so the next cycle's
    # author actually sees them. Regression: this channel was silently empty.
    s = Session(map=str(EXAMPLE02), provider="mock", round_budget=50)
    s.run_one_cycle(MockProvider(), trace_dir=str(tmp_path / "traces"))
    fb = s.last_feedback()
    assert fb is not None
    assert len(fb.yielded) == 1
