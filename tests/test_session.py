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


def test_run_one_cycle_is_one_request_and_snowball_grows():
    s = Session(map=str(EXAMPLE02), provider="mock", round_budget=50)

    p1 = _Count()
    s.run_one_cycle(p1)
    assert p1.n == 1 and not p1.has_feedback          # cycle 1: no feedback, one request
    assert len(s.cycles) == 1 and s.cycles[0].stop_reason is Result.SUCCESS

    p2 = _Count()
    s.run_one_cycle(p2)
    assert p2.n == 1 and p2.has_feedback              # cycle 2: feedback present, one request
    assert p2.snowball_len == 1                        # snowball = the 1 prior cycle
    assert len(s.cycles) == 2
