"""Experiment/session orchestration: NN_slug parsing, stop-on-success, same-source."""
from pathlib import Path

from pecei.experiment import parse_experiment, run_experiment
from pecei.infra import Result, Trace
from pecei.llm import MockProvider
from pecei.llm.protocol import TurnInput, TurnOutput
from pecei.session import CycleRecord, Session, auto_session

REPO = Path(__file__).resolve().parents[1]
CORRIDOR = REPO / "src" / "pecei" / "maps" / "01_corridor.yaml"


class _Count:
    """Counts decide() calls and records the TurnInput for inspection."""
    name = "count"

    def __init__(self) -> None:
        self._m = MockProvider()
        self.n = 0
        self.instructions: list[str | None] = []
        self.snowball_lens: list[int] = []

    def decide(self, turn: TurnInput) -> TurnOutput:
        self.n += 1
        self.instructions.append(turn.instructions)
        self.snowball_lens.append(len(turn.snowball))
        return self._m.decide(turn)


def _write_maps(d: Path) -> None:
    (d / "02_b.yaml").write_text("width: 2\nheight: 1\nentities: []\n", encoding="utf-8")
    (d / "01_a.yaml").write_text("width: 2\nheight: 1\nentities: []\n", encoding="utf-8")
    (d / "ignore_this.yaml").write_text("width: 1\nheight: 1\nentities: []\n", encoding="utf-8")


def test_parse_experiment_orders_by_numeric_prefix(tmp_path):
    _write_maps(tmp_path)
    refs = parse_experiment(tmp_path)
    assert [r.slug for r in refs] == ["a", "b"]        # NN_ order, not dict order
    assert [r.index for r in refs] == [1, 2]
    assert "ignore_this" not in [r.slug for r in refs]  # non-NN files skipped


def test_run_experiment_injects_session_k_of_n(tmp_path):
    out = tmp_path / "out"
    # use the real solvable map for session 1; a trivial unsolvable map for session 2
    import shutil
    shutil.copy(CORRIDOR, tmp_path / "01_corridor.yaml")
    (tmp_path / "02_nowhere.yaml").write_text(
        "width: 3\nheight: 1\ngoal: [0, 0]\nentities:\n"
        "  - name: ego\n    anchor: [2, 0]\n    orientation: EAST\n    is_ego: true\n"
        "    components:\n      - { offset: [0, 0], type: brain }\n", encoding="utf-8")
    sessions = run_experiment(tmp_path, _Count(), out_dir=out, budget=2)
    assert len(sessions) == 2
    assert sessions[0].success_count >= 1                  # corridor solved
    assert sessions[0].instructions and "1 of 2" in sessions[0].instructions
    assert sessions[1].instructions and "2 of 2" in sessions[1].instructions
    assert len(list(out.glob("*.session.json"))) == 2


def test_auto_session_stops_on_first_success(tmp_path):
    prov = _Count()
    s = Session(map=str(CORRIDOR), provider="mock", round_budget=50)
    auto_session(s, prov, tmp_path / "s.json", budget=10, trace_dir=tmp_path / "tr")
    assert prov.n == 1                                   # solved on the first cycle
    assert s.cycles[-1].stop_reason is Result.SUCCESS
    assert (tmp_path / "s.json").exists()


def test_auto_session_writes_transcript_on_end(tmp_path):
    s = Session(map=str(CORRIDOR), provider="mock", round_budget=50)
    auto_session(s, _Count(), tmp_path / "s.session.json", budget=10, trace_dir=None)
    assert (tmp_path / "s.transcript.txt").exists()      # dumped next to the session


def test_auto_session_skips_transcript_when_disabled(tmp_path):
    s = Session(map=str(CORRIDOR), provider="mock", round_budget=50)
    auto_session(s, _Count(), tmp_path / "s.session.json", budget=10, trace_dir=None,
                 dump_transcript=False)
    assert not (tmp_path / "s.transcript.txt").exists()


def test_auto_session_respects_budget_when_unsolved(tmp_path):
    prov = _Count()
    s = Session(map=str(CORRIDOR), provider="mock", round_budget=1)  # budget 1 round -> never solves
    auto_session(s, prov, tmp_path / "s.json", budget=3, trace_dir=None)
    assert prov.n == 3                                   # budget caps cycles
    assert s.success_count == 0


def test_session_same_source_no_duplicate_yielded(tmp_path):
    s = Session(map=str(CORRIDOR), provider="mock", round_budget=50)
    s.run_one_cycle(MockProvider(), trace_dir=tmp_path / "tr")
    rec = s.cycles[0]
    # session stores only the summary; yields live in the trace (same-source)
    assert rec.trace_path is not None and Path(rec.trace_path).exists()
    dumped = rec.model_dump()
    assert "yielded" not in dumped                        # not duplicated into session JSON
    fb = s.last_feedback()
    assert fb is not None and fb.stop_reason is Result.SUCCESS  # rebuilt from the cycle
    assert Trace.read(rec.trace_path).events              # trace is replayable


def test_snowball_lags_one_cycle_because_last_is_in_feedback(tmp_path):
    # The last cycle lives in full detail in `feedback`; the snowball carries
    # only the EARLIER cycles, so it lags by one (only grows from cycle 3 on).
    prov = _Count()
    s = Session(map=str(CORRIDOR), provider="mock", round_budget=50)
    s.run_one_cycle(prov, trace_dir=None)
    assert prov.snowball_lens == [0]                    # cycle 1: nothing prior
    s.run_one_cycle(prov, trace_dir=None)
    assert prov.snowball_lens == [0, 0]                 # cycle 2: the 1 prior is IN feedback
    s.run_one_cycle(prov, trace_dir=None)
    assert prov.snowball_lens == [0, 0, 1]              # cycle 3: cycle 1 in snowball, 2 in feedback
