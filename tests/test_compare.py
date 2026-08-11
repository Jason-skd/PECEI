"""Warm-vs-cold comparison harness tests.

Verifies both arms run, metrics are extracted correctly, sessions land in
distinct subdirectories, and the JSON/CSV outputs are written. Uses
``MockProvider`` (offline; ``memory_llm=None`` deterministic path) so no network.
"""
from pathlib import Path

from pecei.compare import (
    ARM_COLD,
    ARM_WARM,
    ComparisonResult,
    MapMetric,
    _extract_metric,
    run_compare,
)
from pecei.infra import Result
from pecei.llm import MockProvider
from pecei.session import CycleRecord, Session

REPO = Path(__file__).resolve().parents[1]
MAPS = REPO / "src" / "pecei" / "maps"


def _metric(session: Session, slug: str, arm: str) -> MapMetric:
    return _extract_metric(session, slug=slug, arm=arm, session_path=f"out/{arm}/{slug}.session.json")


def test_mock_solves_corridor_in_one_epoch(tmp_path):
    """MockProvider solves 01_corridor in cycle 1 (5 FORWARD) on both arms."""
    import shutil

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    shutil.copy(MAPS / "01_corridor.yaml", train_dir / "01_corridor.yaml")
    shutil.copy(MAPS / "01_corridor.yaml", test_dir / "01_corridor.yaml")

    result = run_compare(
        train_dir, test_dir, MockProvider(),
        budget=5, round_budget=50,
        out_dir=tmp_path / "out",
    )

    assert isinstance(result, ComparisonResult)
    assert len(result.warm) == 1
    assert len(result.cold) == 1
    assert result.warm[0].slug == "corridor"          # slug strips the NN_ prefix
    assert result.warm[0].solved is True
    assert result.warm[0].epochs_to_success == 1
    assert result.cold[0].solved is True
    assert result.cold[0].epochs_to_success == 1

    # Session files isolated per arm (no clobbering); run_experiment names them by slug
    assert (tmp_path / "out" / "warm" / "corridor.session.json").exists()
    assert (tmp_path / "out" / "cold" / "corridor.session.json").exists()
    assert (tmp_path / "out" / "warm_train" / "corridor.session.json").exists()
    # Metrics artifacts
    assert (tmp_path / "out" / "comparison.json").exists()
    assert (tmp_path / "out" / "comparison.csv").exists()


def test_extract_metric_unsolved():
    """A session that never succeeds: solved=False, epochs=len, rounds=full sum."""
    s = Session(map="x.yaml", provider="mock", round_budget=50)
    for i in range(3):
        s.cycles.append(CycleRecord(
            index=i + 1, script="act(FORWARD)",
            stop_reason=Result.SCRIPT_ENDED, rounds=5,
        ))
    m = _metric(s, "x", ARM_COLD)
    assert m.solved is False
    assert m.epochs_to_success == 3
    assert m.total_rounds == 15


def test_extract_metric_first_success():
    """Success on cycle 3: epochs=3, rounds = sum of cycles 1-3."""
    s = Session(map="x.yaml", provider="mock", round_budget=50)
    for i, reason in enumerate([Result.SCRIPT_ENDED, Result.ROUND_LIMIT_EXCEED, Result.SUCCESS]):
        s.cycles.append(CycleRecord(
            index=i + 1, script="act(FORWARD)",
            stop_reason=reason, rounds=3 + i,
        ))
    m = _metric(s, "x", ARM_WARM)
    assert m.solved is True
    assert m.epochs_to_success == 3
    assert m.total_rounds == 3 + 4 + 5  # 12


def test_extract_metric_success_does_not_count_later_cycles():
    """Rounds after the first SUCCESS are not counted (SUCCESS is terminal anyway)."""
    s = Session(map="x.yaml", provider="mock", round_budget=50)
    for reason in [Result.SUCCESS, Result.SUCCESS]:  # second is unreachable in practice
        s.cycles.append(CycleRecord(index=1, script="x", stop_reason=reason, rounds=4))
    m = _metric(s, "x", ARM_WARM)
    assert m.epochs_to_success == 1
    assert m.total_rounds == 4


def test_comparison_json_round_trip():
    import json

    result = ComparisonResult(
        warm=[MapMetric("a", ARM_WARM, 3, 42, True, "out/warm/a.session.json")],
        cold=[MapMetric("a", ARM_COLD, 7, 120, True, "out/cold/a.session.json")],
        budget=10, round_budget=100,
        train_dir="train/", test_dir="test/", provider="mock",
    )
    d = json.loads(json.dumps(result.to_dict()))
    assert d["maps"][0]["warm"]["epochs_to_success"] == 3
    assert d["maps"][0]["cold"]["epochs_to_success"] == 7
    assert d["maps"][0]["slug"] == "a"

    rows = result.to_csv_rows()
    assert len(rows) == 2
    assert {r["arm"] for r in rows} == {ARM_WARM, ARM_COLD}
