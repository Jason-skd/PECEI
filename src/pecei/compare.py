"""Figure-1 comparison experiment: warm-start vs cold-start.

The same prototype runs the same batch of TEST maps twice:

- **warm**: first trains on a TRAIN map directory with a *shared*
  :class:`MemoryEvolution` (bans accumulate across all train maps), then runs
  each TEST map carrying that same accumulated memory.
- **cold**: runs each TEST map with ``memory=None`` — no carried-over
  experience, every map explored from scratch.

The narrative hypothesis (``docs/NARRATIVE_LOGIC.md`` Figure 1) is that the
warm arm reaches the goal in **fewer cycles and fewer rounds** than the cold
arm on the test maps — i.e. the prototype visibly *learned* from the training
maps.

This module is pure orchestration + metric extraction: it reuses
:func:`pecei.experiment.run_experiment` (which already threads a shared memory
through every session) and :class:`pecei.session.Session`, so no per-session
loop or map parsing is reimplemented here.
"""
from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pecei.experiment import parse_experiment, run_experiment
from pecei.infra import Result
from pecei.llm.protocol import LLMProvider
from pecei.memory import MemoryEvolution
from pecei.session import Session

logger = logging.getLogger("pecei.compare")

ARM_WARM = "warm"
ARM_COLD = "cold"


@dataclass
class MapMetric:
    """Metrics for ONE arm (warm or cold) on ONE test map."""

    slug: str
    arm: str                     # ARM_WARM | ARM_COLD
    epochs_to_success: int       # cycles up to & including first SUCCESS; len(cycles) if never solved
    total_rounds: int            # sum of rounds across cycles up to & including first SUCCESS
    solved: bool                 # True iff SUCCESS was reached within budget
    session_path: str            # saved session JSON (for replay / transcript)


@dataclass
class ComparisonResult:
    """Full warm-vs-cold comparison: per-map metrics for both arms."""

    warm: list[MapMetric] = field(default_factory=list)   # one per test map, in parse order
    cold: list[MapMetric] = field(default_factory=list)
    budget: int = 0
    round_budget: int = 0
    train_dir: str = ""
    test_dir: str = ""
    provider: str = ""

    def to_dict(self) -> dict:
        """Nested dict for ``comparison.json`` — one entry per test map with
        both arms side by side."""
        return {
            "train_dir": self.train_dir,
            "test_dir": self.test_dir,
            "budget": self.budget,
            "round_budget": self.round_budget,
            "provider": self.provider,
            "maps": [
                {
                    "slug": w.slug,
                    "warm": _metric_dict(w),
                    "cold": _metric_dict(c),
                }
                for w, c in zip(self.warm, self.cold)
            ],
        }

    def to_csv_rows(self) -> list[dict]:
        """Flat rows for ``comparison.csv``: one row per (map, arm) pair."""
        rows: list[dict] = []
        for m in [*self.warm, *self.cold]:
            rows.append({
                "slug": m.slug,
                "arm": m.arm,
                "epochs_to_success": m.epochs_to_success,
                "total_rounds": m.total_rounds,
                "solved": m.solved,
                "session_path": m.session_path,
            })
        return rows


def _metric_dict(m: MapMetric) -> dict:
    return {
        "epochs_to_success": m.epochs_to_success,
        "total_rounds": m.total_rounds,
        "solved": m.solved,
        "session_path": m.session_path,
    }


def _extract_metric(session: Session, *, slug: str, arm: str, session_path: str) -> MapMetric:
    """Walk ``session.cycles`` for the first SUCCESS; sum rounds up to it.

    A map that never succeeds within budget reports ``solved=False``,
    ``epochs_to_success=len(cycles)`` (= budget, since ``auto_session`` stops at
    budget) and ``total_rounds`` = sum of every cycle's rounds. The session JSON
    is authoritative — ``CycleRecord.stop_reason`` / ``.rounds`` need no trace
    re-read.
    """
    for i, cycle in enumerate(session.cycles):
        if cycle.stop_reason is Result.SUCCESS:
            return MapMetric(
                slug=slug, arm=arm,
                epochs_to_success=i + 1,
                total_rounds=sum(c.rounds for c in session.cycles[: i + 1]),
                solved=True,
                session_path=session_path,
            )
    return MapMetric(
        slug=slug, arm=arm,
        epochs_to_success=len(session.cycles),
        total_rounds=sum(c.rounds for c in session.cycles),
        solved=False,
        session_path=session_path,
    )


def _metrics_from_sessions(
    sessions: list[Session], refs, arm: str, out_subdir: Path
) -> list[MapMetric]:
    """Build a MapMetric per test session, paired with its MapRef slug."""
    metrics: list[MapMetric] = []
    for sess, ref in zip(sessions, refs):
        session_path = str(out_subdir / f"{ref.slug}.session.json")
        metrics.append(_extract_metric(sess, slug=ref.slug, arm=arm, session_path=session_path))
    return metrics


def run_compare(
    train_dir: str | Path,
    test_dir: str | Path,
    provider: LLMProvider,
    *,
    budget: int = 10,
    round_budget: int = 100,
    out_dir: str | Path = "compare",
    memory_llm: Callable[[str], str] | None = None,
    dump_transcript: bool = False,
    resume: bool = False,
) -> ComparisonResult:
    """Run the warm-vs-cold comparison and persist metrics + sessions.

    Parameters
    ----------
    train_dir:
        Directory of ``NN_slug`` maps the warm arm trains on (its memory
        accumulates across these). The cold arm never touches these.
    test_dir:
        Directory of ``NN_slug`` maps BOTH arms are evaluated on.
    memory_llm:
        Optional ``Callable[[str], str]`` handed to the warm arm's
        :class:`MemoryEvolution` for LLM-driven compression of failures into
        bans. ``None`` (default) uses the deterministic rule-based fallback —
        fine for offline/CI runs, but for the real Figure-1 experiment wire a
        real LLM so the warm arm learns genuinely useful bans.

    Output layout (under ``out_dir``)::

        warm_train/   # warm-arm training sessions (side-effect only)
        warm/         # warm-arm test sessions
        cold/         # cold-arm test sessions
        comparison.json
        comparison.csv
    """
    out = Path(out_dir)
    train_refs = parse_experiment(train_dir)
    test_refs = parse_experiment(test_dir)
    if not test_refs:
        raise ValueError(f"no NN_slug maps found in test_dir {test_dir}")

    provider_name = getattr(provider, "name", "mock")

    # ---------------- Phase 1: warm (train → test with accumulated memory) --
    print(f"\n[warm] training on {len(train_refs)} map(s) from {train_dir} ...")
    warm_memory = MemoryEvolution(llm=memory_llm)
    if train_refs:
        run_experiment(
            train_dir, provider,
            out_dir=out / "warm_train", budget=budget, round_budget=round_budget,
            dump_transcript=dump_transcript, memory=warm_memory, resume=resume,
        )
    else:
        logger.warning("train_dir %s has no maps; warm arm gets no training", train_dir)

    print(f"\n[warm] evaluating on {len(test_refs)} test map(s) with accumulated memory ...")
    warm_sessions = run_experiment(
        test_dir, provider,
        out_dir=out / "warm", budget=budget, round_budget=round_budget,
        dump_transcript=dump_transcript, memory=warm_memory, resume=resume,
    )
    warm_metrics = _metrics_from_sessions(warm_sessions, test_refs, ARM_WARM, out / "warm")

    # ---------------- Phase 2: cold (test with no memory) -------------------
    print(f"\n[cold] evaluating on {len(test_refs)} test map(s) with no memory ...")
    cold_sessions = run_experiment(
        test_dir, provider,
        out_dir=out / "cold", budget=budget, round_budget=round_budget,
        dump_transcript=dump_transcript, memory=None, resume=resume,
    )
    cold_metrics = _metrics_from_sessions(cold_sessions, test_refs, ARM_COLD, out / "cold")

    # ---------------- Phase 3: assemble + persist ---------------------------
    result = ComparisonResult(
        warm=warm_metrics, cold=cold_metrics,
        budget=budget, round_budget=round_budget,
        train_dir=str(train_dir), test_dir=str(test_dir), provider=provider_name,
    )
    _write_json(result, out / "comparison.json")
    _write_csv(result, out / "comparison.csv")
    print(f"\n[compare] done -> {out / 'comparison.json'} (+ .csv)")
    return result


def _write_json(result: ComparisonResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def _write_csv(result: ComparisonResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["slug", "arm", "epochs_to_success", "total_rounds", "solved", "session_path"],
        )
        writer.writeheader()
        writer.writerows(result.to_csv_rows())
