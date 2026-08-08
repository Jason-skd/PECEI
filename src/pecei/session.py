"""Epoch system: one spacebar = one script cycle (gradient-free loop).

A :class:`Session` snowballs :class:`CycleRecord`s. Each cycle runs ONE script
(one LLM request) via :func:`pecei.runner.run_script`; its outcome (script +
stop-report + yielded observations) is appended and fed back as
:class:`Feedback` plus the snowball to the next cycle. ``pecei epoch`` drives
this interactively — space advances one cycle, ``Ctrl+C`` saves and quits.
"""
from __future__ import annotations

import sys
import termios
import tty
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pecei.infra import FailureSnapshot, Result
from pecei.llm.protocol import Feedback, LLMProvider
from pecei.runner import run_script


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CycleRecord(BaseModel):
    index: int
    script: str
    stop_reason: Result
    rounds: int
    yielded: list[dict] = Field(default_factory=list)
    failure_snapshot: FailureSnapshot | None = None
    trace_path: str | None = None


class Session(BaseModel):
    map: str
    provider: str = "mock"
    model: str | None = None
    base_url: str | None = None
    round_budget: int = 100
    created: str = Field(default_factory=_now)
    updated: str = Field(default_factory=_now)
    cycles: list[CycleRecord] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        self.updated = _now()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def cycle_summaries(self) -> list[dict[str, Any]]:
        """Compact per-cycle history, fed into the next cycle's snowball."""
        return [
            {
                "index": c.index,
                "stop_reason": c.stop_reason.value,
                "rounds": c.rounds,
                "scripts": [c.script] if c.script else [],
            }
            for c in self.cycles
        ]

    @property
    def success_count(self) -> int:
        return sum(1 for c in self.cycles if c.stop_reason is Result.SUCCESS)

    def run_one_cycle(
        self,
        provider: LLMProvider,
        *,
        trace_dir: str | Path | None = None,
    ) -> CycleRecord:
        """Run ONE script cycle and append its record. The previous cycle's
        outcome becomes this cycle's Feedback; all prior cycles form the snowball."""
        last = self.cycles[-1] if self.cycles else None
        feedback = Feedback(
            stop_reason=last.stop_reason,
            rounds_used=last.rounds,
            yielded=list(last.yielded),
            failure_snapshot=last.failure_snapshot,
        ) if last else None

        run = run_script(
            self.map, provider,
            feedback=feedback, snowball=self.cycle_summaries(),
            round_budget=self.round_budget,
        )

        trace_path: str | None = None
        if trace_dir is not None:
            tdir = Path(trace_dir)
            tdir.mkdir(parents=True, exist_ok=True)
            tp = tdir / f"{Path(self.map).stem}.c{len(self.cycles) + 1:03d}.trace.jsonl"
            run.trace.write(str(tp))
            trace_path = str(tp)

        rec = CycleRecord(
            index=len(self.cycles) + 1,
            script=run.program,
            stop_reason=run.stop_reason,
            rounds=run.rounds,
            yielded=list(run.yielded),
            failure_snapshot=run.failure_snapshot,
            trace_path=trace_path,
        )
        self.cycles.append(rec)
        self.updated = _now()
        return rec


# ---------------- interactive key reading ----------------

def read_key() -> str:
    """Read a single keypress from a tty (cbreak); restore on exit.

    Non-tty stdin (piped / tests) falls back to a line read so the loop stays
    scriptable. Ctrl+C during cbreak raises ``KeyboardInterrupt``.
    """
    if not sys.stdin.isatty():
        return sys.stdin.readline()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def interactive_loop(
    session: Session,
    provider: LLMProvider,
    path: str | Path,
    *,
    trace_dir: str | Path | None = None,
    existed: bool = False,
) -> Session:
    """Run one script cycle per keypress until 'q' / Ctrl+C. Saves after each cycle."""
    if session.cycles:
        print(f"resuming session: {len(session.cycles)} cycle(s), "
              f"{session.success_count} success — map {session.map}")
    try:
        while True:
            # wait FIRST: every cycle (including the first) is triggered by a keypress,
            # so one SPACE == one LLM request. q / Ctrl+C quits before running.
            print(f"[press SPACE for cycle {len(session.cycles) + 1}, q/Ctrl+C to quit] ",
                  end="", flush=True)
            key = read_key()
            print()
            if key.strip().lower() == "q":
                break
            rec = session.run_one_cycle(provider, trace_dir=trace_dir)
            session.save(path)
            _print_cycle(rec)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if session.cycles or existed:
            session.save(path)
            print(f"session saved -> {path}  ({len(session.cycles)} cycle(s), "
                  f"{session.success_count} success)")
        else:
            print("no cycles run — no session file written")
    return session


def _print_cycle(rec: CycleRecord) -> None:
    print(f"\n[cycle {rec.index}] stop={rec.stop_reason.value} rounds={rec.rounds}")
    if rec.failure_snapshot:
        print(f"  ended near {tuple(rec.failure_snapshot.pos)}, "
              f"complexity {rec.failure_snapshot.complexity}")
    if rec.script:
        lines = rec.script.splitlines()
        head = lines[0]
        more = " ..." if len(lines) > 1 else ""
        print(f"  script: {head}{more}")
