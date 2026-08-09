"""Epoch system: one spacebar = one script cycle (gradient-free loop).

A :class:`Session` snowballs :class:`CycleRecord`s. Each cycle runs ONE script
(one LLM request) via :func:`pecei.runner.run_script`; its outcome (script +
stop-report) is appended and fed back as :class:`Feedback` plus the snowball to
the next cycle. ``pecei epoch`` drives this interactively — space advances one
cycle and the session stops on the first SUCCESS.

**Same-source**: the per-cycle trace (`*.trace.jsonl`) is the authoritative
record of rounds/yielded/LLM-IO. ``CycleRecord`` keeps only a small summary +
``trace_path``; the yielded observations are derived from the trace on demand
(:meth:`Session.last_feedback`), never duplicated into the session JSON.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

if sys.platform == "win32":  # Unix-only modules; Windows uses msvcrt in read_key()
    import msvcrt
else:
    import termios
    import tty
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pecei.infra import FailureSnapshot, Result, Trace
from pecei.llm.protocol import Feedback, LLMProvider
from pecei.runner import run_script


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_yielded(trace: Trace) -> list[dict]:
    """The observations this script chose to beat(YIELD), recovered from the trace."""
    return [y for ev in trace.events for y in (ev.yielded or [])]


class CycleRecord(BaseModel):
    index: int
    script: str
    stop_reason: Result
    rounds: int
    failure_snapshot: FailureSnapshot | None = None
    compile_error: str | None = None       # set iff stop_reason is COMPILE_ERROR
    prompt: dict | None = None             # {system, user} shown to the author (drift-stable replay record)
    trace_path: str | None = None          # authoritative per-cycle record (rounds/yielded/LLM-IO)


class Session(BaseModel):
    map: str
    provider: str = "mock"
    model: str | None = None
    base_url: str | None = None
    round_budget: int = 100
    instructions: str | None = None        # authoritative author-immutable (e.g. experiment k/N)
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
                "error": c.compile_error,
            }
            for c in self.cycles
        ]

    @property
    def success_count(self) -> int:
        return sum(1 for c in self.cycles if c.stop_reason is Result.SUCCESS)

    def last_feedback(self) -> Feedback | None:
        """Feedback from the last cycle, with yields read from its trace (same-source)."""
        if not self.cycles:
            return None
        last = self.cycles[-1]
        yielded: list[dict] = []
        if last.trace_path and Path(last.trace_path).exists():
            yielded = _trace_yielded(Trace.read(last.trace_path))
        return Feedback(
            stop_reason=last.stop_reason,
            rounds_used=last.rounds,
            yielded=yielded,
            failure_snapshot=last.failure_snapshot,
            compile_error=last.compile_error,
        )

    def run_one_cycle(
        self,
        provider: LLMProvider,
        *,
        trace_dir: str | Path | None = None,
    ) -> CycleRecord:
        """Run ONE script cycle and append its record. The previous cycle's outcome
        becomes this cycle's Feedback; all prior cycles form the snowball."""
        run = run_script(
            self.map, provider,
            feedback=self.last_feedback(),
            snowball=self.cycle_summaries(),
            instructions=self.instructions,
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
            failure_snapshot=run.failure_snapshot,
            compile_error=run.compile_error,
            prompt=run.prompt,
            trace_path=trace_path,
        )
        self.cycles.append(rec)
        self.updated = _now()
        return rec


def auto_session(
    session: Session,
    provider: LLMProvider,
    path: str | Path,
    *,
    budget: int = 10,
    trace_dir: str | Path | None = None,
    dump_transcript: bool = True,
) -> Session:
    """Run cycles automatically until SUCCESS or ``budget`` cycles are exhausted."""
    while True:
        if session.cycles and session.cycles[-1].stop_reason is Result.SUCCESS:
            print(f"[auto] SUCCESS in cycle {session.cycles[-1].index}")
            break
        if len(session.cycles) >= budget:
            print(f"[auto] budget {budget} cycles exhausted ({session.success_count} success)")
            break
        rec = session.run_one_cycle(provider, trace_dir=trace_dir)
        session.save(path)
        _print_cycle(rec)
    session.save(path)
    if dump_transcript:
        _dump_transcript(session, path)
    return session


# ---------------- interactive key reading ----------------

def read_key() -> str:
    """Read a single keypress from a tty (cbreak); restore on exit.

    Non-tty stdin (piped / tests) falls back to a line read so the loop stays
    scriptable. Ctrl+C during cbreak raises ``KeyboardInterrupt``.
    """
    if not sys.stdin.isatty():
        return sys.stdin.readline()
    if sys.platform == "win32":
        return msvcrt.getwch()
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
    dump_transcript: bool = True,
) -> Session:
    """Run one script cycle per keypress until SUCCESS / 'q' / Ctrl+C. Saves each cycle."""
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
            if rec.stop_reason is Result.SUCCESS:
                print(f"SUCCESS in cycle {rec.index} — session complete.")
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if session.cycles or existed:
            session.save(path)
            print(f"session saved -> {path}  ({len(session.cycles)} cycle(s), "
                  f"{session.success_count} success)")
            if dump_transcript:
                _dump_transcript(session, path)
        else:
            print("no cycles run — no session file written")
    return session


def _print_cycle(rec: CycleRecord) -> None:
    print(f"\n[cycle {rec.index}] stop={rec.stop_reason.value} rounds={rec.rounds}")
    if rec.compile_error:
        print(f"  compile error: {rec.compile_error}")
    if rec.failure_snapshot:
        print(f"  ended near {tuple(rec.failure_snapshot.pos)}, "
              f"complexity {rec.failure_snapshot.complexity}")
    if rec.script:
        lines = rec.script.splitlines()
        head = lines[0]
        more = " ..." if len(lines) > 1 else ""
        print(f"  script: {head}{more}")


def _dump_transcript(session: "Session", path: str | Path) -> None:
    """Write ``<path>.transcript.txt`` next to a just-saved session (lazy import)."""
    from pecei import transcript

    if not session.cycles:
        return
    tp = transcript.write(session, path)
    print(f"transcript -> {tp}")
