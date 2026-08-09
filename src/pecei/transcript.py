"""Transcript: render a session's authored scripts (AST -> text) for humans.

The pretty text already lives on each ``CycleRecord.script`` (it *is*
``pretty(program)``); this module consolidates it into one readable document —
one block per cycle with a stop-report header, and optionally the recorded
prompt. **epoch** granularity works off a single trace file
(``TraceEvent.program`` on its first round).

Used two ways: the ``pecei transcript`` CLI command (epoch / session / experiment),
and the auto-dump that writes ``<stem>.transcript.txt`` next to a session when it
ends (``auto_session`` / ``interactive_loop``).
"""
from __future__ import annotations

from pathlib import Path

from pecei.infra import Trace
from pecei.session import Session


def _stem(map_path: str) -> str:
    return Path(map_path).stem or map_path


def _cycle_block(c, *, with_prompts: bool) -> list[str]:
    out = [f"===== cycle {c.index} — {c.stop_reason.value} ({c.rounds} rounds) ====="]
    if c.compile_error:
        out.append(f"(compile error: {c.compile_error})")
    out.append(c.script if c.script else "(no script)")
    if with_prompts and c.prompt:
        out += ["---- prompt ----", "[system]", c.prompt.get("system") or "",
                "[user]", c.prompt.get("user") or ""]
    return out


def render(session: Session, *, with_prompts: bool = False) -> str:
    """Full per-cycle transcript of ``session``'s authored scripts."""
    header = [
        f"# PECEI transcript — {_stem(session.map)}  "
        f"({len(session.cycles)} cycle(s), {session.success_count} success)",
        f"# map: {session.map}   provider: {session.provider}   model: {session.model or '-'}",
        "",
    ]
    for c in session.cycles:
        header.extend(_cycle_block(c, with_prompts=with_prompts))
        header.append("")
    return "\n".join(header).rstrip() + "\n"


def render_trace(trace_path: str | Path) -> str:
    """epoch granularity: the single script recorded on a trace's first round."""
    trace = Trace.read(trace_path)
    program = next((e.program for e in trace.events if e.program), None)
    lines = [f"===== trace: {Path(trace_path).name} ({len(trace.events)} rounds) =====",
             program or "(no script)"]
    return "\n".join(lines) + "\n"


def default_path(session_path: str | Path) -> Path:
    """``<name>.session.json`` -> ``<name>.transcript.txt`` in the same dir."""
    p = Path(session_path)
    if p.name.endswith(".session.json"):
        base = p.name[: -len(".session.json")]
        return p.with_name(base + ".transcript.txt")
    return p.with_suffix(".transcript.txt")


def write(session: Session, session_path: str | Path, *, with_prompts: bool = False) -> Path:
    """Render ``session`` to ``<session_path>.transcript.txt``; return the path."""
    out = default_path(session_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(session, with_prompts=with_prompts), encoding="utf-8")
    return out
