"""Experiment runner: train a folder of maps, one session per map.

Map files follow the ``NN_slug.yaml`` convention (two-digit order prefix); a
directory of them forms an experiment. Each map trains as its own Session until
SUCCESS or cycle budget; the session's authoritative ``instructions`` carry the
experiment context ("this is session k of N: <slug>").

When a shared ``MemoryEvolution`` is supplied (``memory=``), every cycle's
feedback is remembered and its rendered context is injected into the next cycle
— so the bans learned on earlier maps follow the agent onto later maps. That
shared instance is the *warm-start* arm of the comparison experiment; the
cold-start arm simply runs each map with ``memory=None``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

from pecei.llm.protocol import LLMProvider
from pecei.session import Session, auto_session

_MAP_RE = re.compile(r"^(?P<num>\d{2,})_(?P<slug>.+?)\.(?:yaml|yml|json)$")


class MapRef(NamedTuple):
    """(index, path, slug) for one map in an experiment, ordered by filename."""
    index: int
    path: Path
    slug: str


def parse_experiment(directory: str | Path) -> list[MapRef]:
    """Scan ``directory`` for ``NN_slug.{yaml,yml,json}`` maps, ordered by the
    numeric prefix. Returns MapRef(index, path, slug); index starts at 1."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"experiment dir not found: {directory}")
    out: list[tuple[int, Path, str]] = []
    for p in d.iterdir():
        m = _MAP_RE.match(p.name)
        if m:
            out.append((int(m.group("num")), p, m.group("slug")))
    out.sort(key=lambda t: t[0])
    return [MapRef(i + 1, path, slug) for i, (_n, path, slug) in enumerate(out)]


def run_experiment(
    directory: str | Path,
    provider: LLMProvider,
    *,
    out_dir: str | Path = "sessions",
    budget: int = 10,
    round_budget: int = 100,
    dump_transcript: bool = True,
    memory: Any | None = None,
) -> list[Session]:
    """Train each map in ``directory`` as its own session, in order. Returns the
    list of Sessions (each saved to ``out_dir/<slug>.session.json``).

    ``memory``: an optional shared ``MemoryEvolution`` threaded through every
    session's cycles, so learned bans accumulate across the maps trained here.
    """
    maps = parse_experiment(directory)
    total = len(maps)
    if total == 0:
        print(f"[experiment] no NN_slug maps found in {directory}")
        return []

    out = Path(out_dir)
    sessions: list[Session] = []
    for ref in maps:
        print(f"\n=== session {ref.index}/{total}: {ref.slug} ({ref.path.name}) ===")
        sess = Session(
            map=str(ref.path),
            provider=getattr(provider, "name", "mock"),
            round_budget=round_budget,
            instructions=f"experiment session {ref.index} of {total}: map '{ref.slug}'",
        )
        sess_path = out / f"{ref.slug}.session.json"
        trace_dir = out / f"{ref.slug}.traces"
        auto_session(sess, provider, sess_path, budget=budget, trace_dir=trace_dir,
                     dump_transcript=dump_transcript, memory=memory)
        sessions.append(sess)

    solved = sum(1 for s in sessions if s.success_count > 0)
    print(f"\n[experiment] done: {solved}/{total} maps solved")
    return sessions
