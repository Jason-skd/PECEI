"""Replay frame model: flatten session cycles into a uniform frame stream.

The replay base unit is an **epoch** (one cycle's trace). A session is a list of
epochs; each epoch contributes one *banner* frame (cycle N: stop_reason, script)
followed by one truth frame per round (rebuilt via ``world_at_step``). All frames
share one shape so the app/UI can scrub/seek/play without special-casing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pecei.infra import Result, Trace


@dataclass
class Frame:
    epoch: int                  # 1-based cycle index
    kind: str                   # 'banner' | 'step'
    step: int                   # round index within the cycle (0 = initial pose)
    n_steps: int
    world: object | None = None  # reconstructed World (None for banner)
    event: object | None = None  # TraceEvent for this step (None on banner / terminal)
    stop_reason: str = ""
    script: str = ""
    n_epochs: int = 1
    map_path: str = ""


@dataclass
class ReplayModel:
    map_path: str
    epochs: list[list[Frame]] = field(default_factory=list)

    @property
    def frames(self) -> list[Frame]:
        return [f for ep in self.epochs for f in ep]

    def epoch_starts(self) -> list[int]:
        """Flat-frame index where each epoch begins."""
        starts, i = [], 0
        for ep in self.epochs:
            starts.append(i)
            i += len(ep)
        return starts


def build_from_traces(map_path: str, items: list[tuple[int, Trace, str, str]]) -> ReplayModel:
    """items: list of (epoch_index, trace, stop_reason, script)."""
    from pecei.runner import world_at_step

    n_epochs = len(items)
    model = ReplayModel(map_path=map_path)
    for idx, trace, stop, script in items:
        frames: list[Frame] = []
        frames.append(Frame(epoch=idx, kind="banner", step=0, n_steps=len(trace.events),
                            stop_reason=stop, script=script, n_epochs=n_epochs, map_path=map_path))
        for s in range(len(trace.events) + 1):
            ev = trace.events[s] if s < len(trace.events) else None
            frames.append(Frame(
                epoch=idx, kind="step", step=s, n_steps=len(trace.events),
                world=world_at_step(map_path, trace, s), event=ev,
                stop_reason=stop, script=script, n_epochs=n_epochs, map_path=map_path,
            ))
        model.epochs.append(frames)
    return model


def build_from_trace(map_path: str, trace_path: str) -> ReplayModel:
    """Single-epoch replay from one trace file."""
    trace = Trace.read(trace_path)
    return build_from_traces(map_path, [(1, trace, Result.SUCCESS.value if trace.events else "", "")])


def build_from_session(map_path: str, session) -> ReplayModel:
    """Multi-epoch replay from a Session (each cycle's trace becomes an epoch)."""
    items: list[tuple[int, Trace, str, str]] = []
    for c in session.cycles:
        if c.trace_path and Path(c.trace_path).exists():
            items.append((c.index, Trace.read(c.trace_path), c.stop_reason.value, c.script))
    if not items:
        raise ValueError("session has no replayable cycle traces (run with tracing on)")
    return build_from_traces(map_path, items)
