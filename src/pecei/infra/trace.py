"""Trace log: JSONL of per-round events (ground truth + actor obs + LLM I/O slots).

Each line is one :class:`TraceEvent`. ``llm_request`` / ``llm_response`` are
reserved opaque slots (filled in M6 when the LLM is wired); they're present now
so the format is forward-compatible. ``observation`` is a serialized
Observation (what the actor saw that round).
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    round: int
    actor: str | None = None
    action: str | None = None                     # ActionType value, or None
    moved: bool | None = None
    blocked: bool | None = None
    anchor_after: tuple[int, int] | None = None
    orientation_after: str | None = None
    observation: dict | None = None               # serialized Observation
    program: str | None = None                    # pretty(program) for the cycle owning this round
    llm_request: dict | None = None               # reserved (M6): raw LLM request
    llm_response: dict | None = None              # reserved (M6): raw LLM response
    yielded: list[dict] = Field(default_factory=list)  # serialized obs yielded this round


class Trace:
    """In-memory list of TraceEvents with JSONL write/read."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def append(self, event: TraceEvent) -> None:
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for event in self.events:
                f.write(event.model_dump_json() + "\n")

    @classmethod
    def read(cls, path: str | Path) -> "Trace":
        trace = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                trace.events.append(TraceEvent.model_validate_json(line))
        return trace
