"""Infrastructure layer: trace log, match report, complexity (loss function).

Imports world + observation only. Holds no game/LLM state of its own; consumes
serialized snapshots (dicts) from those layers. LLM I/O is stored opaquely.
"""
from .complexity import complexity
from .report import FailureSnapshot, MatchReport, Result, build_report
from .trace import Trace, TraceEvent

__all__ = [
    "FailureSnapshot",
    "MatchReport",
    "Result",
    "Trace",
    "TraceEvent",
    "build_report",
    "complexity",
]
