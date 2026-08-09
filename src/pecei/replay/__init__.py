"""Replay layer: epoch-based replay viewer + session browser.

- ``replay_viewer(map, trace)``: replay a single epoch (one trace).
- ``replay_session(map, session)``: browse a session's epochs, replay one, or
  play from the selected epoch through to the end.

Image tiles (``render/assets/*.png``) are used when present; the renderer falls
back to color+shape blocks when they're missing. UI text is Pillow-rendered
(pygame fonts are broken on py3.14). Entry points are lazily imported so the
render layer can reuse ``replay.assets`` without a circular import.
"""
from __future__ import annotations

__all__ = ["replay_viewer", "replay_session"]


def replay_viewer(map_path: str, trace_path: str) -> None:
    from .app import replay_viewer as _rv
    return _rv(map_path, trace_path)


def replay_session(map_path: str, session) -> None:
    from .app import replay_session as _rs
    return _rs(map_path, session)
