"""Render layer: headless tile renderer (Grid -> pygame.Surface) + pygame shell.

Imports world + pygame only. Renderer produces a Surface without opening a
window (headless, testable); ``show`` opens the display. Reused by live/replay
(M7) and PNG export.
"""
from .io import save_image
from .preview import show
from .renderer import Renderer

__all__ = ["Renderer", "save_image", "show"]
