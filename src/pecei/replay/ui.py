"""Replay UI widgets: Pillow-text labels, buttons, selection list.

pygame fonts are broken on py3.14, so all text is Pillow-rendered (``render_text``).
Buttons are simple colored rects with a text label and a hit-test; the epoch list
is a clickable selection panel. This keeps the viewer self-contained.
"""
from __future__ import annotations

import pygame

from pecei.render.text import render_text

PAD_BG = (16, 18, 24)
BTN_BG = (52, 56, 70)
BTN_HL = (80, 120, 200)
SEL_BG = (40, 70, 120)
FG = (235, 235, 240)
FG_DIM = (150, 154, 166)


class Label:
    def __init__(self, text: str, size: int = 14, color=FG):
        self.surf = render_text(text, size, color)

    def draw(self, screen: pygame.Surface, x: int, y: int) -> None:
        screen.blit(self.surf, (x, y))


class Button:
    def __init__(self, rect: tuple[int, int, int, int], label: str, key=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.key = key
        self._surf = render_text(label, 14, FG)

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def draw(self, screen: pygame.Surface, hot: bool = False) -> None:
        pygame.draw.rect(screen, BTN_HL if hot else BTN_BG, self.rect, border_radius=6)
        pygame.draw.rect(screen, FG_DIM, self.rect, 1, border_radius=6)
        tw, th = self._surf.get_size()
        screen.blit(self._surf, (self.rect.centerx - tw // 2, self.rect.centery - th // 2))


def draw_list(screen: pygame.Surface, rect: pygame.Rect, entries: list[str], selected: int,
              top: int = 0) -> None:
    """Draw a scrollable selection list of epoch rows. ``entries`` are label strings."""
    pygame.draw.rect(screen, PAD_BG, rect)
    row_h = 22
    y = rect.y + 6
    max_rows = max(1, (rect.h - 12) // row_h)
    for i in range(top, min(top + max_rows, len(entries))):
        row = pygame.Rect(rect.x + 6, y, rect.w - 12, row_h - 4)
        if i == selected:
            pygame.draw.rect(screen, SEL_BG, row, border_radius=4)
        color = FG if i == selected else FG_DIM
        lbl = render_text(entries[i], 13, color)
        screen.blit(lbl, (row.x + 6, row.y + 2))
        y += row_h


def list_hit(rect: pygame.Rect, pos, top: int = 0) -> int | None:
    """Return the list index hit by ``pos`` (accounting for scroll ``top``), else None."""
    if not rect.collidepoint(pos):
        return None
    row_h = 22
    rel = pos[1] - (rect.y + 6)
    if rel < 0:
        return None
    return top + rel // row_h
