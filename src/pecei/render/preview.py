"""pygame shell: display a (static) world in a window.

This is the M2 verification surface — open a map and eyeball that it renders
into a sensible world. live/replay modes (M7) will extend this shell.
"""
from __future__ import annotations

import pygame

from pecei.world.entity import Entity
from pecei.world.grid import Grid

from .renderer import Renderer


def show(grid: Grid, entities: list[Entity] | None = None, title: str = "PECEI") -> None:
    """Open a window showing ``grid``. Close with ESC, Q, or the window button."""
    renderer = Renderer()
    frame = renderer.render(grid, entities)
    w, h = frame.get_size()
    pygame.display.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(title)
    screen.blit(frame, (0, 0))
    pygame.display.flip()

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
    pygame.display.quit()
