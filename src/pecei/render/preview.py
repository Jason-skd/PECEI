"""pygame shell: display a (static) world in a window.

M2/M3 verification surface — open a level and eyeball that it renders into a
sensible world (entities, outlines, facing, goal). live/replay (M7) will extend
this shell.
"""
from __future__ import annotations

import pygame

from pecei.world.world import World

from .renderer import Renderer


def show(world: World, title: str = "PECEI") -> None:
    """Open a window showing ``world``. Close with ESC, Q, or the window button."""
    renderer = Renderer()
    frame = renderer.render(world.grid, list(world.entities.values()), world.goal)
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
