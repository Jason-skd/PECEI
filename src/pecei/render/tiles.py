"""Tile rendering constants: palette, glyphs, sizes.

Borrowed in spirit from MiniGrid's tile rendering (one fixed-size tile per
cell, blitted into a grid image), but with our own component-typed palette.
"""
from __future__ import annotations

from pecei.world.component import ComponentType

TILE = 48  # default pixels per cell

BG = (24, 26, 32)
GRID_LINE = (44, 48, 58)
GOAL = (96, 220, 120)
EGO_OUTLINE = (245, 245, 245)
OTHER_OUTLINE = (130, 134, 146)
LABEL_FG = (245, 245, 245)
STACK_DOT = (245, 245, 245)

COLOR: dict[ComponentType, tuple[int, int, int]] = {
    ComponentType.STONE: (130, 130, 138),
    ComponentType.WOOD: (150, 100, 55),
    ComponentType.FIRE: (232, 93, 47),
    ComponentType.WATER: (64, 130, 201),
    ComponentType.WHEEL: (70, 72, 80),
    ComponentType.BRAIN: (214, 57, 128),
    ComponentType.METAL: (178, 182, 196),
}

LABEL: dict[ComponentType, str] = {
    ComponentType.STONE: "#",
    ComponentType.WOOD: "w",
    ComponentType.FIRE: "*",
    ComponentType.WATER: "~",
    ComponentType.WHEEL: "o",
    ComponentType.BRAIN: "@",
    ComponentType.METAL: "m",
}
