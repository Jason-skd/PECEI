"""Headless renderer: Grid (+ Entities) -> pygame.Surface.

Opens no window. The preview shell (``preview.py``) is responsible for display;
live/replay (M7) and PNG export reuse this renderer via ``pygame.image.save``.

Note: pygame's font modules are unreliable on Python 3.14 (sysfont circular
import), so this renderer is font-free — tiles are distinguished by color plus
a small per-type shape marker. The text glyph mapping lives in the ASCII dump
(``Grid.ascii``) and in ``tiles.LABEL`` for reference.
"""
from __future__ import annotations

import pygame

from pecei.world.component import ComponentType
from pecei.world.entity import Entity
from pecei.world.grid import Grid

from . import tiles


class Renderer:
    """Render ground-truth world state to a pygame.Surface.

    ``use_images=True`` + an assets dir swaps the flat color blocks for PNG tiles
    (loaded once, scaled to the cell size); a missing tile falls back to the
    color+shape block, so images are strictly optional.
    """

    def __init__(self, tile: int = tiles.TILE, use_images: bool = False,
                 assets_dir: str | None = None) -> None:
        pygame.init()
        self.tile = tile
        self.use_images = use_images
        self._tiles: dict = {}
        if use_images:
            from pecei.replay.assets import load_tiles
            self._tiles = load_tiles(tile, assets_dir)

    def render(
        self,
        grid: Grid,
        entities: list[Entity] | None = None,
        goal: tuple[int, int] | None = None,
    ) -> pygame.Surface:
        w, h = grid.width * self.tile, grid.height * self.tile
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(tiles.BG)

        eid_to_entity = {e.eid: e for e in (entities or [])}

        # top occupant's entity per cell (for unified outlines + ego highlight).
        # Non-entity marker occupants (e.g. the GOAL stamp, eid="goal") are not
        # actors: they are rendered via the goal ring, not as a cell block.
        cell_eid: dict[tuple[int, int], str] = {}
        for y in range(grid.height):
            for x in range(grid.width):
                occs = [o for o in grid.occupants(x, y) if o.eid in eid_to_entity]
                if occs:
                    cell_eid[(x, y)] = occs[-1].eid

        for (x, y) in cell_eid:
            occs = [o for o in grid.occupants(x, y) if o.eid in eid_to_entity]
            self._draw_cell(surf, x, y, occs[-1].component.ctype, stacked=len(occs) > 1)

        self._draw_outlines(surf, cell_eid, eid_to_entity)
        for e in (entities or []):
            if e.has_brain:
                self._draw_facing(surf, e)

        # grid lines on top
        for x in range(grid.width + 1):
            pygame.draw.line(surf, tiles.GRID_LINE, (x * self.tile, 0), (x * self.tile, h))
        for y in range(grid.height + 1):
            pygame.draw.line(surf, tiles.GRID_LINE, (0, y * self.tile), (w, y * self.tile))

        # goal marker (image if available, else green ring)
        if goal is not None:
            gx, gy = goal
            if grid.in_bounds(gx, gy):
                img = self._tiles.get("goal")
                if img is not None:
                    surf.blit(img, (gx * self.tile, gy * self.tile))
                else:
                    cx = gx * self.tile + self.tile // 2
                    cy = gy * self.tile + self.tile // 2
                    pygame.draw.circle(surf, tiles.GOAL, (cx, cy), self.tile // 2 - 3, 3)

        return surf

    def _draw_cell(
        self, surf: pygame.Surface, x: int, y: int, ctype: ComponentType, stacked: bool
    ) -> None:
        t = self.tile
        img = self._tiles.get(ctype.value)
        if img is not None:
            surf.blit(img, (x * t, y * t))
            if stacked:
                pygame.draw.circle(surf, tiles.STACK_DOT, (x * t + t - 7, y * t + 7), 3)
            return
        inset = 4
        rect = pygame.Rect(x * t + inset, y * t + inset, t - 2 * inset, t - 2 * inset)
        color = tiles.COLOR[ctype]
        if ctype is ComponentType.WHEEL:
            pygame.draw.circle(surf, color, rect.center, rect.width // 2)
        else:
            pygame.draw.rect(surf, color, rect, border_radius=6)
        self._draw_mark(surf, rect, ctype)
        if stacked:
            pygame.draw.circle(surf, tiles.STACK_DOT, (x * t + t - 7, y * t + 7), 3)

    def _draw_mark(self, surf: pygame.Surface, rect: pygame.Rect, ctype: ComponentType) -> None:
        """Small white shape marker so types read even without text."""
        cx, cy = rect.center
        r = rect.width // 6
        m = tiles.LABEL_FG
        if ctype is ComponentType.BRAIN:
            pygame.draw.circle(surf, m, (cx, cy), r, 2)                       # ring
        elif ctype is ComponentType.FIRE:
            pygame.draw.polygon(surf, m, [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)])
        elif ctype is ComponentType.WATER:
            pygame.draw.line(surf, m, (cx - r, cy - 2), (cx + r, cy - 2), 2)
            pygame.draw.line(surf, m, (cx - r, cy + 2), (cx + r, cy + 2), 2)
        elif ctype is ComponentType.METAL:
            pygame.draw.line(surf, m, (cx - r, cy), (cx + r, cy), 2)          # plus
            pygame.draw.line(surf, m, (cx, cy - r), (cx, cy + r), 2)
        elif ctype is ComponentType.WOOD:
            pygame.draw.line(surf, m, (cx - r, cy), (cx + r, cy), 2)          # grain
        elif ctype is ComponentType.STONE:
            inner = rect.inflate(-rect.width // 2, -rect.height // 2)
            pygame.draw.rect(surf, m, inner, 1)                               # inner frame
        # WHEEL: the circular base tile is distinctive enough

    def _draw_outlines(
        self,
        surf: pygame.Surface,
        cell_eid: dict[tuple[int, int], str],
        eid_to_entity: dict[str, Entity],
    ) -> None:
        """Outline each entity as one shape: draw an edge only where the neighbor
        cell belongs to a different entity (so same-entity cells merge visually)."""
        t = self.tile
        edges = [
            (0, -1, lambda px, py: ((px, py), (px + t, py))),          # top
            (1, 0, lambda px, py: ((px + t, py), (px + t, py + t))),   # right
            (0, 1, lambda px, py: ((px, py + t), (px + t, py + t))),   # bottom
            (-1, 0, lambda px, py: ((px, py), (px, py + t))),          # left
        ]
        for (x, y), eid in cell_eid.items():
            ent = eid_to_entity.get(eid)
            color = tiles.EGO_OUTLINE if (ent and ent.is_ego) else tiles.OTHER_OUTLINE
            px, py = x * t, y * t
            for dx, dy, seg in edges:
                if cell_eid.get((x + dx, y + dy)) != eid:
                    p1, p2 = seg(px, py)
                    pygame.draw.line(surf, color, p1, p2, 2)

    def _draw_facing(self, surf: pygame.Surface, entity: Entity) -> None:
        """Draw a facing arrow on the brain cell along the entity's orientation."""
        t = self.tile
        brain_local = entity.brain_local()
        if brain_local is None:
            return
        brain_abs = next(
            (cell for cell, (local, _c) in entity.placements().items() if local == brain_local),
            None,
        )
        if brain_abs is None:
            return
        bx, by = brain_abs
        cx, cy = bx * t + t // 2, by * t + t // 2
        dx, dy = entity.orientation.delta
        tip = (cx + dx * (t // 3), cy + dy * (t // 3))
        base = (cx - dx * (t // 5), cy - dy * (t // 5))
        pygame.draw.line(surf, tiles.EGO_OUTLINE, base, tip, 3)
