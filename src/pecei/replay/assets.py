"""Tile asset loading for the image-based renderer.

Loads ``replay/assets/<type>.png`` per component type (plus goal/empty), caches
them, and scales to the cell size. PNG decoding goes through **Pillow** (pygame's
``image.load`` can't read PNG on Python 3.14 — no SDL_image). Any missing tile is
simply absent from the dict so the renderer can fall back to its color+shape block.
"""
from __future__ import annotations

from pathlib import Path

import pygame

from pecei.world.component import ComponentType

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except Exception:  # pragma: no cover - Pillow is a declared dep
    _HAS_PIL = False

_DEFAULT_DIR = Path(__file__).resolve().parent / "assets"


def _load_png(path: Path) -> pygame.Surface:
    if _HAS_PIL:
        img = _PILImage.open(path).convert("RGBA")
        return pygame.image.frombytes(img.tobytes(), img.size, "RGBA")
    return pygame.image.load(str(path)).convert_alpha()  # pragma: no cover


def load_tiles(tile_size: int, assets_dir: str | Path | None = None) -> dict[str, pygame.Surface]:
    """Load + scale tiles to ``tile_size``. Keys: component-type values + 'goal'/'empty'.
    Missing/unreadable files are simply absent from the dict (renderer falls back)."""
    d = Path(assets_dir) if assets_dir else _DEFAULT_DIR
    out: dict[str, pygame.Surface] = {}
    names = [ct.value for ct in ComponentType] + ["goal", "empty"]
    for name in names:
        p = d / f"{name}.png"
        if not p.is_file():
            continue
        try:
            surf = _load_png(p)
            out[name] = pygame.transform.scale(surf, (tile_size, tile_size))
        except Exception:
            continue
    return out
