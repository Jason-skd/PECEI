"""Image export: pygame.Surface -> file.

pygame's built-in encoder can't write PNG on Python 3.14 (no SDL_image),
so PNG/JPG go through Pillow; BMP falls back to pygame (always available).
"""
from __future__ import annotations

import pygame

try:
    from PIL import Image as _PILImage  # type: ignore

    _HAS_PIL = True
except Exception:  # pragma: no cover - Pillow is a declared dep
    _HAS_PIL = False


def save_image(surf: pygame.Surface, path: str) -> str:
    """Save ``surf`` to ``path``. Returns the path actually written."""
    p = str(path)
    ext = p.rsplit(".", 1)[-1].lower() if "." in p else ""
    if _HAS_PIL and ext in ("png", "jpg", "jpeg"):
        w, h = surf.get_size()
        raw = pygame.image.tobytes(surf, "RGBA")
        img = _PILImage.frombytes("RGBA", (w, h), raw)
        if ext in ("jpg", "jpeg"):
            img = img.convert("RGB")
        img.save(p)
    else:
        if ext and ext != "bmp":
            p = p.rsplit(".", 1)[0] + ".bmp"
        pygame.image.save(surf, p)
    return p
