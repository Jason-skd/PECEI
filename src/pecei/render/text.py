"""Text rendering for the replay UI.

pygame's font modules are broken on Python 3.14 (sysfont/Font circular import),
so UI text is rendered with **Pillow** to a bitmap and converted to a pygame
Surface. Falls back to a plain rectangle if Pillow is unavailable.
"""
from __future__ import annotations

import pygame

try:
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except Exception:  # pragma: no cover - Pillow is a declared dep
    _HAS_PIL = False


def _pil_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    for name in ("menlo", "monaco", "courier"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_text(text: str, size: int = 14, color=(235, 235, 240)) -> pygame.Surface:
    """Render ``text`` to a transparent pygame Surface (multiline-aware)."""
    lines = text.splitlines() or [""]
    if not _HAS_PIL:
        w = max(1, max(len(l) for l in lines) * (size // 2))
        h = len(lines) * (size + 2)
        return pygame.Surface((w, h), pygame.SRCALPHA)
    font = _pil_font(size)
    ascent = size + 2
    widths = []
    probe = Image.new("RGBA", (8, 8))
    pd = ImageDraw.Draw(probe)
    for l in lines:
        try:
            widths.append(int(pd.textlength(l, font=font)))
        except Exception:
            widths.append(len(l) * (size // 2))
    w = max(1, max(widths, default=1))
    h = len(lines) * ascent + 4
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = 2
    for l in lines:
        d.text((0, y), l, font=font, fill=tuple(color) + (255,))
        y += ascent
    raw = img.tobytes()
    return pygame.image.frombytes(raw, (w, h), "RGBA")
