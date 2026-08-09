"""Generate the tile PNGs used by the image-based renderer.

Run:  ``uv run python -m pecei.render.make_assets``  -> writes ``render/assets/*.png``.

Tiles are procedurally drawn (Pillow) so they look like real sprites instead of
flat color blocks. The renderer loads and scales them to the cell size; a missing
tile falls back to the color+shape block, so the renderer never hard-depends on
these files.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

S = 64  # base tile size (px); scaled to the cell size at render time
OUT = Path(__file__).parent / "assets"


def _save(name: str, draw) -> None:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    draw(d)
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name)


def _base(d: ImageDraw.ImageDraw, color, radius=10) -> None:
    d.rounded_rectangle([6, 6, S - 6, S - 6], radius=radius, fill=color)


def stone(d):
    _base(d, (130, 130, 138))
    d.line([16, 24, 48, 24], fill=(90, 90, 100), width=3)
    d.line([16, 40, 48, 40], fill=(90, 90, 100), width=3)
    d.line([32, 10, 32, 24], fill=(90, 90, 100), width=3)
    d.line([24, 24, 24, 40], fill=(90, 90, 100), width=3)


def wood(d):
    _base(d, (150, 100, 55))
    for x in (20, 32, 44):
        d.line([x, 10, x, S - 10], fill=(110, 68, 34), width=3)


def fire(d):
    _base(d, (232, 93, 47))
    d.polygon([(32, 14), (18, 40), (32, 34), (46, 40)], fill=(250, 200, 80))
    d.polygon([(32, 26), (24, 46), (32, 42), (40, 46)], fill=(255, 240, 160))


def water(d):
    _base(d, (64, 130, 201))
    for y in (22, 34, 46):
        d.arc([14, y - 6, 50, y + 6], 0, 180, fill=(180, 220, 250), width=3)


def wheel(d):
    d.ellipse([8, 8, S - 8, S - 8], fill=(70, 72, 80))
    d.ellipse([20, 20, S - 20, S - 20], fill=(150, 152, 165))
    d.ellipse([27, 27, S - 27, S - 27], fill=(60, 62, 70))
    for a in range(0, 360, 60):
        import math
        x = 32 + 18 * math.cos(math.radians(a))
        y = 32 + 18 * math.sin(math.radians(a))
        d.line([32, 32, x, y], fill=(120, 122, 135), width=3)


def brain(d):
    _base(d, (214, 57, 128), radius=16)
    d.ellipse([18, 18, 46, 42], outline=(245, 245, 245), width=3)
    d.line([32, 42, 32, 52], fill=(245, 245, 245), width=3)


def metal(d):
    _base(d, (178, 182, 196))
    d.line([14, 14, 50, 50], fill=(120, 124, 140), width=4)
    d.line([50, 14, 14, 50], fill=(120, 124, 140), width=4)
    d.rectangle([24, 24, 40, 40], outline=(235, 238, 245), width=2)


def goal(d):
    d.ellipse([8, 8, S - 8, S - 8], outline=(96, 220, 120), width=5)
    d.ellipse([26, 26, 38, 38], fill=(96, 220, 120))


def empty(d):
    d.rectangle([0, 0, S, S], fill=(24, 26, 32, 255))


def main() -> None:
    for name, fn in [("stone", stone), ("wood", wood), ("fire", fire), ("water", water),
                     ("wheel", wheel), ("brain", brain), ("metal", metal), ("goal", goal),
                     ("empty", empty)]:
        _save(f"{name}.png", fn)
    print(f"wrote assets -> {OUT}")


if __name__ == "__main__":
    main()
