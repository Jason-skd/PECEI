import os
from pathlib import Path

# Headless: no window server needed for Surface/draw/image.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from pecei.render import Renderer, save_image  # noqa: E402
from pecei.world.map_parser import load_map_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "src" / "pecei" / "maps" / "example01.yaml"


def test_render_example_to_surface():
    spec = load_map_spec(EXAMPLE)
    renderer = Renderer(tile=24)
    surf = renderer.render(spec.build_grid(), spec.build_entities())
    assert isinstance(surf, pygame.Surface)
    assert surf.get_size() == (6 * 24, 5 * 24)  # 6x5 grid @ tile=24


def test_render_to_png(tmp_path):
    spec = load_map_spec(EXAMPLE)
    renderer = Renderer(tile=24)
    surf = renderer.render(spec.build_grid(), spec.build_entities())
    out = tmp_path / "example01.png"
    save_image(surf, str(out))
    assert out.exists() and out.stat().st_size > 0
