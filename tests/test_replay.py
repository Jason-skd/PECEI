"""Replay model + asset-fallback tests (headless; no window opened)."""
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pecei.infra import Result, Trace
from pecei.llm import MockProvider
from pecei.replay import model as M
from pecei.render.assets import load_tiles
from pecei.render import Renderer
from pecei.session import Session

REPO = Path(__file__).resolve().parents[1]
EXAMPLE02 = REPO / "src" / "pecei" / "maps" / "example02.yaml"


def _make_session(tmp_path, cycles=3):
    s = Session(map=str(EXAMPLE02), provider="mock", round_budget=50)
    for _ in range(cycles):
        s.run_one_cycle(MockProvider(), trace_dir=tmp_path / "tr")
    return s


def test_load_tiles_present_and_missing(tmp_path):
    full = load_tiles(48)                       # default assets dir
    assert "stone" in full and "goal" in full
    assert full["stone"].get_size() == (48, 48)
    missing = load_tiles(48, tmp_path / "nope")  # empty dir -> fallback (no tiles)
    assert missing == {}


def test_renderer_image_and_shape_paths(tmp_path):
    from pecei.world import load_world
    w = load_world(str(EXAMPLE02))
    img = Renderer(use_images=True).render(w.grid, list(w.entities.values()), w.goal)
    shape = Renderer().render(w.grid, list(w.entities.values()), w.goal)
    assert img.get_size() == shape.get_size() == (w.grid.width * 48, w.grid.height * 48)


def test_model_flatten_single_trace(tmp_path):
    s = _make_session(tmp_path, cycles=1)
    trace_path = s.cycles[0].trace_path
    model = M.build_from_session(str(EXAMPLE02), s)
    # 1 epoch = 1 banner + (rounds+1) step frames
    rounds = len(Trace.read(trace_path).events)
    assert len(model.epochs) == 1
    assert len(model.epochs[0]) == 1 + rounds + 1
    assert model.epoch_starts() == [0]


def test_model_flatten_session_epochs(tmp_path):
    s = _make_session(tmp_path, cycles=3)
    model = M.build_from_session(str(EXAMPLE02), s)
    assert len(model.epochs) == 3
    starts = model.epoch_starts()
    assert starts[0] == 0 and starts[1] > starts[0] and starts[2] > starts[1]
    # each epoch begins with a banner frame carrying the stop_reason
    for s0, c in zip(starts, s.cycles):
        fr = model.frames[s0]
        assert fr.kind == "banner" and fr.epoch == c.index
        assert fr.stop_reason == c.stop_reason.value
