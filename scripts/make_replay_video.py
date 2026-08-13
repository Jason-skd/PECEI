"""Headless replay-video exporter: existing compare session traces -> mp4.

Storyboard (warm-vs-cold narrative, ~50s @ 1280x720 / 30fps):
  0. title card
  1. corridor (warm_train) — clean 5-round solve: establishes what success looks like
  2. two_walls (warm) — 25-epoch banner montage (the regenerate loop) + a late-epoch
     exploration clip: behavior evolves across epochs, still no solve
  3. detour (cold) — 25 failing banners + a stuck-on-one-cell clip
  4. detour (warm) — epoch 1 full solve, 49 rounds
  5. stat card + end card

Usage (from repo root):
    .venv/bin/python scripts/make_replay_video.py [out.mp4]

Requires only project deps (pygame, Pillow) and system ffmpeg.
"""
from __future__ import annotations

import os

# Must precede any pygame import: render headlessly.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import subprocess
import sys
import tempfile
from pathlib import Path

import pygame

from pecei.infra import Trace
from pecei.render import Renderer
from pecei.render.io import save_image
from pecei.render.text import render_text
from pecei.replay.model import build_from_traces
from pecei.session import Session

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # session maps/traces are stored relative to the repo root

W, H = 1280, 720
FPS = 30
TILE = 72
GRID_X, GRID_Y = 16, 84

BG = (16, 18, 24)
HEADER_BG = (20, 22, 30)
FG = (235, 235, 240)
FG_DIM = (150, 154, 166)
GREEN = (96, 220, 120)
ORANGE = (235, 160, 80)
PINK = (214, 57, 128)
TRACK = (44, 48, 58)

REASON_COLOR = {
    "SUCCESS": GREEN,
    "ROUND_LIMIT_EXCEED": ORANGE,
    "COMPILE_ERROR": PINK,
    "SCRIPT_ENDED": FG_DIM,
    "ENERGY_RUN_OUT": PINK,
}

RENDERER = Renderer(tile=TILE)


# ---------------------------------------------------------------- primitives

def text(surf: pygame.Surface, s: str, x: int, y: int, size: int = 20,
         color: tuple = FG) -> None:
    surf.blit(render_text(s, size, color), (x, y))


def badge(surf: pygame.Surface, s: str, x: int, y: int, color: tuple,
          size: int = 22) -> int:
    """Draw a pill badge; return its height (for stacking)."""
    img = render_text(s, size, color)
    tw, th = img.get_size()
    pad = 10
    rect = pygame.Rect(x, y - 2, tw + 2 * pad, th + 8)
    pygame.draw.rect(surf, (34, 38, 50), rect, border_radius=8)
    pygame.draw.rect(surf, color, rect, 1, border_radius=8)
    surf.blit(img, (x + pad, y + 2))
    return rect.height


def header(surf: pygame.Surface, title: str, sub: str) -> None:
    pygame.draw.rect(surf, HEADER_BG, (0, 0, W, 68))
    text(surf, title, 20, 14, 24, FG)
    text(surf, sub, 20, 44, 16, FG_DIM)


def card(lines: list[tuple[str, int, tuple]]) -> pygame.Surface:
    """Full-canvas text card: centered block of (text, size, color) lines."""
    s = pygame.Surface((W, H))
    s.fill(BG)
    pygame.draw.rect(s, GREEN, (0, 0, 6, H))
    total = sum(sz + 14 for _, sz, _ in lines) - 14
    y = (H - total) // 2
    for txt, size, color in lines:
        s.blit(render_text(txt, size, color), (90, y))
        y += size + 14
    return s


def divider(kicker: str, title: str, sub: str) -> pygame.Surface:
    s = pygame.Surface((W, H))
    s.fill(BG)
    pygame.draw.rect(s, GREEN, (0, 0, 6, H))
    text(s, kicker, 90, 236, 22, GREEN)
    text(s, title, 90, 276, 44, FG)
    text(s, sub, 90, 344, 22, FG_DIM)
    return s


# ---------------------------------------------------------------- frames

def step_frame(fr, title: str, sub: str, epoch_total: int) -> pygame.Surface:
    """One ground-truth world frame + side panel (epoch, reason, rounds, script)."""
    s = pygame.Surface((W, H))
    s.fill(BG)
    header(s, title, sub)
    world = fr.world
    grid_surf = RENDERER.render(world.grid, list(world.entities.values()), world.goal)
    s.blit(grid_surf, (GRID_X, GRID_Y))
    gw, gh = grid_surf.get_size()

    # intra-epoch progress bar
    bar_y = GRID_Y + gh + 16
    pygame.draw.rect(s, TRACK, (GRID_X, bar_y, gw, 6), border_radius=3)
    frac = fr.step / max(1, fr.n_steps)
    if frac > 0:
        color = REASON_COLOR.get(fr.stop_reason, FG_DIM)
        pygame.draw.rect(s, color, (GRID_X, bar_y, int(gw * frac), 6), border_radius=3)

    # side panel
    px = GRID_X + gw + 40
    y = GRID_Y
    text(s, Path(fr.map_path).name, px, y, 16, FG_DIM)
    y += 28
    text(s, f"epoch {fr.epoch} / {epoch_total}", px, y, 30)
    y += 44
    y += badge(s, fr.stop_reason, px, y, REASON_COLOR.get(fr.stop_reason, FG_DIM), 20) + 10
    text(s, f"round {fr.step} / {fr.n_steps}", px, y, 22)
    y += 32
    ev = fr.event
    if ev is not None:
        ok = bool(getattr(ev, "moved", False))
        text(s, f"act({ev.action})", px, y, 20, FG)
        y += 28
        text(s, "moved" if ok else "blocked", px, y, 18, GREEN if ok else ORANGE)
        y += 34
    text(s, "script", px, y, 16, FG_DIM)
    y += 24
    for ln in fr.script.splitlines()[:8]:
        text(s, ln[:44], px, y, 14, FG_DIM)
        y += 19
    return s


def success_flash(fr, title: str, sub: str, epoch_total: int) -> pygame.Surface:
    s = step_frame(fr, title, sub, epoch_total)
    gw = fr.world.grid.width * TILE
    cx = GRID_X + gw // 2
    img = render_text("SUCCESS", 34, GREEN)
    tw, th = img.get_size()
    rect = pygame.Rect(cx - tw // 2 - 18, GRID_Y + 16, tw + 36, th + 20)
    pygame.draw.rect(s, (24, 40, 28), rect, border_radius=10)
    pygame.draw.rect(s, GREEN, rect, 2, border_radius=10)
    s.blit(img, (rect.x + 18, rect.y + 8))
    return s


def banner_frame(epoch: int, n_epochs: int, stop: str, script: str, rounds: int,
                 title: str, sub: str) -> pygame.Surface:
    """Text card for one epoch in a montage (no grid)."""
    s = pygame.Surface((W, H))
    s.fill(BG)
    header(s, title, sub)
    text(s, f"epoch {epoch}", 140, 150, 60, FG)
    text(s, f"/ {n_epochs}", 150, 208, 24, FG_DIM)
    y = 190
    y += badge(s, stop, 330, y, REASON_COLOR.get(stop, FG_DIM), 26) + 14
    text(s, f"{rounds} rounds", 330, y, 22, FG_DIM)
    px = 760
    y = 150
    text(s, "script", px, y, 16, FG_DIM)
    y += 26
    for ln in script.splitlines()[:10]:
        text(s, ln[:56], px, y, 15, FG_DIM)
        y += 21
    return s


# ---------------------------------------------------------------- assembly

class Film:
    """Collects (surface, seconds) and encodes via ffmpeg concat demuxer."""

    def __init__(self, out: Path) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="pecei_frames_"))
        self.out = Path(out)
        self.manifest: list[tuple[Path, float]] = []

    def emit(self, surf: pygame.Surface, sec: float) -> None:
        p = self.dir / f"f_{len(self.manifest):05d}.png"
        save_image(surf, str(p))
        self.manifest.append((p, sec))

    def finish(self) -> None:
        concat = self.dir / "concat.txt"
        with open(concat, "w") as fh:
            for p, sec in self.manifest:
                fh.write(f"file '{p}'\nduration {sec:.3f}\n")
            fh.write(f"file '{self.manifest[-1][0]}'\n")  # concat needs a trailing file
        self.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(concat),
             "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "19", "-movflags", "+faststart", str(self.out)],
            check=True,
        )
        total = sum(sec for _, sec in self.manifest)
        print(f"wrote {self.out}: {len(self.manifest)} unique frames, ~{total:.1f}s")


def _clip(session_path: str, cycle_index: int, title: str, sub: str,
          epoch_total: int, film: Film, dur: float, stop_round: int | None = None,
          hold_success: float | None = None) -> None:
    """Replay one cycle's step frames. stop_round limits rounds; hold_success
    adds a SUCCESS flash hold after a solved clip."""
    sess = Session.load(ROOT / session_path)
    c = sess.cycles[cycle_index - 1]
    model = build_from_traces(sess.map, [(c.index, Trace.read(c.trace_path),
                                          c.stop_reason.value, c.script)])
    ep = model.epochs[0][1:]  # drop banner
    for fr in ep[1 : (stop_round + 1 if stop_round else None)]:
        film.emit(step_frame(fr, title, sub, epoch_total), dur)
    if hold_success:
        film.emit(success_flash(ep[-1], title, sub, epoch_total), hold_success)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "compare" / "replay_video.mp4"
    film = Film(out)

    # 0 — title
    film.emit(card([
        ("PECEI prototype", 56, FG),
        ("how one AI learns to navigate by regenerating its own scripts", 26, FG_DIM),
        ("grid rescue world · each epoch = one blind-run script · no gradients", 20, FG_DIM),
    ]), 2.2)

    # 1 — corridor: the cleanest trained behavior
    _clip("compare/warm_train/corridor.session.json", 1,
          "BASELINE · map 01 corridor",
          "warm-up: 'walk until the goal' — solved in 5 rounds",
          1, film, 0.40, hold_success=1.4)

    # 2 — the regenerate loop on a harder map
    film.emit(divider("1", "THE REGENERATE LOOP",
                      "each epoch: one new script, run blind, then critiqued — no gradients"), 1.8)
    sess = Session.load(ROOT / "compare/warm/two_walls.session.json")
    title = "map 06 two_walls — two staggered walls, one zig-zag"
    sub = "25 epochs on this harder map"
    for c in sess.cycles:
        film.emit(banner_frame(c.index, len(sess.cycles), c.stop_reason.value,
                               c.script, c.rounds, title, sub), 0.30)
    _clip("compare/warm/two_walls.session.json", 19,
          "epoch 19 — the script now explores 26 cells",
          "still no goal, but no longer stuck",
          25, film, 0.15, stop_round=26)

    # 3 — cold start on detour: every epoch fails
    film.emit(divider("2", "COLD START — no prior maps",
                      "same map: 25 epochs, 1,209 rounds — never reaches the goal"), 1.8)
    sess = Session.load(ROOT / "compare/cold/detour.session.json")
    title = "map 05 detour — goal visible, one wall to sidestep"
    sub = "cold: 25 epochs, every one fails"
    for c in sess.cycles:
        film.emit(banner_frame(c.index, len(sess.cycles), c.stop_reason.value,
                               c.script, c.rounds, title, sub), 0.22)
    _clip("compare/cold/detour.session.json", 10,
          "epoch 10 — stuck on one cell",
          "act(FORWARD) -> blocked, again and again",
          25, film, 0.15, stop_round=21)

    # 4 — warm start on detour: epoch 1 solves
    film.emit(divider("3", "WARM START — trained on maps 01-04",
                      "same map, same agent: epoch 1, 49 rounds — solved"), 1.8)
    _clip("compare/warm/detour.session.json", 1,
          "warm · map 05 detour — epoch 1",
          "37 distinct cells explored, then the goal",
          1, film, 0.28, hold_success=1.4)

    # 5 — the numbers
    film.emit(card([
        ("map 05 detour", 24, FG_DIM),
        ("COLD    25 epochs  ·  1,209 rounds  ·  0 solved", 30, ORANGE),
        ("WARM    1 epoch    ·  49 rounds     ·  solved", 30, GREEN),
        ("same map · same agent · trained vs untrained", 24, FG_DIM),
    ]), 3.2)

    # 6 — end
    film.emit(card([
        ("generate -> test -> regenerate", 46, FG),
        ("one general AI · many specialized minds", 26, FG_DIM),
        ("data: compare/warm vs compare/cold — 302 per-epoch trace recordings", 18, FG_DIM),
    ]), 3.0)

    film.finish()


if __name__ == "__main__":
    main()
