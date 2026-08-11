"""Replay application: epoch-based viewer + session browser.

Base replay unit is an **epoch** (one cycle's trace). Two entry points:
- ``replay_viewer(map, trace)`` — replay a single epoch (one trace file).
- ``replay_session(map, session)`` — list a session's epochs, pick one, replay it,
  or press the PLAY→END button (or `c`) to play from the current epoch to the end.

Keys: ←/→ (n/p) step · Home/End jump · PgUp/PgDn switch epoch · SPACE play/pause ·
c play-to-end · q/ESC quit. Mouse: drag the timeline bar, click epoch buttons.
"""
from __future__ import annotations

from pathlib import Path

import pygame

from pecei.render import Renderer

from . import model as M
from . import ui

BAR_H = 40
BTN_H = 34
_TRACK_X = 12
PLAY_MS = 600  # auto-advance interval when playing


class _App:
    def __init__(self, model: M.ReplayModel) -> None:
        self.model = model
        self.renderer = Renderer()
        self.frames = model.frames
        self.starts = model.epoch_starts()
        self.i = self.starts[0] if self.starts else 0  # current flat-frame index
        self.playing = False
        self.to_end = False
        self._last_advance = 0

    @property
    def cur(self) -> M.Frame:
        return self.frames[self.i]

    def epoch_of(self, i: int) -> int:
        """0-based epoch index containing flat-frame ``i``."""
        k = 0
        for j, s in enumerate(self.starts):
            if i >= s:
                k = j
        return k

    def goto_epoch(self, epoch0: int) -> None:
        epoch0 = max(0, min(epoch0, len(self.model.epochs) - 1))
        self.i = self.starts[epoch0]

    def step(self, d: int) -> None:
        self.i = max(0, min(self.i + d, len(self.frames) - 1))

    def on_last_frame(self) -> bool:
        return self.i >= len(self.frames) - 1


def _step_at(mx: int, my: int, n: int, grid_w: int, grid_h: int) -> int | None:
    if not (grid_h <= my <= grid_h + BAR_H):
        return None
    span = max(1, grid_w - 2 * _TRACK_X)
    frac = (mx - _TRACK_X) / span
    if not (0.0 <= frac <= 1.0):
        return None
    return max(0, min(n, round(frac * n)))


def _draw_banner(screen: pygame.Surface, fr: M.Frame) -> None:
    ui.Label(f"epoch {fr.epoch} / {fr.n_epochs}   ·   {fr.stop_reason}   ·   {fr.n_steps} rounds",
             16).draw(screen, 16, 16)
    if fr.script:
        ui.Label("script:", 13, ui.FG_DIM).draw(screen, 16, 44)
        ui.Label(fr.script, 13).draw(screen, 24, 62)


def _run_app(app: _App, title: str) -> None:
    probe = next((f.world for f in app.frames if f.world is not None), None)
    base = app.renderer.render(probe.grid, list(probe.entities.values()), probe.goal)
    grid_w, grid_h = base.get_size()
    win_w = max(grid_w, 560)
    win_h = grid_h + BAR_H + BTN_H

    pygame.display.init()
    screen = pygame.display.set_mode((win_w, win_h))
    buttons = {
        "prev": ui.Button((8, grid_h + BAR_H + 4, 90, 26), "< epoch"),
        "next": ui.Button((104, grid_h + BAR_H + 4, 90, 26), "epoch >"),
        "play": ui.Button((200, grid_h + BAR_H + 4, 90, 26), "play"),
        "end": ui.Button((296, grid_h + BAR_H + 4, 140, 26), "play to end (c)"),
    }

    def draw() -> None:
        screen.fill(ui.PAD_BG)
        fr = app.cur
        if fr.kind == "banner":
            _draw_banner(screen, fr)
        else:
            surf = app.renderer.render(fr.world.grid, list(fr.world.entities.values()), fr.world.goal)
            screen.blit(surf, (0, 0))
        _draw_bar(screen, app, grid_h, win_w)
        for b in buttons.values():
            b.draw(screen)
        pygame.display.set_caption(
            f"{title} — epoch {fr.epoch}/{fr.n_epochs} · {fr.kind} {fr.step}/{fr.n_steps}")
        pygame.display.flip()

    clock = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key in (pygame.K_RIGHT, pygame.K_n):
                    app.step(1)
                elif ev.key in (pygame.K_LEFT, pygame.K_p):
                    app.step(-1)
                elif ev.key == pygame.K_HOME:
                    app.i = 0
                elif ev.key == pygame.K_END:
                    app.i = len(app.frames) - 1
                elif ev.key == pygame.K_PAGEUP:
                    app.goto_epoch(app.epoch_of(app.i) - 1)
                elif ev.key == pygame.K_PAGEDOWN:
                    app.goto_epoch(app.epoch_of(app.i) + 1)
                elif ev.key == pygame.K_SPACE:
                    app.playing = not app.playing
                    app.to_end = False
                elif ev.key == pygame.K_c:
                    app.playing = True
                    app.to_end = True
            elif ev.type == pygame.MOUSEBUTTONDOWN and getattr(ev, "button", 1) == 1:
                target = _step_at(ev.pos[0], ev.pos[1], len(app.frames) - 1, win_w, grid_h)
                if target is not None:
                    app.i = target
                elif buttons["prev"].hit(ev.pos):
                    app.goto_epoch(app.epoch_of(app.i) - 1)
                elif buttons["next"].hit(ev.pos):
                    app.goto_epoch(app.epoch_of(app.i) + 1)
                elif buttons["play"].hit(ev.pos):
                    app.playing = not app.playing
                    app.to_end = False
                elif buttons["end"].hit(ev.pos):
                    app.playing = True
                    app.to_end = True
            elif ev.type == pygame.MOUSEMOTION and ev.buttons[0]:
                target = _step_at(ev.pos[0], ev.pos[1], len(app.frames) - 1, win_w, grid_h)
                if target is not None:
                    app.i = target

        if app.playing:
            now = pygame.time.get_ticks()
            if now - app._last_advance >= PLAY_MS:
                app._last_advance = now
                if app.on_last_frame():
                    app.playing = False
                    app.to_end = False
                else:
                    app.step(1)
        draw()
        clock.tick(30)
    pygame.display.quit()


def _draw_bar(screen: pygame.Surface, app: _App, grid_h: int, win_w: int) -> None:
    y0 = grid_h + BAR_H // 2
    pygame.draw.rect(screen, ui.PAD_BG, (0, grid_h, win_w, BAR_H))
    x0, x1 = _TRACK_X, win_w - _TRACK_X
    span = max(1, x1 - x0)
    pygame.draw.line(screen, (80, 84, 96), (x0, y0), (x1, y0), 3)
    n = len(app.frames) - 1
    for i in range(n + 1):
        x = x0 + (span * i) // max(1, n)
        is_epoch_start = i in app.starts
        color = (214, 57, 128) if is_epoch_start else (90, 94, 106)
        h = 9 if is_epoch_start else 5
        pygame.draw.line(screen, color, (x, y0 - h), (x, y0 + h), 2 if is_epoch_start else 1)
    mx = x0 + (span * app.i) // max(1, n)
    pygame.draw.circle(screen, (96, 220, 120), (mx, y0), 5)


def replay_viewer(map_path: str, trace_path: str) -> None:
    """Replay a single epoch (one trace file)."""
    model = M.build_from_trace(map_path, trace_path)
    _run_app(_App(model), title=f"PECEI replay — {Path(trace_path).name}")


def replay_session(map_path: str, session) -> None:
    """Browse a session's epochs, pick one to start at, replay; PLAY→END (`c` /
    button) plays from the current epoch through to the end."""
    from pecei.session import Session

    if isinstance(session, (str, Path)):
        session = Session.load(session)
    if not session.cycles:
        raise SystemExit("session has no cycles to replay")

    # terminal epoch selector: list epochs, let the debugger choose a start epoch
    print("epochs:")
    for c in session.cycles:
        print(f"  {c.index:2}: {c.stop_reason.value:16} rounds={c.rounds}")
    try:
        choice = input("start at epoch [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""
    start_idx = int(choice) - 1 if choice.isdigit() else 0
    start_idx = max(0, min(start_idx, len(session.cycles) - 1))

    model = M.build_from_session(map_path, session)
    app = _App(model)
    app.goto_epoch(start_idx)
    _run_app(app, f"PECEI session — {Path(session.map).stem} ({len(session.cycles)} epochs)")
