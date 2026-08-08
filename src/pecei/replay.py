"""Replay viewer: scrub through a recorded trace, rendering the world per step.

Ground truth is reconstructed from the map + the trace's recorded actions
(:func:`runner.world_at_step`). A draggable scrubber bar at the bottom shows the
timeline (ticks per round, bright ticks where the LLM made a decision); the
window caption carries the step/total + controls (caption is OS text, so it
works even though pygame's font is broken on py3.14). A terminal side-panel also
logs each step's action, observation, program, and LLM I/O.

Controls: →/n step forward · ←/p step back · Home/End jump · drag the bar · q/ESC quit.
"""
from __future__ import annotations

import pygame

from pecei.infra import Trace
from pecei.render import Renderer

from .runner import world_at_step

BAR_H = 36
_TRACK_X = 12  # scrubber track inset


def bar_step_at(mx: int, my: int, n: int, grid_w: int, grid_h: int, bar_h: int = BAR_H) -> int | None:
    """Map a mouse position to a timeline step, or None if outside the bar."""
    if not (grid_h <= my <= grid_h + bar_h):
        return None
    span = max(1, grid_w - 2 * _TRACK_X)
    frac = (mx - _TRACK_X) / span
    if not (0.0 <= frac <= 1.0):
        return None
    return max(0, min(n, round(frac * n)))


def replay_viewer(map_path: str, trace_path: str) -> None:
    trace = Trace.read(trace_path)
    n = len(trace.events)
    renderer = Renderer()
    step = 0

    def base_frame(s: int) -> pygame.Surface:
        w = world_at_step(map_path, trace, s)
        return renderer.render(w.grid, list(w.entities.values()), w.goal)

    def draw_bar(screen: pygame.Surface, s: int, grid_w: int, grid_h: int) -> None:
        pygame.draw.rect(screen, (16, 18, 24), (0, grid_h, grid_w, BAR_H))
        x0, x1 = _TRACK_X, grid_w - _TRACK_X
        y = grid_h + BAR_H // 2
        span = max(1, x1 - x0)
        pygame.draw.line(screen, (80, 84, 96), (x0, y), (x1, y), 3)
        for i in range(n + 1):                      # a tick per step
            x = x0 + (span * i) // span
            is_decision = i > 0 and trace.events[i - 1].program is not None
            color = (214, 57, 128) if is_decision else (90, 94, 106)
            h = 8 if is_decision else 5
            pygame.draw.line(screen, color, (x, y - h), (x, y + h), 2 if is_decision else 1)
        mx = x0 + (span * s) // span                 # current marker
        pygame.draw.circle(screen, (96, 220, 120), (mx, y), 5)

    def caption(s: int) -> str:
        return f"PECEI replay — step {s}/{n} — arrows / drag bar / Home / End — q quit"

    def draw(s: int) -> None:
        screen.fill((24, 26, 32))
        screen.blit(base_frame(s), (0, 0))
        draw_bar(screen, s, grid_w, grid_h)
        pygame.display.set_caption(caption(s))
        pygame.display.flip()
        _print_step(trace, s)

    pygame.display.init()
    grid_w, grid_h = base_frame(0).get_size()
    screen = pygame.display.set_mode((grid_w, grid_h + BAR_H))
    print("replay controls: →/← step · drag the bar · Home/End · q quit "
          "(per-step detail logs to this terminal)")
    draw(step)

    prev = -1
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key in (pygame.K_RIGHT, pygame.K_n) and step < n:
                    step += 1
                elif ev.key in (pygame.K_LEFT, pygame.K_p) and step > 0:
                    step -= 1
                elif ev.key == pygame.K_HOME:
                    step = 0
                elif ev.key == pygame.K_END:
                    step = n
            elif ev.type == pygame.MOUSEBUTTONDOWN and getattr(ev, "button", 1) == 1:
                target = bar_step_at(ev.pos[0], ev.pos[1], n, grid_w, grid_h)
                if target is not None:
                    step = target
            elif ev.type == pygame.MOUSEMOTION and ev.buttons[0]:
                target = bar_step_at(ev.pos[0], ev.pos[1], n, grid_w, grid_h)
                if target is not None:
                    step = target
        if step != prev:
            prev = step
            draw(step)
    pygame.display.quit()


def _print_step(trace: Trace, step: int) -> None:
    if step >= len(trace.events):
        print(f"\n--- step {step}: END (goal reached or terminal) ---")
        return
    ev = trace.events[step]
    print(
        f"\n=== step {step} | round {ev.round} | {ev.action} "
        f"moved={ev.moved} blocked={ev.blocked} | ego@{ev.anchor_after} "
        f"facing {ev.orientation_after} ==="
    )
    if ev.program:
        print("program (decision made here):")
        for line in ev.program.splitlines():
            print("    " + line)
    if ev.llm_request is not None or ev.llm_response is not None:
        print(f"llm_i/o: request={'yes' if ev.llm_request else 'no'} "
              f"response={'yes' if ev.llm_response else 'no'}")
    obs = ev.observation
    if obs:
        cells = obs.get("cells", {})
        summary = ", ".join(
            f"{rel}:{','.join(cv.get('types') or ['empty'])}" for rel, cv in sorted(cells.items())
        )
        print(f"observed (facing {obs.get('orientation')}): {summary}")
