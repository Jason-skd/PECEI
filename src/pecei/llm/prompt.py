"""Prompt rendering shared by all providers (keeps adapters thin + consistent).

The author is blind to the live simulation AND to the full map: ``render_user``
shows only a PARTIAL seed observation from the start pose (the 90° FOV cone),
the previous cycle's feedback (the script + its yields + stop-report), and the
snowball of earlier cycles — never the god-view layout, never a live per-round
observation.
"""
from __future__ import annotations

from pathlib import Path

from pecei.action import Program

from .protocol import Directive, TurnInput

# The plan tool's input schema = the canonical Program AST. Non-strict tool-use
# accepts the recursive discriminated union (strict structured-output would not).
PROGRAM_SCHEMA = Program.model_json_schema()

PLAN_TOOL_DESCRIPTION = (
    "Emit the agent's COMPLETE script for this cycle — a sequence of statements "
    "that runs autonomously from the start until it stops. The script runs blind "
    "(you receive no feedback until it stops). Statements:\n"
    "  act(FORWARD|BACKWARD|TURNLEFT|TURNRIGHT)  # MOVE — the ONLY valid act args\n"
    "  ob = beat(OBSERVE)                        # sense now; MUST assign to a variable\n"
    "  beat(YIELD, ob)                           # report an observation (fed back to you)\n"
    "  b = ob.front.is_blocked                   # bool predicate\n"
    "  if <bool_var>: ... else: ...              # condition MUST be a bare bool variable\n"
    "TWO VERBS, never mix their arguments: `act` takes ONLY a movement "
    "(FORWARD/BACKWARD/TURNLEFT/TURNRIGHT); `beat` takes ONLY OBSERVE or YIELD. "
    "So `act(YIELD)` and `beat(FORWARD)` are ALWAYS wrong. RULE: an if-condition "
    "is a bool *variable name* (string), never an expression; compute any "
    "predicate into a bool variable first. beat(OBSERVE) must be assigned to a "
    "variable before its attributes are used."
)

# The author's system prompt lives in a co-located markdown file so it can be
# edited as prose without touching Python. ``_SYSTEM_PROMPT_FILE`` resolves
# relative to this module, so it works both from a source checkout and inside an
# installed wheel (uv_build ships the .md with the package by default).
_SYSTEM_PROMPT_FILE = Path(__file__).resolve().parent / "system prompt.md"


def load_system_prompt() -> str:
    """Return the author system prompt, read fresh from ``system prompt.md``.

    Read on every call (not cached): developers can edit the markdown and the
    next cycle picks up the change without a restart, and the per-cycle trace in
    ``runner`` captures exactly what was sent. The entire file *is* the prompt —
    nothing is stripped — so keep developer notes out of it or they reach the
    model. A missing file raises with the resolved path rather than silently
    sending an empty prompt.
    """
    try:
        return _SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError as e:  # pragma: no cover - fail loud, not silent
        raise FileNotFoundError(
            f"System prompt file not found: {_SYSTEM_PROMPT_FILE}"
        ) from e


def _fmt_obs(y: dict) -> str:
    cells = y.get("cells", {})
    if not cells:
        return ""
    body = ", ".join(f"{k}:{'/'.join(v.get('types') or ['empty'])}" for k, v in cells.items())
    return f"facing {y.get('orientation')} at {y.get('anchor')}: {body}"


def render_user(turn: TurnInput) -> str:
    lines = [f"DIRECTIVE: {turn.directive.value}"]

    if turn.instructions:
        lines.append(f"INSTRUCTIONS (authoritative): {turn.instructions}")

    seed = turn.seed_observation
    if seed:
        lines.append(f"goal: {seed.get('goal')}")
        s = _fmt_obs(seed)
        if s:
            lines.append(f"seed observation: {s}")

    fb = turn.feedback
    if fb:
        lines.append(f"last_cycle stopped: {fb.stop_reason.value} after {fb.rounds_used} rounds.")
        if fb.script:
            lines.append("  script:")
            for ln in str(fb.script).splitlines():
                lines.append(f"    {ln}")
        if fb.compile_error:
            lines.append(f"  compile error: {fb.compile_error}")
        if fb.failure_snapshot:
            lines.append(f"  ended near {tuple(fb.failure_snapshot.pos)}, "
                         f"complexity {fb.failure_snapshot.complexity}.")
        if fb.yielded:
            lines.append("you yielded:")
            for y in fb.yielded:
                s = _fmt_obs(y)
                if s:
                    lines.append(f"  {s}")

    if turn.snowball:
        lines.append("prior_cycles (snowball — learn from these):")
        for c in turn.snowball:
            lines.append(f"  cycle {c.get('index')}: {c.get('stop_reason')} in {c.get('rounds')} rounds")
            if c.get("error"):
                lines.append(f"    compile error: {c.get('error')}")
            for s in c.get("scripts") or []:
                lines.append("    script:")
                for ln in str(s).splitlines():
                    lines.append(f"      {ln}")

    if turn.directive is Directive.PLAN:
        lines.append("Call the `plan` tool with your complete script.")
    return "\n".join(lines)
