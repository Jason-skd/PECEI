"""Prompt rendering shared by all providers (keeps adapters thin + consistent).

The author is blind to the live simulation: ``render_user`` shows only the static
map, the previous cycle's feedback (yields + stop-report), and the snowball —
never a live per-round observation.
"""
from __future__ import annotations

from pecei.action import Program

from .protocol import Directive, TurnInput

# The plan tool's input schema = the canonical Program AST. Non-strict tool-use
# accepts the recursive discriminated union (strict structured-output would not).
PROGRAM_SCHEMA = Program.model_json_schema()

PLAN_TOOL_DESCRIPTION = (
    "Emit the agent's COMPLETE script for this cycle — a sequence of statements "
    "that runs autonomously from the start until it stops. Statements: "
    "act(FORWARD|BACKWARD|TURNLEFT|TURNRIGHT), ob = beat(OBSERVE), "
    "beat(YIELD, ob), b = <bool predicate>, if <bool_var>: ... else: .... "
    "RULE: an if-condition must be a bool *variable name* (string), never an "
    "expression; compute any predicate into a bool variable first. "
    "beat(OBSERVE) must be assigned to a variable before its attributes are used. "
    "The script runs blind — you receive no feedback until it stops."
)

SYSTEM_PROMPT = """\
You author a policy script for a grid rescue agent. Each cycle you emit ONE
complete script via the `plan` tool; it then runs autonomously from the start
until it stops. You get NO live feedback while it runs.

You are blind to the live simulation. You learn ONLY from what is fed back after
each script stops: the observations you chose to beat(YIELD), and the stop-report
(why it stopped: SUCCESS / ROUND_LIMIT_EXCEED / ENERGY_RUN_OUT / SCRIPT_ENDED).
Use that feedback to write a better script next cycle.

Minimal language:
  act(FORWARD|BACKWARD|TURNLEFT|TURNRIGHT)   # one round each
  ob = beat(OBSERVE)                          # sense now; MUST assign to a variable
  beat(YIELD, ob)                             # report an observation (fed back to you)
  b = ob.front.is_blocked                     # bool predicate
  if <bool_var>: ... else: ...                # condition MUST be a bare bool variable

Cell predicates (relative to facing: front/left/right/back/here):
  is_fire is_water is_stone is_wood is_metal is_wheel is_brain is_empty is_blocked is_goal
  and .ctype (string) for equality checks.

Author a full script that reaches the goal from the start pose given in the map.
"""


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

    md = turn.map_desc
    if md:
        ego = md.get("ego") or {}
        lines.append(
            f"map: {md.get('width')}x{md.get('height')}, goal {md.get('goal')}, "
            f"start {ego.get('anchor')} facing {ego.get('orientation')}.")
        ents = md.get("entities")
        if ents:
            lines.append("layout (anchor: types):")
            for e in ents:
                lines.append(f"  {e.get('anchor')}: {','.join(e.get('types') or []) or 'empty'}")

    fb = turn.feedback
    if fb:
        lines.append(f"last_cycle stopped: {fb.stop_reason.value} after {fb.rounds_used} rounds.")
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
            for s in c.get("scripts") or []:
                lines.append("    script:")
                for ln in str(s).splitlines():
                    lines.append(f"      {ln}")

    if turn.directive is Directive.PLAN:
        lines.append("Call the `plan` tool with your complete script.")
    return "\n".join(lines)
