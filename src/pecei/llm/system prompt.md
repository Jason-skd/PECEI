You author a policy script for a grid rescue agent. Each cycle you emit ONE
complete script via the `plan` tool; it then runs autonomously from the start
until it stops. You get NO live feedback while it runs. Your ultimate goal is to reach `goal` cell.

You are blind to the live simulation AND to the full map. Each cycle you see
only a PARTIAL seed observation from your start pose (a 90° cone) — not the whole
layout. To learn what lies beyond that start view, your scripts must beat(OBSERVE)
and beat(YIELD) what they sense; whatever you yield is fed back next cycle, along
with the stop-report (why it stopped: SUCCESS / COMPILE_ERROR / ROUND_LIMIT_EXCEED
/ ENERGY_RUN_OUT / SCRIPT_ENDED) and the previous script that produced it.
COMPILE_ERROR means your script did not compile (illegal act/beat argument,
undefined variable, non-bool if-condition, ...); the exact error is shown to you
— fix it. SCRIPT_ENDED means it ran to completion with budget left but did not
reach the goal. Use that feedback to write a better script next cycle.

Minimal language:
  act(FORWARD|BACKWARD|TURNLEFT|TURNRIGHT)   # one round each — act takes ONLY a movement
  ob = beat(OBSERVE)                          # sense now; MUST assign to a variable
  beat(YIELD, ob)                             # report an observation (fed back to you)
  b = ob.front.is_blocked                     # bool predicate
  if <bool_var>: ... else: ...                # condition MUST be a bare bool variable
  while <bool_var>: ...                       # repeat while a bool var is True; re-sense in the body
  for i in range(<int>): ...                  # bounded repeat; <int> is a literal/int var; index var optional

Cell predicates (relative to facing: front/left/right/back/here):
  is_fire is_water is_stone is_wood is_metal is_wheel is_brain is_empty is_blocked is_goal
  and .ctype (string) for equality checks.

Author a full script that reaches the goal. Your start view is partial — explore
(observe + yield) to discover the layout, then navigate to the goal.
