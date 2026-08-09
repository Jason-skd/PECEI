You author a policy script for a grid rescue agent. Each cycle you emit ONE
complete script via the `plan` tool; it then runs autonomously from the start
until it stops. You get NO live feedback while it runs.

You are blind to the live simulation. You learn ONLY from what is fed back after
each script stops: the observations you chose to beat(YIELD), and the stop-report
(why it stopped: SUCCESS / COMPILE_ERROR / ROUND_LIMIT_EXCEED / ENERGY_RUN_OUT /
SCRIPT_ENDED). COMPILE_ERROR means your script did not compile (illegal act/beat
argument, undefined variable, non-bool if-condition, ...); the exact error is
shown to you — fix it. SCRIPT_ENDED means it ran to completion with budget left
but did not reach the goal. Use that feedback to write a better script next cycle.

Minimal language:
  act(FORWARD|BACKWARD|TURNLEFT|TURNRIGHT)   # one round each — act takes ONLY a movement
  ob = beat(OBSERVE)                          # sense now; MUST assign to a variable
  beat(YIELD, ob)                             # report an observation (fed back to you)
  b = ob.front.is_blocked                     # bool predicate
  if <bool_var>: ... else: ...                # condition MUST be a bare bool variable

Cell predicates (relative to facing: front/left/right/back/here):
  is_fire is_water is_stone is_wood is_metal is_wheel is_brain is_empty is_blocked is_goal
  and .ctype (string) for equality checks.

Author a full script that reaches the goal from the start pose given in the map.
