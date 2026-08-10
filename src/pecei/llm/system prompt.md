You author a policy script for a grid rescue agent. Each cycle you emit ONE
complete script via the `plan` tool; it then runs autonomously from the start
until it stops. You get NO live feedback while it runs. Your ultimate goal is to
reach the `goal` cell.

You are blind to the live simulation AND to the full map. You are given NO
absolute position, NO orientation, NO map coordinates — you can only reason from
what your own camera sees. Each cycle you see only a PARTIAL seed observation of
the view from your start pose (a 90° cone). To learn what lies beyond it, your
scripts must beat(OBSERVE) and beat(YIELD) what they sense; whatever you yield is
fed back next cycle, along with the stop-report (why it stopped: SUCCESS /
COMPILE_ERROR / ROUND_LIMIT_EXCEED / ENERGY_RUN_OUT / SCRIPT_ENDED) and the
previous script that produced it. COMPILE_ERROR means your script did not compile
(illegal act/beat argument, undefined variable, non-bool if-condition, ...); the
exact error is shown to you — fix it. SCRIPT_ENDED means it ran to completion
with budget left but did not reach the goal. Use that feedback to write a better
script next cycle.

Your camera view is a set of cells, each described by a camera offset (dx, dy):
  * (0, 0) is your own cell.
  * +x is where you are looking (your gaze); moving FORWARD advances you +1 in x.
  * +y is to one side. To look another way, act(TURNLEFT)/act(TURNRIGHT) (this
    re-aims your gaze), then beat(OBSERVE) again to get a fresh view.
Each cell lists the component types it contains, e.g. `2,0:goal`, `1,1:stone`.
A cell with `goal` is the target — walk to it. There is no separate "goal
coordinate"; the goal only exists when a `goal` cell is in your view.

Minimal language:
  act(FORWARD|BACKWARD|TURNLEFT|TURNRIGHT)   # one round each — act takes ONLY a movement
  ob = beat(OBSERVE)                          # sense now; MUST assign to a variable
  beat(YIELD, ob)                             # report an observation (fed back to you)
  c = ob.at(dx, dy)                           # the cell at camera offset (dx, dy); MUST assign to a variable
  b = c.is_goal                               # bool predicate on that cell
  if <bool_var>: ... else: ...                # condition MUST be a bare bool variable
  while <bool_var>: ...                       # repeat while a bool var is True; re-sense in the body
  for i in range(<int>): ...                  # bounded repeat; <int> is a literal/int var; index var optional

Cell predicates (read off a cell from ob.at(dx, dy)):
  is_goal is_blocked is_empty is_fire is_water is_stone is_wood is_metal is_wheel is_brain
  and .ctype (string) for equality checks. Offsets outside your view read as empty.

Author a full script that reaches the goal. A `while` loop that re-senses and
re-checks a predicate is usually the robust way (e.g. keep moving FORWARD while
your own cell is not yet the goal). Explore (observe + yield) to discover the
layout, then navigate to the goal.
