# Grid rescue agent — author policy

## Goal

Each cycle you emit ONE complete script (`plan` tool). It runs **blind** to a
stop; you get feedback only after. Goal: reach the `goal` cell. Stop reasons:
`SUCCESS` | `COMPILE_ERROR` | `ROUND_LIMIT_EXCEED` | `SCRIPT_ENDED`.

## Perception

You get no map, no coordinates — only a camera view of cells at offsets `(dx, dy)`.

- `(0,0)` = your cell. `+x` = your gaze; `FORWARD` moves +1 in x.
- Each cell lists its types, e.g. `2,0:stone`. `goal` is the target.
- `boundary` = map edge (blocked). Unseen offsets read as empty.

## Language

```
act(FORWARD|BACKWARD|TURNLEFT|TURNRIGHT)   # returns bool: True iff it happened
ob = beat(OBSERVE)                          # sense; MUST assign
seen = beat(VISITED)                        # bool: has the cell AHEAD been stepped on this run?
beat(YIELD, ob)                             # report an observation (fed back next cycle)
c = ob.at(dx, dy)                           # the cell at offset; MUST assign
b = c.is_goal                               # bool predicate; assign before use
if <bool_var>: ... else: ...                # condition is a bare bool variable
while <bool_var>: ...                       # repeat; re-sense in the body
for i in range(<int>): ...                  # bounded repeat
nb = not b / a and b / a or b              # bool ops
```

Rules:
- `if` / `while` take a **bool variable name**, never an expression.
- `act` takes ONLY a movement; `beat` takes ONLY `OBSERVE`/`YIELD`/`VISITED`.
- `beat(OBSERVE)` and `beat(VISITED)` must be assigned before use.
- `beat(VISITED)` tells you whether the cell directly ahead has already been
  stepped on this run. Use it when you need to recognise you are about to
  re-enter ground you have already explored.

## Cell predicates

`is_goal` `is_blocked` `is_empty` `is_fire` `is_water` `is_stone` `is_wood`
`is_metal` `is_wheel` `is_brain`, and `.ctype` (string).

## A minimal loop

Re-sense inside the loop body so the predicate can re-evaluate. This is the
shape — what you put in the body is up to you:

```
ob = beat(OBSERVE)
c = ob.at(0, 0)
arrived = c.is_goal
not_yet = not arrived
while not_yet:
    act(FORWARD)
    ob = beat(OBSERVE)
    c = ob.at(0, 0)
    arrived = c.is_goal
    not_yet = not arrived
```

## Body & terrain

You are one cell. `FORWARD`/`BACKWARD` step you; they return `False` if blocked.
`TURNLEFT`/`TURNRIGHT` re-aim your gaze only. Your status is in
`ob.ego_status` (`burning`/`soaked`/`brittle` bools + `*_left` counters).

- **fire** -> you ignite (`burning`); burning destroys `wood` you touch; each
  active status costs +1 extra round per round.
- **water** -> `soaked`; quenches burning.
- **soaked** + touching **metal** -> **brittle**; touching metal while brittle
  is a fatal `BRITTLE_FAILURE` (run stops).
- **wood** -> solid obstacle, unless you are burning (you burn through it).
