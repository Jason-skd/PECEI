# Grid rescue agent — author policy

## 1. Goal & loop

Each cycle you emit ONE complete script (the `plan` tool). It runs **blind** from
the start until it stops — you get NO feedback while it runs. Your goal: reach the
`goal` cell. After the script stops you learn why (`SUCCESS` / `COMPILE_ERROR` /
`ROUND_LIMIT_EXCEED` / `SCRIPT_ENDED`) plus what your script yielded; use that to
write a better script next cycle.

- `COMPILE_ERROR`: your script was malformed; the exact error is shown — fix it.
- `SCRIPT_ENDED`: ran out of statements with budget left, goal not reached.
- `ROUND_LIMIT_EXCEED`: a loop never terminated (100 rounds) — your "keep going"
  condition never became false; fix the stopping predicate.

## 2. What you can perceive

You get **no map, no coordinates, no compass** — only your own camera view: a set
of cells at offsets `(dx, dy)`.

- `(0, 0)` = your own cell. `+x` = where you look (your gaze); `FORWARD` moves
  you +1 in `x`. `+y` = one side.
- To look another way: `act(TURNLEFT)` / `act(TURNRIGHT)` re-aims your gaze, then
  `beat(OBSERVE)` for a fresh view.
- Each cell lists its component types, e.g. `2,0:goal`, `1,1:stone`. A `goal`
  cell is the target — walk to it. You only see the goal when its cell is in view.

## 3. Language

### 3.1 Statements
```
act(FORWARD|BACKWARD|TURNLEFT|TURNRIGHT)   # one round; act takes ONLY a movement
ob = beat(OBSERVE)                          # sense now; MUST assign to a variable
beat(YIELD, ob)                             # report an observation (fed back next cycle)
c = ob.at(dx, dy)                           # the cell at offset (dx,dy); MUST assign
b = c.is_goal                               # a bool predicate; assign before use
nb = not b                                  # bool ops: and / or / not
if <bool_var>: ... else: ...                # condition is a bare bool VARIABLE
while <bool_var>: ...                       # repeat while a bool var is True; re-sense in the body
for i in range(<int>): ...                  # bounded repeat; <int> literal or int var
```

### 3.2 Rules
- `if` / `while` take a **bool variable name** (string), never an expression —
  compute any predicate into a bool variable first.
- `beat(OBSERVE)` must be assigned before its cell's attributes are used.
- `act` takes ONLY a movement; `beat` takes ONLY `OBSERVE`/`YIELD`. `act(YIELD)`
  and `beat(FORWARD)` are always wrong.

### 3.3 Cell predicates (on a cell from `ob.at(dx, dy)`)
`is_goal` `is_blocked` `is_empty` `is_fire` `is_water` `is_stone` `is_wood`
`is_metal` `is_wheel` `is_brain`, and `.ctype` (string).

- A cell of type `boundary` is the **map edge** — it is blocked, you cannot walk
  past it. Sense it and turn away (treat it like a wall).
- Offsets outside your view (behind you, beyond range) read as empty.

## 4. Worked example — walk to the goal

The robust idiom is a `while` loop that re-senses each step. Note the condition
is **negated**: keep going while you are NOT there yet.

```
ob = beat(OBSERVE)
c = ob.at(0, 0)          # my own cell
arrived = c.is_goal
not_yet = not arrived
while not_yet:
    act(FORWARD)
    ob = beat(OBSERVE)
    c = ob.at(0, 0)
    arrived = c.is_goal
    not_yet = not arrived
beat(YIELD, ob)
```

To detour around an obstacle, sense the cell ahead and turn when blocked, e.g.
`ahead = ob.at(1, 0); if ahead.is_blocked: act(TURNRIGHT) else: act(FORWARD)` —
but remember a `while` loop must always be able to terminate (its predicate must
eventually flip), or it hits the round limit.

## 5. Terrain effects & your body state

Terrain cells have material effects on you (your body is wheel/metal/brain):

- **fire** (`is_fire`): standing on fire ignites you -> `burning`. While burning
  you destroy any `wood` you touch (wood stops blocking), but each active status
  costs **+1 extra round** per round.
- **water** (`is_water`): standing on water makes you `soaked` and quenches
  burning immediately.
- **metal** (`is_metal`): harmless normally. BUT if you are **soaked**, your
  metal part becomes **brittle** — touching a metal cell while brittle is a
  **fatal failure** (`BRITTLE_FAILURE`): the run stops. Never walk into metal
  while wet.
- **wood** (`is_wood`): a solid obstacle — unless you are burning, in which case
  you burn through it.

Your own status is in `ob.ego_status` (dict): `burning`, `soaked`, `brittle`
bools with remaining-round counters (`burning_left`, `soaked_left`, `brittle_left`).
Statuses wear off after a few rounds. `ego_status` is present in every
observation you take, so check it before entering risky terrain.

