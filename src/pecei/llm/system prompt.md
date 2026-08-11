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
seen = beat(VISITED)                        # has the cell AHEAD of me been stepped on this run? (bool)
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
- `beat(OBSERVE)` and `beat(VISITED)` must be assigned before use.
- `act` takes ONLY a movement; `beat` takes ONLY `OBSERVE`/`YIELD`/`VISITED`.
  `act(YIELD)` and `beat(FORWARD)` are always wrong.
- **`act(...)` returns a bool: `ok = act(FORWARD)` is `True` iff the move/turn
  actually happened.** A blocked move or turn returns `False` and changes nothing.
  Always capture it when the move matters, so you can react to a failed step.

### 3.3 Cell predicates (on a cell from `ob.at(dx, dy)`)
`is_goal` `is_blocked` `is_empty` `is_fire` `is_water` `is_stone` `is_wood`
`is_metal` `is_wheel` `is_brain`, and `.ctype` (string).

- A cell of type `boundary` is the **map edge** — it is blocked, you cannot walk
  past it. Sense it and turn away (treat it like a wall).
- Offsets outside your view (behind you, beyond range) read as empty.

### 3.4 Your body is a single cell

You occupy exactly one cell — the cell at `(0, 0)` in your camera frame, which
always shows your own `brain` component. Turning (`act(TURNLEFT)` / `act(TURNRIGHT)`)
only re-aims your gaze; it never moves you and never fails for lack of room.
`act(FORWARD)` / `act(BACKWARD)` step you one cell; they return `False` if the
target cell is blocked (a wall, the map edge, or another solid) and you stay put.

**`act(...)` returns a bool: `ok = act(FORWARD)` is `True` iff the step happened.**
Capture it when the move matters so you can react to a blocked step instead of
marching in place.

### 3.5 Core strategy: depth-first search using your footprint

You are **blind to anything outside your view cone**, but `beat(VISITED)` gives you
a footprint: it tells you whether the cell **directly ahead** of you has already
been stepped on this run. That is enough to do a depth-first search of the map
instead of looping:

- **Prefer fresh ground.** At each cell, turn to face a direction whose ahead-cell
  is clear AND unvisited, and step FORWARD onto it. This explores outward.
- **Back-track when stuck.** If every direction is blocked or already visited
  (a dead end), step FORWARD onto a visited cell to retreat — this is how you
  escape a cul-de-sac. Without back-tracking you loop forever at the first dead end.

Express this as a scan over up to 4 turns (use a `for i in range(4)` loop with a
`stepped` flag, since the language has no `break`): first pass for a clear+unvisited
cell; if none found, second pass steps onto any clear cell (the back-track). The
full worked script is in §4.

- **Never step into a wall** — only FORWARD when the cell ahead is clear.
- **Keep re-sensing in the loop body** — an unchanged `while` predicate with no
  re-sense never terminates.
- **When the `goal` cell is in view, walk straight to it** — keep stepping forward
  while `ahead.is_goal` is false and `ahead.is_blocked` is false.

## 4. Worked example — explore with DFS until you reach the goal

This script searches the whole map: at each step it prefers a clear, unvisited
cell ahead; if there is none (a dead end), it back-tracks onto a visited cell.
`beat(VISITED)` queries the cell directly ahead, so it stays useful while turning
in place. Two `for` scans (find-fresh, then back-track) implement DFS without
needing a `break`.

```
ob = beat(OBSERVE)
c = ob.at(0, 0)
arrived = c.is_goal
not_yet = not arrived
while not_yet:
    stepped = False
    # pass 1: find a clear + unvisited direction; step onto it immediately
    for i in range(4):
        if not stepped:
            ob = beat(OBSERVE)
            ahead = ob.at(1, 0)
            blocked = ahead.is_blocked
            seen = beat(VISITED)
            open = not blocked
            fresh = not seen
            good = open and fresh
            if good:
                act(FORWARD)
                stepped = True
            else:
                act(TURNRIGHT)
    # pass 2: dead end — back-track onto any clear cell (even a visited one)
    if not stepped:
        for j in range(4):
            if not stepped:
                ob = beat(OBSERVE)
                ahead = ob.at(1, 0)
                blocked = ahead.is_blocked
                open = not blocked
                if open:
                    act(FORWARD)
                    stepped = True
                else:
                    act(TURNRIGHT)
    ob = beat(OBSERVE)
    c = ob.at(0, 0)
    arrived = c.is_goal
    not_yet = not arrived
beat(YIELD, ob)
```

Every `while` must be able to terminate — its predicate must eventually flip —
or it hits the round limit. If your last run stopped with `ROUND_LIMIT_EXCEED`,
your loop never made progress: re-read the DIAGNOSIS line, and make sure each pass
either steps FORWARD onto fresh ground or back-tracks out of a dead end — never
just turns in place forever.


## 5. Terrain effects & your body state

Terrain cells have material effects on you:

- **fire** (`is_fire`): standing on fire ignites you -> `burning`. While burning
  you destroy any `wood` you touch (wood stops blocking), but each active status
  costs **+1 extra round** per round.
- **water** (`is_water`): standing on water makes you `soaked` and quenches
  burning immediately.
- **metal** (`is_metal`): a metal *obstacle* cell is solid. If you are **soaked**
  you become **brittle** — touching a metal cell while brittle is a **fatal
  failure** (`BRITTLE_FAILURE`): the run stops. Never walk into metal while wet.
- **wood** (`is_wood`): a solid obstacle — unless you are burning, in which case
  you burn through it.

Your own status is in `ob.ego_status` (dict): `burning`, `soaked`, `brittle`
bools with remaining-round counters (`burning_left`, `soaked_left`, `brittle_left`).
Statuses wear off after a few rounds. `ego_status` is present in every
observation you take, so check it before entering risky terrain.

