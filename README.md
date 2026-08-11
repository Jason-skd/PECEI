# PECEI

> **Embodied Intelligence — Generate / Regenerate.**
> A grid-world harness where an LLM agent *authors* a policy script for a rigid body, runs it blind, then learns from the yielded observations + stop-report and regenerates — gradient-free.

PECEI is the working prototype for a 14-day Cambridge summer-school project (AI/CS). It implements the **inner loop** of a "differentiation → evolution → regeneration" pipeline described in [`docs/ROADMAP.md`](docs/ROADMAP.md): an LLM writes a complete control program for an embodied agent, executes it autonomously, and improves across cycles using only verbal feedback (no gradients). See the roadmap's §3 triple-loop architecture and §4 generate/regenerate.

---

## How it works

```
            ┌──────── one cycle = ONE LLM request ────────┐
            │                                             │
 author  ──▶│  blind script  →  run to a stop  →  feedback │──▶ next cycle
 (LLM)      │  (no live obs)     (interpreter)     (yields │     (snowball
            │                                     + report)│      grows)
            └─────────────────────────────────────────────┘
```

- **Blind author.** The LLM emits one complete script (`Program` AST via tool-use) and gets **no live observation** while it runs — unlike a per-round reactive agent.
- **Run to a stop.** The interpreter executes the whole script (`act` / `beat(OBSERVE)` / `beat(YIELD)` / `if <bool_var>`); it stops on goal / round-budget / body-end.
- **Learn from feedback.** After each stop the LLM receives the observations *it chose to yield* plus a stop-report (`SUCCESS | ROUND_LIMIT_EXCEED | ENERGY_RUN_OUT | SCRIPT_ENDED`), plus the snowball of all prior cycles. It regenerates from the initial map each time.
- **One keypress = one request.** The interactive `epoch` loop runs exactly one cycle per spacebar press and stops on the first `SUCCESS`.

The policy program is a JSON/dict **AST** (pydantic discriminated union), never free text — the LLM emits it against `Program.model_json_schema()` via non-strict tool-use (strict structured output rejects the recursive schema). The text DSL is only a `pretty()` renderer over the AST.

---

## Install

Requires Python ≥ 3.14.

```bash
git clone <repo> && cd PECEI
uv sync
uv run pytest -q          # should be green
```

### Provider keys (optional)

The agent defaults to `mock` (offline). To use a real LLM, copy `.env.example` → `.env` (gitignored) and fill it in; the CLI loads `.env` at startup and the SDKs read the vars directly:

```bash
cp .env.example .env
# ANTHROPIC_API_KEY=...      ANTHROPIC_BASE_URL=...   (relay/endpoint)
# OPENAI_API_KEY=...         OPENAI_BASE_URL=...
# DEEPSEEK_API_KEY=...
```

Precedence: `--api-key`/`--base-url` flag > `.env`/env > built-in default.

---

## Usage

```bash
# preview a map in a window (ESC/Q to quit) — eyeball that the world renders
uv run python -m pecei.render src/pecei/maps/01_corridor.yaml

# interactive: space advances one cycle, stops on SUCCESS, Ctrl+C saves & quits
uv run pecei epoch src/pecei/maps/01_corridor.yaml --provider mock

# auto-train one map until SUCCESS / cycle budget
uv run pecei auto src/pecei/maps/02_one_wall.yaml --provider mock --budget 20

# train a whole experiment (a folder of NN_slug maps), one session each, in order
uv run pecei experiment src/pecei/maps --provider mock

# replay a single epoch (one trace)
uv run pecei replay src/pecei/maps/04_maze.yaml sessions/04_maze.traces/04_maze.c001.trace.jsonl

# browse a session's epochs, pick one, replay it (or play-to-end)
uv run pecei replay src/pecei/maps/04_maze.yaml --session sessions/04_maze.session.json

# render authored scripts (AST->text): one trace / a session / a whole experiment
uv run pecei transcript sessions/04_maze.traces/04_maze.c001.trace.jsonl
uv run pecei transcript --session sessions/04_maze.session.json
uv run pecei transcript --experiment sessions            # one <slug>.transcript.txt per session
uv run pecei transcript --session sessions/04_maze.session.json --with-prompts --print
```

`auto` / `epoch` / `experiment` write per-cycle traces under `sessions/<map>.traces/` and, **by default, also dump a `sessions/<map>.transcript.txt`** (every cycle's script, one block each) when a session ends — pass `--no-transcript` to skip. The `transcript` command renders the same view on demand; it does no simulation, just surfaces the scripts already stored in the session/trace. Use `--provider anthropic` (or `openai` / `deepseek`) instead of `mock` once `.env` is set.

**Maps** (`src/pecei/maps/`, `NN_slug.yaml` = order + readable name):
- `01_corridor` — straight east; baseline solvability.
- `02_one_wall` — a stone column forces a detour (turning + branching).
- `03_water_obstacle` — stone wall with a *passable* water gap (fire/water don't kill in the MVP; difficulty is geometric).
- `04_maze` — two staggered walls; blind exploration.

All four are BFS-proven solvable, and naive "drive straight" fails on 02–04, so the author can't read the answer off the start.

### Replay controls

`←`/`→` (or `n`/`p`) step · `Home`/`End` jump · `PgUp`/`PgDn` switch epoch · `SPACE` play/pause · `c` (or the **play to end** button) play from the current epoch through to the end · drag the timeline bar · `q`/`ESC` quit. Tiles are sprites (`render/assets/*.png`); the renderer falls back to color+shape blocks if they're missing.

---

## Architecture

Strict one-way layering (the world never knows about layers above it):

```
world/  ──▶  observation/  ──▶  action/ (AST + interpreter)  ──▶  engine/
   │                                                              ▲
   └──────────────────────  infra/ (trace, report, complexity)  ──┘
                                      ▲
   llm/ (protocol + adapters) ────────┤   render/ (headless renderer + assets)
                                      │
            runner/ ──▶  session/ ──▶  cli/ ──▶  replay/   experiment/   transcript/
```

- **world** — ground-truth grid + rigid multi-cell `Entity` (components: stone/wood/fire/water/wheel/brain/metal), YAML/JSON map parser.
- **observation** — conical FOV, leak-free copies (never live `World` refs).
- **action** — policy-script AST + a small interpreter (`interp.run` walks the whole program); `If.test` is a bare bool *variable name* (schema forbids expressions), enforced by `type_check`.
- **engine** — one action = one round + env tick; `at_goal` is the authoritative success check; budget-exhaustion raises.
- **infra** — JSONL `Trace` (per-round truth + `program`/`llm_request`/`llm_response` + `yielded`), `MatchReport`/`Result`/`FailureSnapshot`, entity-aware Dijkstra `complexity`.
- **llm** — `TurnInput{instructions, map_desc, feedback, snowball}` (no live observation), provider-agnostic adapters (`anthropic`/`openai`/`deepseek`/`mock`) behind a `Protocol`. Structured output via tool-use.
- **runner** — `run_script`: one `decide()` per cycle, reloads the map (reset-per-script), runs to a stop, builds the stop-report. `world_at_step` reconstructs truth for replay.
- **session** — snowballs `CycleRecord`s; the trace is the single source of truth (the session stores a summary + `trace_path`, yields are derived, not duplicated).
- **experiment** — scans a `NN_slug` dir, trains each map as a session in order, injects "session k of N" via `instructions`.
- **replay** — epoch-based viewer with image tiles, session browser, play-to-end.
- **transcript** — renders a session's authored scripts (AST→text) at epoch / session / experiment granularity; no simulation, just surfaces the scripts stored on each `CycleRecord`.

### Key invariants
- **No live observation to the author.** The LLM learns only from fed-back yields + report.
- **One LLM request per cycle.** No per-round reactive `decide` loop.
- **Reset-per-script.** Every cycle starts from the initial map.
- **Same-source record.** The per-cycle trace is authoritative; the session references it.
- **Drift-stable prompt record.** Each `CycleRecord.prompt` stores the exact `{system, user}` text shown to the author that cycle, so replay reproduces the author's true context even after `SYSTEM_PROMPT` later changes.

---

## Status

The inner loop (generate → run → feedback → regenerate) is complete and tested (`uv run pytest -q`): mock closed-loop reaches the goal, traces are replayable, the epoch loop stops on success, `auto`/`experiment` orchestrate across maps. Roadmap's **middle loop** (Reflexion + skill atoms + shared memory + principle distillation) and **outer loop** (body regeneration) are scaffolded only — `Directive` carries `REFLECT`/`COMPRESS`/`STORE` placeholders for them — not yet implemented.

## License

MIT (project default). See `docs/ROADMAP.md` for the full design and scope (MVP/Silver/Gold).
