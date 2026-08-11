"""Script runner: ONE LLM-authored script per cycle (gradient-free).

Each :func:`run_script` call is one cycle: reload the map fresh (reset-per-script),
ask the provider for ONE complete script, run it to a stop via the interpreter
(which drives the round engine), then return the script + stop-report + yielded
observations as :class:`Feedback` for the next cycle. There is **no per-round
``decide`` loop** — the author is blind while the script runs and only learns
after it stops.

A malformed script (``COMPILE_ERROR``) costs the cycle nothing to *execute*, so the
runner gives the author a few immediate, failure-aware retries (the compile error
is fed straight back) before recording the cycle as failed — this keeps cycles from
being wasted on pure AST-shape slips.

Stop reasons: ``SUCCESS`` (goal reached) | ``COMPILE_ERROR`` (script didn't
compile — bad AST or type error; message fed back) | ``ROUND_LIMIT_EXCEED``
(budget) | ``ENERGY_RUN_OUT`` (reserved) | ``SCRIPT_ENDED`` (body fully executed,
budget remaining, goal not reached) | ``BRITTLE_FAILURE`` (brittle ego touched a
metal cell). ``world_at_step`` reconstructs ground truth for the replay viewer.
"""
from __future__ import annotations

from dataclasses import dataclass

from pecei.action import (
    BudgetExceeded,
    CompileError,
    Host,
    Interpreter,
    pretty,
)
from pecei.engine import BrittleFailure, RoundEngine
from pecei.infra import FailureSnapshot, Result, Trace, TraceEvent, build_report
from pecei.llm import Directive, Feedback, LLMProvider, TurnInput
from pecei.llm.prompt import load_system_prompt, render_user
from pecei.observation import observe
from pecei.world import ActionType, World, apply_action, load_world

# How many extra attempts the author gets within one cycle when its script fails
# to compile. Each retry re-asks with the compile error fed back. Kept small so a
# cycle is still "one coherent attempt", not an unbounded repair loop.
COMPILE_RETRY_LIMIT = 3


@dataclass
class ScriptRun:
    program: str                              # pretty(program); "" if the author gave none
    stop_reason: Result
    rounds: int
    yielded: list[dict]
    failure_snapshot: FailureSnapshot | None
    compile_error: str | None                 # set iff stop_reason is COMPILE_ERROR
    trace: Trace
    prompt: dict | None                       # {system, user} shown to the author (drift-stable record)
    raw_request: dict | None
    raw_response: dict | None

    def to_feedback(self) -> Feedback:
        return Feedback(
            stop_reason=self.stop_reason,
            rounds_used=self.rounds,
            script=self.program,
            yielded=list(self.yielded),
            failure_snapshot=self.failure_snapshot,
            compile_error=self.compile_error,
        )


def seed_observation(world: World) -> dict:
    """The author's PARTIAL start-pose view: the 90° egocentric camera frame from
    the ego's start anchor (``observe(...).to_dict()``). NOT a god-view map, and
    NO absolute coordinates — no anchor, no orientation, no goal coordinate. The
    goal is only ever visible as a ``goal`` component when its cell enters the
    camera frame; the author must turn/look to find it.

    Given every cycle: each ``decide()`` is a stateless single LLM call, and the
    start pose is identical every cycle (map reloads fresh), so this is the
    author's *persistent* memory of its starting perception — equivalent to a
    one-time seed replayed each turn, not new information.
    """
    ego = world.ego
    return observe(world, ego.eid).to_dict()


def _ahead(world: World) -> tuple[int, int]:
    """The cell one step along the ego's gaze (where FORWARD would land)."""
    ego = world.ego
    ax, ay = ego.anchor
    dx, dy = ego.orientation.delta
    return (ax + dx, ay + dy)


def run_script(
    map_path: str,
    provider: LLMProvider,
    *,
    feedback: Feedback | None = None,
    snowball: list[dict] | None = None,
    instructions: str | None = None,
    round_budget: int = 100,
    ego_eid: str | None = None,
    compile_retries: int = COMPILE_RETRY_LIMIT,
    memory_context: str | None = None,
) -> ScriptRun:
    """Run one cycle: fresh map -> ONE decide() -> run script to stop -> Feedback.

    If the author's script fails to compile, retry up to ``compile_retries`` more
    times *within this same cycle*, feeding the compile error back each time, so a
    cycle is not wasted on a pure AST-shape slip. The cycle still ends in
    ``COMPILE_ERROR`` if every attempt fails.

    ``memory_context`` is the cross-map long-term memory (a shared
    ``MemoryEvolution``'s rendered context) handed in by the caller; it is shown
    to the author as authoritative learned bans.
    """
    attempt_feedback = feedback
    last: ScriptRun | None = None
    for attempt in range(1, compile_retries + 2):   # initial attempt + N retries
        run = _run_once(
            map_path, provider,
            feedback=attempt_feedback,
            snowball=snowball,
            instructions=instructions,
            round_budget=round_budget,
            ego_eid=ego_eid,
            memory_context=memory_context,
        )
        last = run
        if run.stop_reason is not Result.COMPILE_ERROR:
            return run
        if attempt <= compile_retries:
            # Feed the compile error straight back and let the author fix it.
            attempt_feedback = Feedback(
                stop_reason=Result.COMPILE_ERROR,
                rounds_used=0,
                script=run.program,
                compile_error=run.compile_error,
            )
    assert last is not None
    return last


def _run_once(
    map_path: str,
    provider: LLMProvider,
    *,
    feedback: Feedback | None,
    snowball: list[dict] | None,
    instructions: str | None,
    round_budget: int,
    ego_eid: str | None,
    memory_context: str | None,
) -> ScriptRun:
    """One map-fresh decide()+execute pass (a single attempt within a cycle)."""
    world = load_world(map_path)                   # reset-per-script (fresh map each attempt)
    ego = world.ego
    if ego is None:
        raise ValueError("world has no ego entity (is_ego=True)")
    ego_eid = ego_eid or ego.eid
    goal = world.goal

    eng = RoundEngine(world, round_budget=round_budget)
    trace = Trace()
    yielded: list[dict] = []
    # Ego-centric footprint memory for beat(VISITED): every cell the ego's anchor
    # has stepped onto *after moving* this run. The start cell is NOT pre-seeded,
    # so beat(VISITED) reads False on the first cell (you haven't "been here"
    # before) and True only once you return to a cell you already stepped on —
    # the signal that breaks dead-end / circling loops.
    visited: set[tuple[int, int]] = set()

    def on_round(r: int, eid: str, action: ActionType, res) -> None:
        ent = world.entity(eid)
        visited.add(ent.anchor)
        trace.append(TraceEvent(
            round=r, actor=eid, action=action.value, moved=res.moved, blocked=res.blocked,
            anchor_after=ent.anchor, orientation_after=ent.orientation.value,
            observation=observe(world, eid).to_dict(),
        ))

    eng.on_round = on_round
    host = Host(
        act=lambda a: eng.apply(ego_eid, a).moved,
        observe=lambda: observe(world, ego_eid),
        yield_=lambda v: yielded.append(v.to_dict()),
        # beat(VISITED): has the cell DIRECTLY AHEAD (one step along the gaze)
        # been stepped on this run? This is the "am I about to re-tread?" signal
        # a blind agent uses to avoid walking back onto explored ground. Checking
        # the cell ahead (not the current cell) means it stays meaningful while
        # turning in place: turn until the cell ahead is clear AND unvisited.
        visited=lambda: _ahead(world) in visited,
    )
    interp = Interpreter(host)

    turn = TurnInput(
        directive=Directive.PLAN,
        instructions=instructions,
        seed_observation=seed_observation(world),
        feedback=feedback,
        snowball=snowball or [],
        extra=memory_context,
    )
    # Record exactly what the author was shown this cycle. The system prompt may
    # drift across versions, so capturing it per-cycle lets replay reproduce the
    # author's true context regardless of later edits to prompt.py.
    prompt = {"system": load_system_prompt(), "user": render_user(turn)}
    out = provider.decide(turn)                    # ONE LLM request per attempt
    program = out.program
    compile_error: str | None = out.error          # A-layer: malformed AST (parse failed)

    stopped_by = "body"
    if program is not None and compile_error is None:
        try:
            interp.run(program)                    # runs type_check first (B-layer)
        except CompileError as e:
            compile_error = str(e)                 # B-layer: well-formed AST, type error
        except BrittleFailure:
            stopped_by = "brittle"      # brittle ego touched metal (fatal interaction)
        except RuntimeError:
            stopped_by = "budget"        # round budget exhausted mid-script (eng.apply)
        except BudgetExceeded:
            stopped_by = "budget"        # interpreter step budget

    # at_goal is authoritative for SUCCESS, regardless of how execution stopped
    if eng.at_goal(ego_eid):
        result = Result.SUCCESS
    elif compile_error is not None:
        result = Result.COMPILE_ERROR              # script didn't compile (A or B layer)
    elif program is None:
        result = Result.SCRIPT_ENDED               # author gave no script / empty body
    elif stopped_by == "brittle":
        result = Result.BRITTLE_FAILURE            # brittle ego touched a metal cell
    elif stopped_by == "budget":
        result = Result.ROUND_LIMIT_EXCEED
    else:
        result = Result.SCRIPT_ENDED               # body fully executed, goal not reached

    report = build_report(
        world, result, goal=goal, yielded=yielded, round=eng.round, round_budget=round_budget,
        trace=trace,
    )

    program_str = pretty(program) if program is not None else ""
    if trace.events:                              # record this cycle's script/IO on its first round
        trace.events[0].program = program_str or None
        trace.events[0].llm_request = out.raw_request
        trace.events[0].llm_response = out.raw_response
        # Persist what the script beat(YIELD)'d onto the trace so the next cycle's
        # Session.last_feedback() -> _trace_yielded() recovers them (the yielded
        # list is otherwise local-only and lost between cycles).
        trace.events[-1].yielded = list(yielded)

    return ScriptRun(
        program=program_str,
        stop_reason=result,
        rounds=eng.round,
        yielded=yielded,
        failure_snapshot=report.failure_snapshot,
        compile_error=compile_error,
        trace=trace,
        prompt=prompt,
        raw_request=out.raw_request,
        raw_response=out.raw_response,
    )


def world_at_step(map_path, trace: Trace, step: int) -> World:
    """Reconstruct the ground-truth World at replay ``step`` (0..len(trace)).

    Replays the ego's recorded actions from a fresh map load, applying the
    environment tick after each action so effects (burning/soaked/brittle,
    wood destruction) match the live run.
    """
    world = load_world(map_path)
    for i in range(min(step, len(trace.events))):
        ev = trace.events[i]
        apply_action(world, ev.actor, ActionType(ev.action))
        world.tick_environment()
    return world
