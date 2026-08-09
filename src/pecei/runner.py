"""Script runner: ONE LLM-authored script per cycle (gradient-free).

Each :func:`run_script` call is one cycle: reload the map fresh (reset-per-script),
ask the provider for ONE complete script, run it to a stop via the interpreter
(which drives the round engine), then return the script + stop-report + yielded
observations as :class:`Feedback` for the next cycle. There is **no per-round
``decide`` loop** — the author is blind while the script runs and only learns
after it stops.

Stop reasons: ``SUCCESS`` (goal reached) | ``COMPILE_ERROR`` (script didn't
compile — bad AST or type error; message fed back) | ``ROUND_LIMIT_EXCEED``
(budget) | ``ENERGY_RUN_OUT`` (reserved) | ``SCRIPT_ENDED`` (body fully executed,
budget remaining, goal not reached). ``world_at_step`` reconstructs ground truth
for the replay viewer.
"""
from __future__ import annotations

from dataclasses import dataclass

from pecei.action import BudgetExceeded, CompileError, Host, Interpreter, NavObs, pretty
from pecei.engine import RoundEngine
from pecei.infra import FailureSnapshot, Result, Trace, TraceEvent, build_report
from pecei.llm import Directive, Feedback, LLMProvider, TurnInput
from pecei.llm.prompt import SYSTEM_PROMPT, render_user
from pecei.observation import observe
from pecei.world import ActionType, World, apply_action, load_world


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
            yielded=list(self.yielded),
            failure_snapshot=self.failure_snapshot,
            compile_error=self.compile_error,
        )


def map_desc(world: World) -> dict:
    """Static puzzle description fed to the author every cycle (never live state)."""
    ego = world.ego
    return {
        "width": world.grid.width,
        "height": world.grid.height,
        "goal": list(world.goal) if world.goal else None,
        "ego": {"anchor": list(ego.anchor), "orientation": ego.orientation.value} if ego else None,
        "entities": [
            {"anchor": list(e.anchor), "types": [c.ctype.value for c in e.components.values()]}
            for e in world.entities.values()
        ],
    }


def run_script(
    map_path: str,
    provider: LLMProvider,
    *,
    feedback: Feedback | None = None,
    snowball: list[dict] | None = None,
    instructions: str | None = None,
    round_budget: int = 100,
    ego_eid: str | None = None,
) -> ScriptRun:
    """Run one cycle: fresh map -> ONE decide() -> run script to stop -> Feedback."""
    world = load_world(map_path)                   # reset-per-script (fresh map each cycle)
    ego = world.ego
    if ego is None:
        raise ValueError("world has no ego entity (is_ego=True)")
    ego_eid = ego_eid or ego.eid
    goal = world.goal

    eng = RoundEngine(world, round_budget=round_budget)
    trace = Trace()
    yielded: list[dict] = []

    def on_round(r: int, eid: str, action: ActionType, res) -> None:
        ent = world.entity(eid)
        trace.append(TraceEvent(
            round=r, actor=eid, action=action.value, moved=res.moved, blocked=res.blocked,
            anchor_after=ent.anchor, orientation_after=ent.orientation.value,
            observation=observe(world, eid).to_dict(),
        ))

    eng.on_round = on_round
    host = Host(
        act=lambda a: eng.apply(ego_eid, a).moved,
        observe=lambda: NavObs(observe(world, ego_eid), goal),
        yield_=lambda v: yielded.append(v.observation.to_dict()),
    )
    interp = Interpreter(host)

    turn = TurnInput(
        directive=Directive.PLAN,
        instructions=instructions,
        map_desc=map_desc(world),
        feedback=feedback,
        snowball=snowball or [],
    )
    # Record exactly what the author was shown this cycle. The system prompt may
    # drift across versions, so capturing it per-cycle lets replay reproduce the
    # author's true context regardless of later edits to prompt.py.
    prompt = {"system": SYSTEM_PROMPT, "user": render_user(turn)}
    out = provider.decide(turn)                    # ONE LLM request per cycle
    program = out.program
    compile_error: str | None = out.error          # A-layer: malformed AST (parse failed)

    stopped_by = "body"
    if program is not None and compile_error is None:
        try:
            interp.run(program)                    # runs type_check first (B-layer)
        except CompileError as e:
            compile_error = str(e)                 # B-layer: well-formed AST, type error
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
    elif stopped_by == "budget":
        result = Result.ROUND_LIMIT_EXCEED
    else:
        result = Result.SCRIPT_ENDED               # body fully executed, goal not reached

    report = build_report(
        world, result, goal=goal, yielded=yielded, round=eng.round, round_budget=round_budget
    )

    program_str = pretty(program) if program is not None else ""
    if trace.events:                              # record this cycle's script/IO on its first round
        trace.events[0].program = program_str or None
        trace.events[0].llm_request = out.raw_request
        trace.events[0].llm_response = out.raw_response

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

    Replays the ego's recorded actions from a fresh map load. Valid because MVP
    environment ticks are a no-op and only the ego moves.
    """
    world = load_world(map_path)
    for i in range(min(step, len(trace.events))):
        ev = trace.events[i]
        apply_action(world, ev.actor, ActionType(ev.action))
    return world
