"""CLI entry: ``pecei run | epoch | auto | experiment | replay``.

One cycle = one LLM request (the author writes a full script, it runs to a stop,
then feedback is returned). There is no per-round reactive loop.

- ``run``        : one script cycle on a map, print its stop-report.
- ``epoch``      : interactive — space advances one cycle; session stops on SUCCESS.
- ``auto``       : train one map automatically until SUCCESS / cycle budget.
- ``experiment`` : train a folder of ``NN_slug`` maps, one session each, in order.
- ``replay``     : replay a trace (one epoch) or a session (pick an epoch / play-to-end).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pecei.env import load_env
from pecei.llm import make_provider


def main(argv: list[str] | None = None) -> int:
    load_env()  # populate os.environ from .env (api_key / base_url) if present
    parser = argparse.ArgumentParser(prog="pecei", description="PECEI embodied-AI harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run ONE script cycle and print its stop-report")
    p_run.add_argument("map", help="path to a .yaml/.json map")
    p_run.add_argument("--provider", default="mock", help="mock | anthropic | openai | deepseek")
    p_run.add_argument("--round-budget", type=int, default=100)
    p_run.add_argument("--api-key", default=None, help="provider API key (else env)")
    p_run.add_argument("--base-url", default=None, help="provider base URL / relay endpoint")

    p_ep = sub.add_parser(
        "epoch", help="interactive: space advances one cycle, stops on SUCCESS, Ctrl+C saves & quits")
    p_ep.add_argument("map", help="path to a .yaml/.json map")
    p_ep.add_argument("--session", default=None,
                      help="session JSON path (default sessions/<map>.session.json; resumes if it exists)")
    p_ep.add_argument("--provider", default=None, help="mock | anthropic | openai | deepseek (else resume/default)")
    p_ep.add_argument("--round-budget", type=int, default=None)
    p_ep.add_argument("--api-key", default=None, help="provider API key (else env)")
    p_ep.add_argument("--base-url", default=None, help="provider base URL / relay endpoint")
    p_ep.add_argument("--no-trace", action="store_true", help="don't write per-cycle trace files")

    p_auto = sub.add_parser("auto", help="train one map automatically until SUCCESS / cycle budget")
    p_auto.add_argument("map", help="path to a .yaml/.json map")
    p_auto.add_argument("--budget", type=int, default=10, help="max script cycles")
    p_auto.add_argument("--round-budget", type=int, default=100, help="max rounds per script")
    p_auto.add_argument("--session", default=None, help="session JSON path (default sessions/<map>.session.json)")
    p_auto.add_argument("--provider", default="mock", help="mock | anthropic | openai | deepseek")
    p_auto.add_argument("--api-key", default=None, help="provider API key (else env)")
    p_auto.add_argument("--base-url", default=None, help="provider base URL / relay endpoint")
    p_auto.add_argument("--no-trace", action="store_true", help="don't write per-cycle trace files")

    p_exp = sub.add_parser("experiment", help="train a folder of NN_slug maps, one session each")
    p_exp.add_argument("dir", help="directory of NN_slug.{yaml,yml,json} maps")
    p_exp.add_argument("--budget", type=int, default=10, help="max script cycles per session")
    p_exp.add_argument("--round-budget", type=int, default=100, help="max rounds per script")
    p_exp.add_argument("--out", default="sessions", help="output dir for sessions+traces")
    p_exp.add_argument("--provider", default="mock", help="mock | anthropic | openai | deepseek")
    p_exp.add_argument("--api-key", default=None, help="provider API key (else env)")
    p_exp.add_argument("--base-url", default=None, help="provider base URL / relay endpoint")

    p_rep = sub.add_parser("replay", help="replay a trace (one epoch) or a session (pick an epoch)")
    p_rep.add_argument("map", help="path to the map that produced the trace(s)")
    p_rep.add_argument("trace", nargs="?", default=None, help="path to a trace JSONL (single epoch)")
    p_rep.add_argument("--session", default=None, help="path to a session JSON (browse epochs)")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "epoch":
        return _epoch(args)
    if args.cmd == "auto":
        return _auto(args)
    if args.cmd == "experiment":
        return _experiment(args)
    if args.cmd == "replay":
        return _replay(args)
    return 2


def _provider_kwargs(args, provider_name: str) -> dict:
    kwargs: dict = {}
    if provider_name != "mock":
        if getattr(args, "api_key", None):
            kwargs["api_key"] = args.api_key
        if getattr(args, "base_url", None):
            kwargs["base_url"] = args.base_url
    return kwargs


def _run(args) -> int:
    from pecei.runner import run_script

    provider = make_provider(args.provider, **_provider_kwargs(args, args.provider))
    run = run_script(args.map, provider, round_budget=args.round_budget)

    print(f"stop_reason: {run.stop_reason.value}")
    print(f"rounds: {run.rounds}  yielded: {len(run.yielded)}")
    if run.failure_snapshot:
        print(f"ended near {tuple(run.failure_snapshot.pos)}, "
              f"complexity {run.failure_snapshot.complexity}")
    print(f"script:\n{run.program}")

    trace_path = str(Path(args.map).with_suffix(".trace.jsonl"))
    run.trace.write(trace_path)
    print(f"\ntrace -> {trace_path}")
    print(f"replay: uv run python -m pecei replay {args.map} {trace_path}")
    return 0


def _epoch(args) -> int:
    from pecei.session import Session, interactive_loop

    map_path = args.map
    session_path = args.session or f"sessions/{Path(map_path).stem}.session.json"

    if Path(session_path).exists():
        session = Session.load(session_path)
        print(f"loaded session {session_path} ({len(session.cycles)} cycle(s))")
    else:
        session = Session(
            map=map_path,
            provider=args.provider or "mock",
            base_url=args.base_url,
            round_budget=args.round_budget if args.round_budget is not None else 100,
        )
    existed = Path(session_path).exists()

    if args.provider:
        session.provider = args.provider
    if args.base_url is not None:
        session.base_url = args.base_url
    if args.round_budget is not None:
        session.round_budget = args.round_budget

    provider = make_provider(session.provider, **_provider_kwargs(args, session.provider))
    trace_dir = None if args.no_trace else f"sessions/{Path(map_path).stem}.traces"
    interactive_loop(session, provider, session_path, trace_dir=trace_dir, existed=existed)
    return 0


def _auto(args) -> int:
    from pecei.session import Session, auto_session

    map_path = args.map
    session_path = args.session or f"sessions/{Path(map_path).stem}.session.json"
    session = Session(
        map=map_path, provider=args.provider, base_url=args.base_url,
        round_budget=args.round_budget,
    )
    provider = make_provider(args.provider, **_provider_kwargs(args, args.provider))
    trace_dir = None if args.no_trace else f"sessions/{Path(map_path).stem}.traces"
    auto_session(session, provider, session_path, budget=args.budget, trace_dir=trace_dir)
    return 0


def _experiment(args) -> int:
    from pecei.experiment import run_experiment

    provider = make_provider(args.provider, **_provider_kwargs(args, args.provider))
    run_experiment(
        args.dir, provider,
        out_dir=args.out, budget=args.budget, round_budget=args.round_budget,
    )
    return 0


def _replay(args) -> int:
    from pecei.replay import replay_session, replay_viewer

    if args.session:
        replay_session(args.map, args.session)
        return 0
    if not args.trace:
        raise SystemExit("replay needs a trace path, or --session <session.json>")
    replay_viewer(args.map, args.trace)
    return 0
