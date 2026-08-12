"""CLI entry: ``pecei epoch | auto | experiment | replay | transcript``.

One cycle = one LLM request (the author writes a full script, it runs to a stop,
then feedback is returned). There is no per-round reactive loop.

- ``epoch``      : interactive — space advances one cycle; session stops on SUCCESS.
- ``auto``       : train one map automatically until SUCCESS / cycle budget.
- ``experiment`` : train a folder of ``NN_slug`` maps, one session each, in order.
- ``replay``     : replay a trace (one epoch) or a session (pick an epoch / play-to-end).
- ``transcript`` : render authored scripts (AST→text) from a trace / session / experiment.
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
    p_ep.add_argument("--no-transcript", action="store_true", help="don't auto-write <stem>.transcript.txt on session end")

    p_auto = sub.add_parser("auto", help="train one map automatically until SUCCESS / cycle budget")
    p_auto.add_argument("map", help="path to a .yaml/.json map")
    p_auto.add_argument("--budget", type=int, default=10, help="max script cycles")
    p_auto.add_argument("--round-budget", type=int, default=100, help="max rounds per script")
    p_auto.add_argument("--session", default=None, help="session JSON path (default sessions/<map>.session.json)")
    p_auto.add_argument("--provider", default="mock", help="mock | anthropic | openai | deepseek")
    p_auto.add_argument("--api-key", default=None, help="provider API key (else env)")
    p_auto.add_argument("--base-url", default=None, help="provider base URL / relay endpoint")
    p_auto.add_argument("--no-trace", action="store_true", help="don't write per-cycle trace files")
    p_auto.add_argument("--no-transcript", action="store_true", help="don't auto-write <stem>.transcript.txt on session end")

    p_exp = sub.add_parser("experiment", help="train a folder of NN_slug maps, one session each")
    p_exp.add_argument("dir", help="directory of NN_slug.{yaml,yml,json} maps")
    p_exp.add_argument("--budget", type=int, default=10, help="max script cycles per session")
    p_exp.add_argument("--round-budget", type=int, default=100, help="max rounds per script")
    p_exp.add_argument("--out", default="sessions", help="output dir for sessions+traces")
    p_exp.add_argument("--provider", default="mock", help="mock | anthropic | openai | deepseek")
    p_exp.add_argument("--api-key", default=None, help="provider API key (else env)")
    p_exp.add_argument("--base-url", default=None, help="provider base URL / relay endpoint")
    p_exp.add_argument("--no-transcript", action="store_true", help="don't auto-write <slug>.transcript.txt per session")

    p_cmp = sub.add_parser(
        "compare", help="warm-vs-cold comparison: train on train_dir, then test both arms on test_dir")
    p_cmp.add_argument("train_dir", help="directory of NN_slug maps to train on (warm-start phase)")
    p_cmp.add_argument("test_dir", help="directory of NN_slug maps to evaluate BOTH arms on")
    p_cmp.add_argument("--budget", type=int, default=10, help="max script cycles per session")
    p_cmp.add_argument("--round-budget", type=int, default=100, help="max rounds per script")
    p_cmp.add_argument("--out", default="compare", help="output dir for comparison results")
    p_cmp.add_argument("--provider", default="mock", help="mock | anthropic | openai | deepseek")
    p_cmp.add_argument("--api-key", default=None, help="provider API key (else env)")
    p_cmp.add_argument("--base-url", default=None, help="provider base URL / relay endpoint")
    p_cmp.add_argument("--model", default=None, help="author model override (e.g. claude-haiku-4-5)")
    p_cmp.add_argument("--memory-model", default=None,
                       help="model for memory compression (defaults to --model / provider default)")
    p_cmp.add_argument("--no-llm-memory", action="store_true",
                       help="use deterministic (rule-based) memory compression instead of an LLM")
    p_cmp.add_argument("--plot", action="store_true", help="also render a grouped bar chart PNG")
    p_cmp.add_argument("--no-transcript", action="store_true", help="don't auto-write transcripts per session")
    p_cmp.add_argument("--resume", action="store_true",
                       help="skip sessions already at budget/SUCCESS (replay their memory); "
                            "restart a long compare after a crash without redoing finished maps")

    p_rep = sub.add_parser("replay", help="replay a trace (one epoch) or a session (pick an epoch)")
    p_rep.add_argument("map", help="path to the map that produced the trace(s)")
    p_rep.add_argument("trace", nargs="?", default=None, help="path to a trace JSONL (single epoch)")
    p_rep.add_argument("--session", default=None, help="path to a session JSON (browse epochs)")

    p_tr = sub.add_parser(
        "transcript", help="render authored scripts (AST->text) from a trace / session / experiment")
    p_tr.add_argument("trace", nargs="?", default=None, help="path to a trace JSONL (epoch granularity)")
    p_tr.add_argument("--session", default=None, help="path to a session JSON (session granularity)")
    p_tr.add_argument("--experiment", default=None, help="dir of *.session.json (experiment granularity)")
    p_tr.add_argument("--out", default=None, help="output file (session; default <stem>.transcript.txt)")
    p_tr.add_argument("--with-prompts", action="store_true", help="include each cycle's recorded prompt")
    p_tr.add_argument("--print", action="store_true", help="print to stdout instead of writing a file")

    args = parser.parse_args(argv)
    if args.cmd == "epoch":
        return _epoch(args)
    if args.cmd == "auto":
        return _auto(args)
    if args.cmd == "experiment":
        return _experiment(args)
    if args.cmd == "compare":
        return _compare(args)
    if args.cmd == "replay":
        return _replay(args)
    if args.cmd == "transcript":
        return _transcript(args)
    return 2


def _provider_kwargs(args, provider_name: str) -> dict:
    kwargs: dict = {}
    if provider_name != "mock":
        if getattr(args, "api_key", None):
            kwargs["api_key"] = args.api_key
        if getattr(args, "base_url", None):
            kwargs["base_url"] = args.base_url
    return kwargs


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
    interactive_loop(session, provider, session_path, trace_dir=trace_dir, existed=existed,
                     dump_transcript=not args.no_transcript)
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
    auto_session(session, provider, session_path, budget=args.budget, trace_dir=trace_dir,
                 dump_transcript=not args.no_transcript)
    return 0


def _experiment(args) -> int:
    from pecei.experiment import run_experiment

    provider = make_provider(args.provider, **_provider_kwargs(args, args.provider))
    run_experiment(
        args.dir, provider,
        out_dir=args.out, budget=args.budget, round_budget=args.round_budget,
        dump_transcript=not args.no_transcript,
    )
    return 0


def _compare(args) -> int:
    from pecei.compare import run_compare

    provider = make_provider(args.provider, **_provider_kwargs(args, args.provider))
    if args.model and hasattr(provider, "model"):
        provider.model = args.model

    memory_llm = None
    if not args.no_llm_memory and args.provider != "mock":
        memory_llm = _memory_llm_callable(args)

    result = run_compare(
        args.train_dir, args.test_dir, provider,
        budget=args.budget, round_budget=args.round_budget,
        out_dir=args.out, memory_llm=memory_llm,
        dump_transcript=not args.no_transcript,
        resume=args.resume,
    )
    _print_comparison_summary(result)

    if args.plot:
        from pecei.plotting import plot_comparison

        out = Path(args.out)
        plot_comparison(result, out / "comparison_epochs.png", metric="epochs")
        plot_comparison(result, out / "comparison_rounds.png", metric="rounds")
        print(f"charts -> {out / 'comparison_epochs.png'} (+ _rounds.png)")
    return 0


def _memory_llm_callable(args):
    """Build a ``(str)->str`` callable for MemoryEvolution's compression stage.

    Uses the SAME provider credentials as the author, so the warm arm's learned
    bans are distilled by a real LLM rather than the weak deterministic
    fallback (which Figure 1 needs to show a warm < cold gap). The callable is a
    plain prompt->string call — distinct from the author's structured-output
    tool-use. Returns None (caller falls back to deterministic) if the provider
    has no plain-completion path.
    """
    kwargs = _provider_kwargs(args, args.provider)
    name = args.provider

    if name in ("anthropic", "claude"):
        import anthropic

        client = anthropic.Anthropic(**kwargs)
        model = args.memory_model or args.model or "claude-haiku-4-5"

        def _call(prompt: str) -> str:
            resp = client.messages.create(
                model=model, max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    return block.text
            return ""
        return _call

    if name == "openai":
        import openai

        client = openai.OpenAI(**kwargs)
        model = args.memory_model or args.model or "gpt-4o-mini"

        def _call(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model, max_tokens=512, messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        return _call

    # mock / deepseek / unknown: no plain-completion adapter wired -> deterministic fallback
    return None


def _print_comparison_summary(result) -> None:
    print("\n=== Warm vs Cold Comparison ===")
    print(f"{'map':22s} {'warm (epochs/rounds)':24s} {'cold (epochs/rounds)':24s}")
    for w, c in zip(result.warm, result.cold):
        ws = f"{w.epochs_to_success}/{w.total_rounds}" + ("" if w.solved else " (unsolved)")
        cs = f"{c.epochs_to_success}/{c.total_rounds}" + ("" if c.solved else " (unsolved)")
        print(f"{w.slug:22s} {ws:24s} {cs:24s}")


def _replay(args) -> int:
    from pecei.replay import replay_session, replay_viewer

    if args.session:
        replay_session(args.map, args.session)
        return 0
    if not args.trace:
        raise SystemExit("replay needs a trace path, or --session <session.json>")
    replay_viewer(args.map, args.trace)
    return 0


def _transcript(args) -> int:
    from pecei import transcript
    from pecei.session import Session

    with_prompts = args.with_prompts

    # experiment: a directory of *.session.json
    if args.experiment:
        d = Path(args.experiment)
        paths = sorted(d.glob("*.session.json"))
        if not paths:
            print(f"[transcript] no *.session.json found in {d}")
            return 1
        for i, sp in enumerate(paths, 1):
            sess = Session.load(sp)
            if args.print:
                print(f"############ session {i}/{len(paths)}: {Path(sess.map).stem} ############")
                print(transcript.render(sess, with_prompts=with_prompts), end="")
            else:
                tp = transcript.write(sess, sp, with_prompts=with_prompts)
                print(f"transcript -> {tp}")
        return 0

    # session: one session JSON
    if args.session:
        sess = Session.load(args.session)
        text = transcript.render(sess, with_prompts=with_prompts)
        if args.print:
            print(text, end="")
        else:
            out = Path(args.out) if args.out else transcript.default_path(args.session)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"transcript -> {out}")
        return 0

    # epoch: one trace JSONL
    if not args.trace:
        raise SystemExit("transcript needs a trace path, or --session / --experiment")
    print(transcript.render_trace(args.trace), end="")
    return 0
