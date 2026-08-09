"""Transcript rendering: per-cycle script dump from a session / trace (no simulation)."""
from pecei import transcript
from pecei.infra import Result, Trace, TraceEvent
from pecei.session import CycleRecord, Session


def _session_with_cycles() -> Session:
    s = Session(map="src/pecei/maps/04_maze.yaml", provider="mock")
    s.cycles.append(CycleRecord(index=1, script="act(FORWARD)\nact(FORWARD)",
                                stop_reason=Result.SUCCESS, rounds=2))
    s.cycles.append(CycleRecord(index=2, script="", stop_reason=Result.COMPILE_ERROR,
                                rounds=0, compile_error="bad arg"))
    return s


def test_render_lists_each_cycle_with_header_and_script():
    text = transcript.render(_session_with_cycles())
    assert "cycle 1 — SUCCESS (2 rounds)" in text
    assert "act(FORWARD)" in text
    assert "cycle 2 — COMPILE_ERROR" in text
    assert "(compile error: bad arg)" in text
    assert "(no script)" in text                          # empty-script cycle


def test_render_with_prompts_includes_recorded_prompt():
    s = Session(map="x.yaml")
    s.cycles.append(CycleRecord(index=1, script="act(FORWARD)", stop_reason=Result.SUCCESS,
                                rounds=1, prompt={"system": "SYS", "user": "USR"}))
    text = transcript.render(s, with_prompts=True)
    assert "[system]" in text and "SYS" in text
    assert "[user]" in text and "USR" in text
    assert "SYS" not in transcript.render(s)              # hidden without the flag


def test_write_dumps_next_to_session(tmp_path):
    sp = tmp_path / "04_maze.session.json"
    out = transcript.write(_session_with_cycles(), sp)
    assert out == tmp_path / "04_maze.transcript.txt"
    assert out.exists()
    assert "cycle 1 — SUCCESS" in out.read_text(encoding="utf-8")


def test_default_path_handles_non_session_suffix(tmp_path):
    assert transcript.default_path(tmp_path / "a.session.json") == tmp_path / "a.transcript.txt"
    assert transcript.default_path(tmp_path / "a.json") == tmp_path / "a.transcript.txt"


def test_render_trace_reads_program_from_first_round(tmp_path):
    t = Trace()
    t.append(TraceEvent(round=1, program="act(FORWARD)\nact(FORWARD)"))
    t.append(TraceEvent(round=2, program=None))
    tp = tmp_path / "x.trace.jsonl"
    t.write(tp)
    text = transcript.render_trace(tp)
    assert "act(FORWARD)" in text
    assert "2 rounds" in text


def test_render_trace_handles_missing_program(tmp_path):
    t = Trace()
    t.append(TraceEvent(round=1, program=None))
    tp = tmp_path / "x.trace.jsonl"
    t.write(tp)
    assert "(no script)" in transcript.render_trace(tp)
