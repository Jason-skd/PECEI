"""MemoryEvolution tests: the three-stage contract, with regressions for every
bug fixed in the cleanup (no duplicated context, decay that bites, disjoint
buffer/archive, offline fallbacks, real-snapshot tokens, live-score ranking)."""
from __future__ import annotations

from pecei.infra import FailureSnapshot, Result
from pecei.llm import Feedback
from pecei.memory import MemoryEvolution
from pecei.memory.evolution import BufferItem, _describe_snapshot


def _fb(
    *,
    complexity: float | None = 10.0,
    rounds: int = 5,
    stop: Result = Result.SCRIPT_ENDED,
    pos: tuple[int, int] = (1, 2),
    components: tuple[str, ...] = ("brain",),
    compile_error: str | None = None,
) -> Feedback:
    snap = FailureSnapshot(
        pos=pos,
        current_state={
            "eid": "ego",
            "anchor": list(pos),
            "orientation": "NORTH",
            "is_ego": True,
            "components": [
                {"offset": [0, 0], "type": t, "attrs": {}} for t in components
            ],
        },
        complexity=complexity,
    )
    return Feedback(
        stop_reason=stop,
        rounds_used=rounds,
        script="act(FORWARD)",
        failure_snapshot=snap,
        compile_error=compile_error,
    )


# --------------------------------------------------------------------------- #
# Stage 1: buffer / decay / disjointness
# --------------------------------------------------------------------------- #

def test_remember_advances_epoch_internally():
    m = MemoryEvolution(llm=None)
    assert m.epoch == 0
    m.remember(_fb())
    m.remember(_fb())
    assert m.epoch == 2                         # time decay is driven by the module itself


def test_context_has_no_duplicates():
    # Regression: every used to be emitted twice (buffer + appended pool).
    m = MemoryEvolution(llm=None)
    for i in range(3):
        m.remember(_fb(pos=(i, 0)))
    lines = [ln for ln in m.get_current_context().splitlines() if ln]
    assert len(lines) == len(set(lines))        # each ban appears exactly once
    assert len(lines) == 3                       # all three sit in the working set (< capacity)


def test_decay_reduces_live_score():
    m = MemoryEvolution(alpha=0.5)
    item = BufferItem(content="x", complexity=10.0, feedback_epoch=0, score=10.0)
    assert m._current_score(item) == 10.0
    m.epoch = 10
    assert m._current_score(item) == 5.0         # 10 - 0.5 * 10


def test_stale_veteran_displaced_by_fresh_entry():
    # A high-complexity veteran, once decayed, must yield to a fresh low-complexity entry.
    m = MemoryEvolution(alpha=1.0, buffer_capacity=2, llm=None)
    m.remember(_fb(complexity=10.0, pos=(0, 0)))   # A, epoch 0
    m.remember(_fb(complexity=10.0, pos=(1, 0)))   # B, epoch 1
    m.epoch = 20                                   # simulate the passage of time
    m.remember(_fb(complexity=2.0, pos=(2, 0)))    # C: live score 2 beats A's -10

    assert len(m.buffer) == 2
    assert len(m.extra_directives) == 1            # A was demoted into the archive
    contents = [b.content for b in m.buffer] + m.extra_directives
    assert len(contents) == len(set(contents))     # buffer and archive stay disjoint


def test_context_uses_live_not_birth_score():
    # Birth-score ranking would put the high-complexity veteran first; the live
    # decayed score must rank the fresh low-complexity entry above it.
    m = MemoryEvolution(alpha=1.0, buffer_capacity=3, llm=None)
    m.remember(_fb(complexity=10.0, pos=(0, 0)))   # A, birth score 10
    m.epoch = 20
    m.remember(_fb(complexity=2.0, pos=(1, 0)))    # B, birth score 2

    ctx = m.get_current_context()
    assert ctx.index("pos=(1, 0)") < ctx.index("pos=(0, 0)")   # B ranks above A


# --------------------------------------------------------------------------- #
# Stage 2: compression (pure + offline fallbacks)
# --------------------------------------------------------------------------- #

def test_compress_feedback_is_pure():
    m = MemoryEvolution(llm=None)
    m.compress_feedback(_fb())
    assert m.buffer == [] and m.extra_directives == []   # no store mutation


def test_rule_based_directive_honours_triple_contract():
    m = MemoryEvolution(llm=None)
    d = m.remember(_fb())
    assert d.startswith("IF [") and "THEN [" in d and "BECAUSE [" in d
    assert any(b.content == d for b in m.buffer)


def test_empty_llm_output_falls_back_to_rules():
    # Regression: an empty LLM directive used to push a STALE pool entry into the
    # buffer with the new feedback's metadata. It now falls back cleanly.
    m = MemoryEvolution(llm=lambda _p: "")
    d = m.remember(_fb())
    assert d.startswith("IF [") and "THEN [" in d and "BECAUSE [" in d
    assert len(m.buffer) == 1 and m.buffer[0].content == d


def test_llm_exception_falls_back_without_crash():
    def boom(_p):
        raise RuntimeError("down")

    m = MemoryEvolution(llm=boom)
    d = m.remember(_fb())
    assert d.startswith("IF [")                    # deterministic fallback, loop survives


def test_default_complexity_falls_back_to_rounds():
    m = MemoryEvolution(llm=None)
    m.remember(_fb(complexity=None, rounds=7))
    assert m.buffer[0].complexity == 7.0            # snapshot.complexity None -> rounds_used


# --------------------------------------------------------------------------- #
# Stage 3: defragmentation (archive only)
# --------------------------------------------------------------------------- #

def test_defrag_compacts_archive_below_half():
    m = MemoryEvolution(buffer_capacity=2, defrag_threshold=50, llm=None)
    for i in range(10):                             # distinct failures -> steady eviction
        m.remember(_fb(pos=(i, 0), complexity=float(10 + i)))
    assert m.should_defragment()

    before = m.extra_directives_size()
    buffer_snapshot = list(m.buffer)                # working set must be untouched
    m.defragment_memory()

    assert m.extra_directives_size() < before
    assert len(m.extra_directives) == 1             # merged into one blob
    assert m.buffer == buffer_snapshot              # defrag never touches the live buffer


def test_defrag_dedupes_identical_conditions():
    m = MemoryEvolution(buffer_capacity=2, defrag_threshold=50, llm=None)
    for _ in range(8):                              # SAME failure -> identical bans archived
        m.remember(_fb(pos=(1, 1), complexity=10.0))
    assert m.should_defragment()
    assert len(m.extra_directives) > 1

    m.defragment_memory()
    assert len(m.extra_directives) == 1             # identical IF-conditions collapse to one


def test_defrag_is_noop_below_threshold():
    m = MemoryEvolution(defrag_threshold=10_000, llm=None)
    m.remember(_fb())
    pool = list(m.extra_directives)
    m.defragment_memory()
    assert m.extra_directives == pool               # untouched


# --------------------------------------------------------------------------- #
# snapshot rendering: real FailureSnapshot fields
# --------------------------------------------------------------------------- #

def test_describe_snapshot_uses_real_fields():
    snap = FailureSnapshot(
        pos=(3, 4),
        current_state={
            "eid": "ego",
            "anchor": [3, 4],
            "orientation": "EAST",
            "is_ego": True,
            "components": [
                {"offset": [0, 0], "type": "brain", "attrs": {}},
                {"offset": [1, 0], "type": "wood", "attrs": {}},
            ],
        },
        complexity=7.0,
    )
    s = _describe_snapshot(snap)
    assert "pos=(3, 4)" in s
    assert "complexity=7.0" in s
    assert "ego@(3, 4):EAST" in s                   # the ego entity graph is surfaced
    assert "brain" in s and "wood" in s             # component inventory, not dropped


def test_describe_snapshot_none_is_safe():
    assert _describe_snapshot(None) == "state_unknown"
