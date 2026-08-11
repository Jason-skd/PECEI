"""MemoryEvolution -- three-stage evolving memory for the black-box game-clearing agent.

The agent's raw per-cycle memory used to be a naive cache (keep the 5 most
"complex" failures) plus an ever-growing blob of appended hints. That design has
two fatal flaws:

1. Replacement keyed on raw complexity gets stuck in a local optimum -- one
   spectacularly "complex" failure can occupy a slot forever and starve the
   buffer of fresh, explorable experience.
2. Appended hints grow unboundedly and eventually overflow the context window.

This module replaces that with a three-stage evolution algorithm:

- Stage 1  Buffer:  time-decay exploration score
    ``Score = complexity - alpha * (current_epoch - feedback_epoch)``
    Old entries decay as the epoch advances, so high-complexity veterans
    naturally lose their grip and fresh feedback keeps entering the buffer.
- Stage 2  Compress: structured IF-THEN-BECAUSE triples
    Every failure is distilled into a strict ``IF ... THEN ... BECAUSE ...``
    ban with concrete state tokens; no fuzzy words like "careful"/"watch out".
- Stage 3  Defrag: periodic compaction
    Once the archived directives exceed a character budget, they are
    merged/coarsened (by the LLM, or a deterministic fallback) into a shorter
    high-level strategy set.

Data-flow contract -- the working set and the archive are DISJOINT::

    feedback
       |
       v
   compress_feedback()   --[Stage 2]-->  directive (a rule string; PURE, no store mutation)
       |
       v
   update_buffer()       --[Stage 1]-->  BufferItem (working set, <= capacity)
                                                 |
                            on eviction: demoted v
                                       extra_directives  --[Stage 3]-->  defragment_memory()

``get_current_context()`` emits the live working set (buffer, ranked by the
*current* decayed score) followed by the archive (``extra_directives``). A rule
lives in exactly one store at any time, so the rendered context never duplicates
a ban. The epoch is advanced internally by ``remember()``, so time decay always
bites -- the loop never has to manage it.

Design constraints honoured here:

- Zero external dependencies: no Redis, no vector DB, no disk state. All data
  lives in plain in-memory Python objects handed back via the context string.
- Backwards compatible: ``get_current_context()`` keeps its name and shape so
  the main game loop can call it unchanged.
- Observable: every buffer insertion / replacement / demotion / compression /
  defrag is emitted at INFO level on the ``pecei.memory`` logger (library
  modules here do not print; only the interactive CLI layer does).
- Offline-capable: the optional ``llm`` is a plain callable ``prompt: str -> str``.
  When it is ``None`` (or it raises, or returns nothing usable) deterministic
  in-memory fallbacks keep the pipeline running so an experiment still runs
  without any external service.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("pecei.memory")


#: Format contract injected into the LLM compression prompt.
COMPRESS_PROMPT_TEMPLATE: str = (
    "Distil this failure into ONE strict ban. Output exactly: "
    "IF [concrete combination of game-state tokens] "
    "THEN [the action that is forbidden] "
    "BECAUSE [the concrete consequence]. "
    "No fuzzy wording ('careful', 'watch out'); cite concrete state tokens or "
    "numeric values."
)

#: Merge directive for periodic defragmentation.
DEFRAG_PROMPT_TEMPLATE: str = (
    "Merge all of the following bans. Drop duplicates and contradictions, "
    "generalise similar IF-conditions into one higher-level strategy, and keep "
    "the output under 50% of the input length."
)

#: Exploration decay coefficient, configurable at construction time.
DEFAULT_ALPHA: float = 0.1
#: Buffer capacity -- the working set of failure memories.
DEFAULT_BUFFER_CAPACITY: int = 5
#: Defrag triggers when the archived directives exceed this many chars.
DEFAULT_DEFRAG_THRESHOLD: int = 2000


def _default_complexity(feedback: Any) -> float:
    """Extract a numeric difficulty from a Feedback-like object.

    Prefers ``failure_snapshot.complexity`` (entity-aware path cost to goal);
    falls back to ``rounds_used`` (a long burn usually means a hard failure)
    then 1.0 so the pipeline never crashes on a SUCCESS / compile-error feedback.
    """
    snap = getattr(feedback, "failure_snapshot", None)
    if snap is not None:
        c = getattr(snap, "complexity", None)
        if c is not None:
            return float(c)
    rounds = getattr(feedback, "rounds_used", None)
    if rounds is not None:
        return float(rounds)
    return 1.0


@dataclass
class BufferItem:
    """One failure memory held in the evolving working-set buffer."""

    content: str            # distilled ban text (IF/THEN/BECAUSE)
    complexity: float       # raw difficulty signal captured at birth
    feedback_epoch: int     # global epoch when the feedback was born
    score: float            # birth-time snapshot of the decayed score (for logging)

    def age(self, current_epoch: int) -> int:
        return max(0, current_epoch - self.feedback_epoch)

class MemoryEvolution:
    """Core memory & evolution module of the game-clearing agent.

    Main-loop usage (backwards compatible)::

        memory = MemoryEvolution(llm=my_llm)          # or llm=None for pure in-memory
        feedback = session.last_feedback()
        memory.remember(feedback)                      # compress + buffer update + tick
        context = memory.get_current_context()         # unchanged interface
        if memory.should_defragment():
            memory.defragment_memory()
    """

    def __init__(
        self,
        *,
        alpha: float = DEFAULT_ALPHA,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
        defrag_threshold: int = DEFAULT_DEFRAG_THRESHOLD,
        llm: Callable[[str], str] | None = None,
    ) -> None:
        self.alpha = alpha
        self.buffer_capacity = buffer_capacity
        self.defrag_threshold = defrag_threshold
        self.llm = llm
        self.epoch = 0                       # advanced internally by remember()
        self.buffer: list[BufferItem] = []   # stage 1 working set (<= capacity)
        self.extra_directives: list[str] = []  # stage 2/3 archive (evicted + defrag)

    # ------------------------------------------------------------------ #
    # public API (backwards compatible with the main game loop)
    # ------------------------------------------------------------------ #

    def remember(self, feedback: Any) -> str:
        """One-shot ingestion: compress ``feedback`` then update the buffer.

        Auto-advances the epoch, so each call is one exploration round and time
        decay always bites (the loop never has to manage ``epoch`` itself).
        Returns the distilled ban text. Compression never yields an empty
        directive -- an offline fallback fills it in -- so a feedback is always
        represented in the working set.
        """
        directive = self.compress_feedback(feedback)
        if directive:
            self.update_buffer(feedback, directive)
        self.epoch += 1
        return directive

    def get_current_context(self) -> str:
        """Assemble the full "additional prompt" handed to the author.

        The live working set comes first (ranked by *current* decayed score),
        then the archive. The two stores are disjoint, so no ban is ever
        duplicated. Empty memory yields an empty string so the caller's prompt
        building logic is unchanged.
        """
        parts: list[str] = [item.content for item in self._ranked_buffer()]
        parts.extend(self.extra_directives)
        return "\n".join(p for p in parts if p)

    # ------------------------------------------------------------------ #
    # persistence (cross-map carry-over): the learnable state, minus the llm
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Snapshot the learnable state (epoch + buffer + archive + config).

        Plain-JSON so a shared memory can be handed between sessions / processes
        (the warm-start arm of the comparison experiment). The optional ``llm``
        callable is NOT serialised — it is re-attached after ``from_dict``.
        """
        return {
            "alpha": self.alpha,
            "buffer_capacity": self.buffer_capacity,
            "defrag_threshold": self.defrag_threshold,
            "epoch": self.epoch,
            "buffer": [
                {
                    "content": it.content,
                    "complexity": it.complexity,
                    "feedback_epoch": it.feedback_epoch,
                    "score": it.score,
                }
                for it in self.buffer
            ],
            "extra_directives": list(self.extra_directives),
        }

    @classmethod
    def from_dict(cls, data: dict, *, llm: Callable[[str], str] | None = None) -> "MemoryEvolution":
        """Rebuild a MemoryEvolution from :meth:`to_dict` output (llm re-attached)."""
        m = cls(
            alpha=data.get("alpha", DEFAULT_ALPHA),
            buffer_capacity=data.get("buffer_capacity", DEFAULT_BUFFER_CAPACITY),
            defrag_threshold=data.get("defrag_threshold", DEFAULT_DEFRAG_THRESHOLD),
            llm=llm,
        )
        m.epoch = int(data.get("epoch", 0))
        m.buffer = [
            BufferItem(
                content=it["content"],
                complexity=float(it["complexity"]),
                feedback_epoch=int(it["feedback_epoch"]),
                score=float(it["score"]),
            )
            for it in data.get("buffer", [])
        ]
        m.extra_directives = [str(d) for d in data.get("extra_directives", [])]
        return m

    # ------------------------------------------------------------------ #
    # Stage 1: time-decay exploration buffer
    # ------------------------------------------------------------------ #

    def _score(self, complexity: float, feedback_epoch: int) -> float:
        """Exploration score: raw complexity minus a time-decay penalty.

        ``Score = complexity - alpha * (current_epoch - feedback_epoch)``
        The older an entry gets, the less it is worth -- so a stale high-
        complexity veteran can no longer monopolise a slot forever.
        """
        return complexity - self.alpha * (self.epoch - feedback_epoch)

    def _current_score(self, item: BufferItem) -> float:
        """Re-evaluate an incumbent's score at the *current* epoch.

        ``item.score`` is a birth-time snapshot (handy for logging); every live
        decision -- replacement, ranking -- must use this decayed value so time
        decay actually bites.
        """
        return self._score(item.complexity, item.feedback_epoch)

    def _ranked_buffer(self) -> list[BufferItem]:
        """Buffer sorted by live decayed score (highest = most relevant first)."""
        return sorted(self.buffer, key=self._current_score, reverse=True)

    def update_buffer(self, feedback: Any, directive: str) -> None:
        """Insert ``directive`` (born from ``feedback``) into the working set.

        - Buffer not yet full: append directly.
        - Buffer full: if the newcomer's score beats the weakest incumbent's
          *live* score, evict that incumbent (demoting it into the archive) and
          insert the newcomer; otherwise drop the newcomer -- it lost the
          working-set contest and is not worth retaining.

        Plain ``complexity`` is never used for comparison; only the decayed
        score is.
        """
        complexity = _default_complexity(feedback)
        feedback_epoch = self.epoch
        score = self._score(complexity, feedback_epoch)

        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(
                BufferItem(
                    content=directive,
                    complexity=complexity,
                    feedback_epoch=feedback_epoch,
                    score=score,
                )
            )
            logger.info(
                "[Buffer] appended %d/%d complexity=%.2f score=%.2f epoch=%d",
                len(self.buffer), self.buffer_capacity, complexity, score, self.epoch,
            )
            return

        idx_min = min(
            range(len(self.buffer)),
            key=lambda i: self._current_score(self.buffer[i]),
        )
        incumbent = self.buffer[idx_min]
        incumbent_live = self._current_score(incumbent)
        if score > incumbent_live:
            self.buffer[idx_min] = BufferItem(
                content=directive,
                complexity=complexity,
                feedback_epoch=feedback_epoch,
                score=score,
            )
            self._demote(incumbent)
            logger.info(
                "[Buffer] replaced live-score=%.2f (complexity=%.2f, epoch=%d) "
                "-> score=%.2f (complexity=%.2f, epoch=%d); incumbent archived",
                incumbent_live, incumbent.complexity, incumbent.feedback_epoch,
                score, complexity, self.epoch,
            )
        else:
            logger.info(
                "[Buffer] kept working set; newcomer score=%.2f <= min=%.2f, dropped",
                score, incumbent_live,
            )

    def _demote(self, item: BufferItem) -> None:
        """Move a graduated rule from the working set into the archive.

        The archive is the sole input to defragmentation; the working-set buffer
        is never compacted (it holds the currently-relevant hypotheses).
        """
        if item.content:
            self.extra_directives.append(item.content)

    # ------------------------------------------------------------------ #
    # Stage 2: structured IF-THEN-BECAUSE compression
    # ------------------------------------------------------------------ #

    def compress_feedback(self, feedback: Any) -> str:
        """Distil a failure into one strict IF/THEN/BECAUSE ban.

        Pure w.r.t. the memory stores: it returns the directive string and never
        mutates the buffer or the archive. When no LLM is wired in (or it raises,
        or returns nothing usable) a deterministic rule-based fallback produces
        the triple from the feedback fields so the loop still runs offline.
        """
        state_desc = _describe_snapshot(getattr(feedback, "failure_snapshot", None))
        consequence = _describe_failure(feedback)

        if self.llm is not None:
            prompt = (
                f"{COMPRESS_PROMPT_TEMPLATE}\n\n"
                f"Failure feedback:\n"
                f"- stop reason: {getattr(feedback, 'stop_reason', None)}\n"
                f"- rounds used: {getattr(feedback, 'rounds_used', None)}\n"
                f"- failure state: {state_desc}\n"
                f"- consequence: {consequence}\n"
                f"- script excerpt: {(getattr(feedback, 'script', '') or '')[:500]}\n"
            )
            try:
                directive = (self.llm(prompt) or "").strip()
            except Exception as exc:  # noqa: BLE001 - arbitrary llm callable; never crash the loop
                logger.warning("compress_feedback LLM call failed: %s", exc)
                directive = ""
            if directive:
                logger.info("[Compress] LLM directive (%d chars)", len(directive))
                return directive

        directive = self._rule_based_directive(state_desc, consequence)
        logger.info("[Compress] rule-based directive (%d chars)", len(directive))
        return directive

    def _rule_based_directive(self, state_desc: str, consequence: str) -> str:
        """Offline fallback that still honours the strict triple contract."""
        state = state_desc or "unknown_state"
        return (
            f"IF [{state}] THEN [do not repeat the action that led here] "
            f"BECAUSE [{consequence}]"
        )

    # ------------------------------------------------------------------ #
    # Stage 3: periodic defragmentation (archive only)
    # ------------------------------------------------------------------ #

    def extra_directives_size(self) -> int:
        """Total character size of the archived directives."""
        return sum(len(d) for d in self.extra_directives)

    def should_defragment(self) -> bool:
        """Defrag when the archive exceeds the char threshold."""
        return self.extra_directives_size() > self.defrag_threshold

    def defragment_memory(self) -> None:
        """Merge all archived directives into a compact high-level strategy set.

        Only the archive is compacted. The live working-set buffer is left
        untouched (it holds the currently-relevant hypotheses); defrag
        consolidates the longer-term background knowledge that has been demoted
        out of the working set. The merged result overwrites the archive.
        """
        if not self.should_defragment():
            return
        old_text = "\n".join(self.extra_directives)
        old_size = len(old_text)
        target = max(1, old_size // 2)

        merged = ""
        if self.llm is not None:
            prompt = f"{DEFRAG_PROMPT_TEMPLATE}\n\nCurrent bans:\n{old_text}"
            try:
                merged = (self.llm(prompt) or "").strip()
            except Exception as exc:  # noqa: BLE001 - arbitrary llm callable; never crash the loop
                logger.warning("defragment_memory LLM call failed: %s", exc)
        if not merged:
            merged = self._rule_based_merge(old_text, target)

        self.extra_directives = [merged] if merged else []
        logger.info(
            "[Defrag] compressed %d -> %d chars (%d directive(s))",
            old_size, len(merged), len(self.extra_directives),
        )

    def _rule_based_merge(self, old_text: str, target: int) -> str:
        """Deterministic merge fallback: dedupe by IF-condition key, then cap.

        Groups directives by their IF head, keeps the most specific (longest)
        survivor per group, and hard-caps the output at ``target`` chars -- still
        a strict shrink from the input.
        """
        lines = [ln.strip() for ln in old_text.splitlines() if ln.strip()]
        seen: dict[str, str] = {}
        for ln in lines:
            if "IF " in ln:
                key = ln.split("IF ", 1)[1].split("]", 1)[0].strip()[:32]
            else:
                key = ln[:32]
            if key not in seen or len(ln) > len(seen[key]):
                seen[key] = ln
        merged = "\n".join(seen.values())
        if len(merged) > target:
            merged = merged[:target].rsplit("\n", 1)[0]
        return merged


# ---------------------------------------------------------------------- #
# snapshot helpers -- render the REAL FailureSnapshot fields as tokens
# ---------------------------------------------------------------------- #

def _describe_snapshot(snapshot: Any) -> str:
    """Render a FailureSnapshot (pos + ego entity graph + complexity) as tokens.

    ``FailureSnapshot`` carries ``pos``, ``complexity`` and ``current_state``
    (the ego's ``Entity.to_dict()``: anchor, orientation, component inventory).
    All three are surfaced as concrete tokens for the IF-CONDITION; the component
    inventory is the richest signal, so it is included whenever present.
    """
    if snapshot is None:
        return "state_unknown"

    parts: list[str] = []
    pos = getattr(snapshot, "pos", None)
    if pos is not None:
        parts.append(f"pos={tuple(pos)}")
    comp = getattr(snapshot, "complexity", None)
    if comp is not None:
        parts.append(f"complexity={comp}")

    state = getattr(snapshot, "current_state", None)
    if isinstance(state, dict):
        ego_desc = _describe_ego(state)
        if ego_desc:
            parts.append(ego_desc)

    return ",".join(parts) if parts else "state_unknown"


def _describe_ego(ego: dict) -> str:
    """Compact token for an ego entity graph (``Entity.to_dict()``)."""
    anchor = ego.get("anchor")
    orientation = ego.get("orientation")
    ctypes = sorted({c.get("type") for c in ego.get("components", []) if c.get("type")})
    head = "ego"
    if anchor is not None:
        head += f"@{tuple(anchor)}"
    if orientation:
        head += f":{orientation}"
    if ctypes:
        head += f"[{','.join(ctypes)}]"
    return head


def _describe_failure(feedback: Any) -> str:
    """Concrete consequence token: stop reason (+ compile-error text if any)."""
    bits: list[str] = []
    stop = getattr(feedback, "stop_reason", None)
    bits.append(f"stop_reason={getattr(stop, 'value', stop)}")
    err = getattr(feedback, "compile_error", None)
    if err:
        bits.append(f"compile_error={err[:120]}")
    return ", ".join(bits)
