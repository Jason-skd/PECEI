"""MemoryEvolution -- three-stage evolving memory for the black-box game-clearing agent.

The agent's raw per-cycle memory used to be a naive cache (keep the 5 most
"complex" failures) plus an ever-growing blob of appended hints. That design has
two fatal flaws:

1. Buffer replacement keyed on raw complexity gets stuck in a local optimum --
   one spectacularly "complex" failure can occupy a slot forever and starve the
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
    Once the accumulated extra directives exceed a character budget, all of
    them are merged/coarsened by the LLM into a shorter high-level strategy set.

Design constraints honoured here:

- Zero external dependencies: no Redis, no vector DB, no disk state. All data
  lives in plain in-memory Python objects / the current prompt context.
- Backwards compatible: ``get_current_context()`` keeps its name and shape so
  the main game loop can keep calling it unchanged.
- Observable: every buffer replacement / compression / defrag logs a one-line
  ``print`` (e.g. ``[Defrag] Compressed from 2500 to 1100 chars``).

The optional ``llm`` argument is a plain callable ``prompt: str -> str``. When
it is ``None`` the module falls back to deterministic in-memory rules so the
experiment still runs without any external service; wiring in the real
``LLMProvider`` is a one-liner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("pecei.memory")


#: Hard-coded format contract injected into the LLM compression prompt.
COMPRESS_PROMPT_TEMPLATE: str = (
    "请将本次失败抽象为一条禁令。输出格式严格为："
    "IF [游戏状态特征组合] THEN [禁止执行的动作] BECAUSE [简述后果]。"
    "严禁使用模糊词汇（如'小心''注意'），必须提及具体的数值或状态标识。"
)

#: Hard-coded merge directive for periodic defragmentation.
DEFRAG_PROMPT_TEMPLATE: str = (
    "请合并以下所有禁令，剔除重复或矛盾的项。"
    "将同类的IF条件归纳为更通用的高阶策略，"
    "输出长度必须压缩至现有文本的50%以内。"
)

#: Exploration decay coefficient, configurable at construction time.
DEFAULT_ALPHA: float = 0.1
#: Buffer capacity -- the top-5 working set of failure memories.
DEFAULT_BUFFER_CAPACITY: int = 5
#: Defrag triggers when the accumulated extra directives exceed this many chars.
DEFAULT_DEFRAG_THRESHOLD: int = 2000


def _default_complexity(feedback: Any) -> float:
    """Extract a numeric complexity from a Feedback-like object.

    Prefers ``failure_snapshot.complexity``; falls back to ``rounds_used``
    (a long burn usually means a hard failure) then 1.0 so the pipeline never
    crashes on a SUCCESS / compile-error feedback.
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
    """One failure memory held in the evolving buffer."""

    content: str                          # distilled ban text (IF/THEN/BECAUSE)
    complexity: float                      # raw difficulty signal
    feedback_epoch: int                    # global epoch when the feedback was born
    score: float                           # complexity - alpha * age

    def age(self, current_epoch: int) -> int:
        return max(0, current_epoch - self.feedback_epoch)


class MemoryEvolution:
    """Core memory & evolution module of the game-clearing agent.

    Main loop usage (backwards compatible)::

        memory = MemoryEvolution(llm=my_llm)          # or llm=None for pure in-memory
        feedback = session.last_feedback()
        memory.remember(feedback)                      # compress + buffer update
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
        llm: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.alpha = alpha
        self.buffer_capacity = buffer_capacity
        self.defrag_threshold = defrag_threshold
        self.llm = llm
        self.epoch = 0                      # global round counter, advanced by the loop
        self.buffer: list[BufferItem] = []  # stage 1 working set (<= capacity)
        self.extra_directives: list[str] = []  # stage 2/3 appended hint pool

    # ------------------------------------------------------------------ #
    # public API (backwards compatible with the main game loop)
    # ------------------------------------------------------------------ #

    def remember(self, feedback: Any) -> str:
        """One-shot ingestion: compress ``feedback`` then update the buffer.

        Returns the distilled ban text so the caller can see what was learned.
        """
        directive = self.compress_feedback(feedback)
        self.update_buffer(feedback)
        return directive

    def get_current_context(self) -> str:
        """Assemble the full "additional prompt" handed to the author.

        Buffer bans come first (ranked by score), then the extra directives
        pool. Empty memory yields an empty string so the caller's prompt
        building logic is unchanged.
        """
        parts: list[str] = []
        for item in sorted(self.buffer, key=lambda b: b.score, reverse=True):
            parts.append(item.content)
        parts.extend(self.extra_directives)
        return "\n".join(p for p in parts if p)

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

        ``item.score`` is a birth-time snapshot; comparisons for replacement
        must use the live decayed value so time decay actually bites.
        """
        return self._score(item.complexity, item.feedback_epoch)

    def update_buffer(self, feedback: Any) -> None:
        """Insert ``feedback`` into the buffer using the exploration score.

        - Buffer not yet full (capacity 5): append directly, no replacement.
        - Buffer full: compare scores -- if the new entry's Score beats the
          lowest-Score incumbent, evict that loser and insert the newcomer.
          Plain ``complexity`` comparison is never used.

        The replacement is logged for experiment observation.
        """
        complexity = _default_complexity(feedback)
        feedback_epoch = self.epoch
        score = self._score(complexity, feedback_epoch)
        directive = self._latest_directive(feedback)

        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(
                BufferItem(
                    content=directive,
                    complexity=complexity,
                    feedback_epoch=feedback_epoch,
                    score=score,
                )
            )
            print(
                f"[Buffer] appended {len(self.buffer)}/{self.buffer_capacity} "
                f"complexity={complexity:.2f} score={score:.2f} epoch={self.epoch}"
            )
            return

        # Buffer is full: replace the weakest incumbent iff the newcomer wins.
        # All incumbents are re-scored at the *current* epoch so the decayed
        # veterans lose their grip instead of camping on birth-time scores.
        idx_min = min(
            range(len(self.buffer)),
            key=lambda i: self._current_score(self.buffer[i]),
        )
        incumbent = self.buffer[idx_min]
        incumbent_live = self._current_score(incumbent)
        if score > incumbent_live:
            victim = incumbent
            self.buffer[idx_min] = BufferItem(
                content=directive,
                complexity=complexity,
                feedback_epoch=feedback_epoch,
                score=score,
            )
            print(
                f"[Buffer] replaced live-score={incumbent_live:.2f} "
                f"(complexity={victim.complexity:.2f}, epoch={victim.feedback_epoch}) "
                f"-> score={score:.2f} (complexity={complexity:.2f}, epoch={self.epoch})"
            )
        else:
            print(
                f"[Buffer] kept buffer; new score={score:.2f} <= min="
                f"{incumbent_live:.2f}, no replacement"
            )

    # ------------------------------------------------------------------ #
    # Stage 2: structured IF-THEN-BECAUSE compression
    # ------------------------------------------------------------------ #

    def compress_feedback(self, feedback: Any) -> str:
        """Distil a failure into one strict IF/THEN/BECAUSE ban.

        The LLM prompt is force-injected with the hard-coded format contract;
        no fuzzy wording is allowed, concrete state tokens are mandatory. When
        no LLM is wired in, a deterministic rule-based fallback produces the
        triple from the feedback fields so the loop still runs offline.
        """
        snapshot = getattr(feedback, "failure_snapshot", None)
        state_desc = _describe_snapshot(snapshot)
        consequence = _describe_failure(feedback)

        if self.llm is not None:
            prompt = (
                f"{COMPRESS_PROMPT_TEMPLATE}\n\n"
                f"失败反馈如下：\n"
                f"- 停止原因: {getattr(feedback, 'stop_reason', None)}\n"
                f"- 已用轮数: {getattr(feedback, 'rounds_used', None)}\n"
                f"- 失败状态: {state_desc}\n"
                f"- 后果: {consequence}\n"
                f"- 涉及脚本片段: {(getattr(feedback, 'script', '') or '')[:500]}\n"
            )
            try:
                directive = (self.llm(prompt) or "").strip()
            except Exception as exc:  # never let a compression hiccup kill the loop
                logger.warning("compress_feedback LLM call failed: %s", exc)
                directive = self._rule_based_directive(state_desc, consequence)
        else:
            directive = self._rule_based_directive(state_desc, consequence)

        if directive:
            self.extra_directives.append(directive)
            print(f"[Compress] +1 directive ({len(directive)} chars)")
        else:
            print("[Compress] empty directive, skipped")
        return directive

    def _rule_based_directive(self, state_desc: str, consequence: str) -> str:
        """Offline fallback that still honours the strict triple contract."""
        state = state_desc or "unknown_state"
        return (
            f"IF [{state}] THEN [do not repeat the failed action at this "
            f"state] BECAUSE [{consequence}]"
        )

    def _latest_directive(self, feedback: Any) -> str:
        """The directive most recently produced for ``feedback`` (buffer slot text)."""
        # compress_feedback already appended it; the newest pool entry is ours.
        if self.extra_directives:
            return self.extra_directives[-1]
        return self._rule_based_directive(
            _describe_snapshot(getattr(feedback, "failure_snapshot", None)),
            _describe_failure(feedback),
        )

    # ------------------------------------------------------------------ #
    # Stage 3: periodic defragmentation
    # ------------------------------------------------------------------ #

    def extra_directives_size(self) -> int:
        """Total character size of the accumulated extra directives."""
        return sum(len(d) for d in self.extra_directives)

    def should_defragment(self) -> bool:
        """Defrag when the extra-directive pool exceeds the char threshold."""
        return self.extra_directives_size() > self.defrag_threshold

    def defragment_memory(self) -> None:
        """Merge all extra directives into a compact high-level strategy set.

        Only triggers when the accumulated directives exceed the threshold
        (callers may also call it unconditionally at loop start; it is a no-op
        below the threshold). The LLM is asked to drop duplicates/contradictions
        and generalise, and must output <= 50% of the input length. The result
        overwrites the extra-directive pool in place.
        """
        if not self.should_defragment():
            return
        old_text = "\n".join(self.extra_directives)
        old_size = len(old_text)
        target = max(1, old_size // 2)

        merged = ""
        if self.llm is not None:
            prompt = f"{DEFRAG_PROMPT_TEMPLATE}\n\n当前禁令列表：\n{old_text}"
            try:
                merged = (self.llm(prompt) or "").strip()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("defragment_memory LLM call failed: %s", exc)
        if not merged:
            merged = self._rule_based_merge(old_text, target)

        self.extra_directives = [merged] if merged else []
        new_size = len(merged)
        print(
            f"[Defrag] Compressed from {old_size} to {new_size} chars "
            f"({len(self.extra_directives)} directive(s))"
        )

    def _rule_based_merge(self, old_text: str, target: int) -> str:
        """Deterministic in-memory merge fallback: dedupe by IF-condition key.

        Groups directives by their IF head, keeps the most specific survivor per
        group, then hard-caps the output at ``target`` chars when the LLM is
        unavailable -- still a strict shrink from the input.
        """
        lines = [ln.strip() for ln in old_text.splitlines() if ln.strip()]
        seen: dict[str, str] = {}
        for ln in lines:
            if "IF " in ln:
                key = ln.split("IF ", 1)[1].split("]")[0].strip()[:24]
            else:
                key = ln[:24]
            # prefer longer (more specific) survivors within the same group
            if key not in seen or len(ln) > len(seen[key]):
                seen[key] = ln
        merged = "\n".join(seen.values())
        if len(merged) > target:
            merged = merged[:target].rsplit("\n", 1)[0]
        return merged


# ---------------------------------------------------------------------- #
# small helpers
# ---------------------------------------------------------------------- #

def _describe_snapshot(snapshot: Any) -> str:
    if snapshot is None:
        return "state_unknown"
    pos = getattr(snapshot, "pos", None)
    parts: list[str] = []
    if pos is not None:
        try:
            parts.append(f"pos={tuple(pos)}")
        except TypeError:
            parts.append(f"pos={pos}")
    for key in ("energy", "hp", "round", "complexity"):
        val = getattr(snapshot, key, None)
        if val is not None:
            parts.append(f"{key}={val}")
    return ",".join(parts) if parts else "state_unknown"


def _describe_failure(feedback: Any) -> str:
    stop = getattr(feedback, "stop_reason", None)
    if stop is not None:
        return f"stop_reason={getattr(stop, 'value', stop)}"
    return "stop_reason=unknown"
