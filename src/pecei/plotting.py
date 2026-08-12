"""Grouped bar chart for the warm-vs-cold comparison (Figure 1).

matplotlib is imported **lazily** inside :func:`plot_comparison`, so this module
is import-safe (and :mod:`pecei.compare` never pulls matplotlib) — the chart is
only rendered when the caller explicitly asks for it (e.g. ``pecei compare
--plot``).

Palette: the two arms are a 2-slot categorical encoding. The validated
CVD-safe pair is warm = orange ``#eb6834`` (the trained prototype) and cold =
blue ``#2a78d6`` (the from-scratch baseline) — orange/blue is the canonical
colorblind-distinguishable pair, and these specific hexes clear the adjacent-
pair CVD separation check.
"""
from __future__ import annotations

from pathlib import Path

from pecei.compare import ARM_COLD, ARM_WARM, ComparisonResult

# Validated CVD-safe 2-slot categorical palette (orange/blue).
WARM_COLOR = "#eb6834"
COLD_COLOR = "#2a78d6"


def plot_comparison(
    result: ComparisonResult,
    out_path: str | Path,
    *,
    metric: str = "epochs",
    title: str | None = None,
) -> Path:
    """Render a grouped bar chart: X = test maps, two bars per map (warm/cold).

    Parameters
    ----------
    metric:
        ``"epochs"`` (default) plots cycles-to-first-success;
        ``"rounds"`` plots total-rounds-to-first-success. Either way lower =
        better, so the warm bars should sit below the cold bars if the
        prototype learned.
    """
    import matplotlib.pyplot as plt  # lazy: keep compare.py matplotlib-free

    if metric not in ("epochs", "rounds"):
        raise ValueError(f"metric must be 'epochs' or 'rounds', got {metric!r}")
    field = "epochs_to_success" if metric == "epochs" else "total_rounds"

    labels = [m.slug for m in result.warm]
    warm_vals = [getattr(m, field) for m in result.warm]
    cold_vals = [getattr(m, field) for m in result.cold]
    warm_solved = [m.solved for m in result.warm]
    cold_solved = [m.solved for m in result.cold]

    n = len(labels)
    x = range(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * n + 2), 5))
    bars_w = ax.bar([i - width / 2 for i in x], warm_vals, width,
                    label="Warm-start (trained)", color=WARM_COLOR, edgecolor="white", linewidth=0.5)
    bars_c = ax.bar([i + width / 2 for i in x], cold_vals, width,
                    label="Cold-start (from scratch)", color=COLD_COLOR, edgecolor="white", linewidth=0.5)

    # Hatch the unsolved bars so they read distinctly from solved-in-budget.
    for bars, solved_flags in [(bars_w, warm_solved), (bars_c, cold_solved)]:
        for b, s in zip(bars, solved_flags):
            if not s:
                b.set_hatch("//")
                b.set_alpha(0.55)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Cycles to first success" if metric == "epochs" else "Total rounds to first success")
    ax.set_xlabel("Test map")
    ax.set_title(title or "Warm-start vs cold-start (lower is better)")
    ax.legend(frameon=False)
    # Y axis is a count — give it integer ticks starting at 0.
    ax.set_ylim(bottom=0)
    if metric == "epochs":
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
