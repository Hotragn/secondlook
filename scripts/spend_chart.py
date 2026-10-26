"""Chart for Step 2: pinball loss and P10-P90 coverage per model, with 95% CIs.

Reads eval/results/step2_spend.json; every number drawn and every word of the
verdict line comes from that file. The main TabPFN row is the accent colour;
other rows are a neutral gray, and the row labels carry model identity.

Run: uv run python -I scripts/spend_chart.py   (or: make eval-spend)
Writes eval/results/step2_spend.png and eval/results/step2_spend.svg.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results" / "step2_spend.json"
OUT = ROOT / "eval" / "results" / "step2_spend"

# Reference palette tokens (dataviz skill, light mode).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, ACCENT = "#e1e0d9", "#c3c2b7", "#2a78d6"

PANELS = [
    ("pinball_q10", "P10 pinball loss"),
    ("pinball_q50", "Median (P50) pinball loss"),
    ("pinball_q90", "P90 pinball loss"),
    ("coverage_p10_p90", "Inside the P10–P90 band"),
]


def tick_dollars(x, _pos=None):
    return f"${x / 1000:g}K" if abs(x) >= 1000 else f"${x:,.0f}"


def verdict(r: dict, main: str) -> str:
    paired = r["metrics"]["paired_difference_vs_main"]["all"]
    beats, ties, loses = [], [], []
    for model, d in paired.items():
        lo, hi = d["pinball_q50"]["ci95"]
        (beats if lo > 0 else loses if hi < 0 else ties).append(model.lower())
    parts = []
    if beats:
        parts.append(f"beats {' and '.join(beats)}")
    if ties:
        parts.append(f"is tied with {' and '.join(ties)}")
    if loses:
        parts.append(f"loses to {' and '.join(loses)}")
    return (f"At the median, {main} {', '.join(parts)} "
            "(tie = the 95% interval of the paired difference includes $0).")


def main() -> None:
    r = json.loads(RESULTS.read_text())
    main_model = f"TabPFN {r['tabpfn']['main_version']}"
    rows = r["metrics"]["unweighted"]["all"]
    order = [main_model] + [m for m in rows if m != main_model]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4.1), sharey=True,
                             gridspec_kw={"wspace": 0.14})
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(order)))[::-1]

    for ax, (metric, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(length=0, labelsize=9, colors=MUTED)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="semibold", pad=10)

        his = []
        for y, model in zip(ys, order):
            cell = rows[model][metric]
            lo, hi = cell["ci95"]
            his.append(hi)
            color = ACCENT if model == main_model else MUTED
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([cell["value"]], [y], s=60, color=color, edgecolors=SURFACE,
                       linewidths=2, zorder=3)
            if model == main_model:
                label = f"{cell['value']:.0%}" if metric.startswith("coverage") else f"${cell['value']:,.0f}"
                ax.annotate(label, (hi, y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK_2)

        if metric.startswith("coverage"):
            ax.set_xlim(0, 1.0)
            ax.axvline(0.8, color=INK_2, linewidth=1, zorder=1)
            ax.annotate("80% target", (0.8, ys[0] + 0.55), xytext=(-4, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8.5, color=INK_2)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0%}"))
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        else:
            ax.set_xlim(0, max(his) * 1.18)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(FuncFormatter(tick_dollars))

    axes[0].set_yticks(ys, order)
    axes[0].tick_params(axis="y", labelsize=10, colors=INK)
    axes[0].set_ylim(-0.6, len(order) - 0.2)

    s = r["split"]
    fig.text(0.012, 0.975, "Predicting next year's health spending from this year's", fontsize=13.5,
             color=INK, fontweight="semibold", va="top")
    fig.text(0.012, 0.905,
             f"MEPS Panel 28 adults 18–64 with private coverage in 2023 and 2024. Trained on {s['train']:,} people, "
             f"tested on {s['test']:,} in {s['test_households']} households. Lower loss is better; "
             "lines are 95% household-bootstrap intervals.",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.012, 0.075, verdict(r, main_model), fontsize=9, color=INK_2)
    fig.text(0.012, 0.025, "Source: eval/results/step2_spend.json, produced by make eval-spend.",
             fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.74, bottom=0.24)
    for ext in ("png", "svg"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(ROOT)}.png and .svg")


if __name__ == "__main__":
    main()
