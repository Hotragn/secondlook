"""Chart for Step 3 (eval A), drawn only from eval/results/step3_reader.json.

Per model: the share of the 260 gold values read correctly, flagged as needing a
check, or read wrong (status colours, always labelled); the share of coverage
examples that reconcile; and seconds per PDF. Peak memory is in the row label.

Run: uv run python -I scripts/reader_chart.py   (or: make eval-reader)
Writes eval/results/step3_reader.png and eval/results/step3_reader.svg.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results" / "step3_reader.json"
OUT = ROOT / "eval" / "results" / "step3_reader"

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, ACCENT = "#e1e0d9", "#c3c2b7", "#2a78d6"
STATUS = [("correct", "Correct", "#0ca30c"), ("needs_check", "Needs checking", "#fab219"), ("wrong", "Wrong", "#d03b3b")]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0, labelsize=9, colors=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main() -> None:
    r = json.loads(RESULTS.read_text())
    models = list(r["results"])
    ys = list(range(len(models)))[::-1]
    labels = []
    for m in models:
        mem = r["results"][m].get("memory", {}).get("peak_ollama_size_bytes")
        labels.append(f"{m}\n{mem / 1e9:.1f} GB resident" if mem else m)

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [2.2, 1, 1], "wspace": 0.12})
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    style(ax)
    ax.set_title("Held-out gold values (20 SBCs × 13 fields)", loc="left", fontsize=10.5, color=INK,
                 fontweight="semibold", pad=10)
    for y, m in zip(ys, models):
        g = r["results"][m]["gold_fields"]
        left = 0.0
        for key, name, color in STATUS:
            share = g["counts"].get(key, 0) / g["n"]
            ax.barh(y, share, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2)
            if share >= 0.06:
                ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if key != "needs_check" else INK)
            left += share
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0%}"))
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", labelsize=9.5, colors=INK)
    ax.set_ylim(-0.6, len(models) - 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in STATUS]
    ax.legend(handles, [n for _, n, _ in STATUS], loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3,
              frameon=False, fontsize=9, labelcolor=INK, handlelength=1, columnspacing=1.2)

    for ax, title, get, fmt, xmax in [
        (axes[1], "Coverage examples that reconcile", lambda x: x["coverage_examples"]["match_rate"],
         lambda v: f"{v:.0%}", 1.0),
        (axes[2], "Seconds per PDF (mean)", lambda x: x["seconds_per_pdf"]["mean"], lambda v: f"{v:.0f} s", None),
    ]:
        style(ax)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="semibold", pad=10)
        vals = [get(r["results"][m]) for m in models]
        for y, v in zip(ys, vals):
            ax.plot([0, v], [y, y], color=ACCENT, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([v], [y], s=60, color=ACCENT, edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.annotate(fmt(v), (v, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)
        ax.set_xlim(0, xmax or max(vals) * 1.3)
        if xmax == 1.0:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0%}"))
        ax.set_yticks(ys, [""] * len(ys))
        ax.set_ylim(-0.6, len(models) - 0.4)

    fig.text(0.012, 0.975, "Reading plan documents locally with Gemma 4", fontsize=13.5, color=INK,
             fontweight="semibold", va="top")
    fig.text(0.012, 0.905, "Every value must be a verified quote from the PDF, parsed by code; anything else is "
             "flagged for checking instead of guessed. Scored once on 20 held-out 2027 SBCs.",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.012, 0.02, "Source: eval/results/step3_reader.json, produced by make eval-reader. "
             "Gold values were transcribed by Claude, not a person (eval/README.md).",
             fontsize=8, color=MUTED, parse_math=False)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.66, bottom=0.14)
    for ext in ("png", "svg"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(ROOT)}.png and .svg")


if __name__ == "__main__":
    main()
