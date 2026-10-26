"""Chart for Step 4 (benchmark C), drawn only from eval/results/step4_decision.json.

Per strategy, pooled over the three 2027 menus: share of people given the truly
cheapest plan, mean regret and 90th-percentile regret, with 95% household-bootstrap
intervals. TabPFN expected cost (the method under test) is the accent colour.

Run: uv run python -I scripts/decision_chart.py   (or: make eval-decision)
Writes eval/results/step4_decision.png and eval/results/step4_decision.svg.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results" / "step4_decision.json"
OUT = ROOT / "eval" / "results" / "step4_decision"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, ACCENT = "#e1e0d9", "#c3c2b7", "#2a78d6"
NAMES = {"tabpfn_expected": "TabPFN expected cost", "tabpfn_risk_capped": "TabPFN risk-capped",
         "lowest_premium": "Lowest premium", "same_as_last_year": "Same as last year",
         "lowest_deductible": "Lowest deductible"}
PANELS = [("hit_rate", "Got the truly cheapest plan"), ("mean_regret", "Mean regret per person"),
          ("p90_regret", "90th-percentile regret")]


def money(x, _p=None):
    return f"${x:,.0f}"


def main() -> None:
    r = json.loads(RESULTS.read_text())
    pooled = r["pooled"]
    order = list(NAMES)
    ys = list(range(len(order)))[::-1]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True, gridspec_kw={"wspace": 0.14})
    fig.patch.set_facecolor(SURFACE)
    off = []
    for ax, (metric, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(length=0, labelsize=9, colors=MUTED)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="semibold", pad=10)
        cells = {s: pooled[s][metric] for s in order}
        if metric == "hit_rate":
            top = 1.0
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0%}"))
        else:
            top = 1.25 * max(c["ci95"][1] for s, c in cells.items() if s != "lowest_deductible")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(FuncFormatter(money))
        ax.set_xlim(0, top)
        for y, s in zip(ys, order):
            c = cells[s]
            color = ACCENT if s == "tabpfn_expected" else MUTED
            if c["value"] > top:
                off.append(f"lowest deductible {title.lower()} {money(c['value'])}")
                continue
            ax.plot([c["ci95"][0], min(c["ci95"][1], top)], [y, y], color=color, linewidth=2,
                    solid_capstyle="round", zorder=2)
            ax.scatter([c["value"]], [y], s=60, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
            label = f"{c['value']:.0%}" if metric == "hit_rate" else money(c["value"])
            ax.annotate(label, (min(c["ci95"][1], top), y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2)
    axes[0].set_yticks(ys, [NAMES[s] for s in order])
    axes[0].tick_params(axis="y", labelsize=10, colors=INK)
    axes[0].set_ylim(-0.6, len(order) - 0.4)

    paired = r["paired_vs_tabpfn_expected"]
    lp, sl = paired["lowest_premium"], paired["same_as_last_year"]
    verdict = (f"Lowest premium beat TabPFN by ${-lp['mean_regret_minus_tabpfn']:,.0f} mean regret per person "
               f"(95% CI ${-lp['ci95'][1]:,.0f} to ${-lp['ci95'][0]:,.0f}); TabPFN beat same-as-last-year by "
               f"${sl['mean_regret_minus_tabpfn']:,.0f} (${sl['ci95'][0]:,.0f} to ${sl['ci95'][1]:,.0f}).")
    fig.text(0.012, 0.975, "Which way of choosing a 2027 plan costs people least", fontsize=13.5, color=INK,
             fontweight="semibold", va="top")
    fig.text(0.012, 0.905, f"{r['people']} held-out MEPS adults, each run through 3 real 2027 employer menus (WA PEBB, "
             "OR PEBB, Tennessee) with their actual 2024 care.\nRegret = what they paid minus the cheapest plan. "
             "Lines are 95% household-bootstrap intervals.", fontsize=9.5, color=INK_2, va="top", parse_math=False,
             linespacing=1.4)
    fig.text(0.012, 0.07, verdict, fontsize=9, color=INK_2, parse_math=False)
    fig.text(0.012, 0.02, "Off scale, not drawn: " + "; ".join(off) + ". Source: eval/results/step4_decision.json "
             "(make eval-decision); approximations are listed in the JSON.", fontsize=8, color=MUTED, parse_math=False)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.68, bottom=0.2)
    for ext in ("png", "svg"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(ROOT)}.png and .svg")


if __name__ == "__main__":
    main()
