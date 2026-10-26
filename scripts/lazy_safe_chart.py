"""Chart for Step 4b, drawn only from eval/results/step4b_lazy_safe.json.

Top row: how well each flagger finds the people the lazy rule costs more than
$250 (AUC, PR-AUC, recall at the 20% training flag rate). Bottom row: what each
policy costs people (share given the cheapest plan, mean and 90th-percentile
regret). 95% household-bootstrap intervals; TabPFN rows are the accent colour.

Run: uv run python -I scripts/lazy_safe_chart.py   (or: make eval-lazy)
Writes eval/results/step4b_lazy_safe.png and eval/results/step4b_lazy_safe.svg.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results" / "step4b_lazy_safe.json"
OUT = ROOT / "eval" / "results" / "step4b_lazy_safe"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, ACCENT = "#e1e0d9", "#c3c2b7", "#2a78d6"
FLAGGERS = ["TabPFN classifier v2", "Logistic regression", "Gradient boosting", "Year-1 spend rule", "Any chronic condition"]
POLICIES = {
    "Lowest premium (always lazy)": "Always lowest premium",
    "Lazy unless flagged: TabPFN classifier v2 (at_risk_250)": "Lazy unless TabPFN flags",
    "Lazy unless flagged: Logistic regression (at_risk_250)": "Lazy unless logistic flags",
    "Lazy unless flagged: Gradient boosting (at_risk_250)": "Lazy unless boosting flags",
    "Same as last year": "Same as last year",
    "TabPFN expected cost (always)": "Always TabPFN pick",
    "Lazy unless flagged: Oracle flags (reference) (at_risk_250)": "Perfect flags (reference)",
}


def money(x, _p=None):
    return f"${x:,.0f}"


def pct(x, _p=None):
    return f"{x:.0%}"


def panel(ax, title, names, cells, fmt, xmax=None, accent=lambda n: False):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0, labelsize=9, colors=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="semibold", pad=8)
    ys = list(range(len(names)))[::-1]
    top = xmax or 1.25 * max(c["ci95"][1] for c in cells)
    for y, n, c in zip(ys, names, cells):
        color = ACCENT if accent(n) else MUTED
        ax.plot([c["ci95"][0], min(c["ci95"][1], top)], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([c["value"]], [y], s=56, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(fmt(c["value"]), (min(c["ci95"][1], top), y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, top)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_ylim(-0.6, len(names) - 0.4)
    return ys


def main() -> None:
    r = json.loads(RESULTS.read_text())
    det = r["detection_test"]["at_risk_250"]
    pol = r["policies_test"]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), gridspec_kw={"wspace": 0.12, "hspace": 0.55})
    fig.patch.set_facecolor(SURFACE)
    is_tab = lambda n: n.startswith("TabPFN") or "TabPFN flags" in n
    for ax, (key, title, fmt, xmax) in zip(axes[0], [
            ("auc", "AUC: ranks costly cases first", lambda v, _p=None: f"{v:.2f}", 1.0),
            ("pr_auc", "PR-AUC", lambda v, _p=None: f"{v:.2f}", 1.0),
            ("recall", "Recall at the 20% training flag rate", pct, 1.0)]):
        ys = panel(ax, title, FLAGGERS, [det[n][key] for n in FLAGGERS], fmt, xmax, is_tab)
    axes[0][0].set_yticks(ys, FLAGGERS)
    axes[0][0].tick_params(axis="y", labelsize=9.5, colors=INK)
    for ax in axes[0][1:]:
        ax.set_yticks(ys, [""] * len(ys))
    names = list(POLICIES)
    for ax, (key, title, fmt, xmax) in zip(axes[1], [
            ("hit_rate", "Got the truly cheapest plan", pct, 1.0),
            ("mean_regret", "Mean regret per person-menu", money, None),
            ("p90_regret", "90th-percentile regret", money, None)]):
        ys = panel(ax, title, [POLICIES[n] for n in names], [pol[n][key] for n in names], fmt, xmax, is_tab)
    axes[1][0].set_yticks(ys, [POLICIES[n] for n in names])
    axes[1][0].tick_params(axis="y", labelsize=9.5, colors=INK)
    for ax in axes[1][1:]:
        ax.set_yticks(ys, [""] * len(ys))

    lr = r["lazy_regret_test"]
    fig.text(0.012, 0.985, "When is it safe to just take the lowest premium?", fontsize=13.5, color=INK,
             fontweight="semibold", va="top")
    fig.text(0.012, 0.95, f"{r['rows']['test']:,} held-out person-menu cases (758 MEPS adults × 3 real 2027 menus). "
             f"The lowest premium cost more than $250 extra in {lr['share_over_250']:.0%} of them. "
             "Flaggers were trained on other people; lines are 95% household-bootstrap intervals.",
             fontsize=9.5, color=INK_2, va="top", parse_math=False)
    fig.text(0.012, 0.515, "Finding the costly cases (lazy regret over $250)", fontsize=11, color=INK, fontweight="semibold")
    fig.text(0.012, 0.035, r["headline"], fontsize=8.5, color=INK_2, parse_math=False, wrap=True)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.86, bottom=0.14)
    for ext in ("png", "svg"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(ROOT)}.png and .svg")


if __name__ == "__main__":
    main()
