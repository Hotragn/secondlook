"""Every headline number, read from the result JSON that produced it.

Writes eval/NUMBERS.md: each number with its JSON file, key path, 95% interval and the make
command that produces it. Fails if README.md does not show a headline exactly as formatted
here, so the README cannot drift from the results.

Run: uv run python -I scripts/numbers.py   (or: make numbers)
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
OUT = ROOT / "eval" / "NUMBERS.md"
README = ROOT / "README.md"


def pct(x):
    return f"{x:.0%}"


def pct1(x):
    return f"{x:.1%}"


def usd(x):
    return f"${x:,.0f}"


def signed_usd(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


def auc(x):
    return f"{x:.3f}"


def count(x):
    return f"{len(x) if isinstance(x, list) else x:,}"


C = ("pooled",)
READ = ("results", "gemma4:e4b-it-qat", "gold_fields")
FLAG = ("detection_test", "at_risk_250", "TabPFN classifier v2")
LAZY = ("policies_test", "Lazy unless flagged: TabPFN classifier v2 (at_risk_250)")
ORACLE = ("policies_test", "Lazy unless flagged: Oracle flags (reference) (at_risk_250)")
CURVE = ("reaches_full_data_accuracy",)
SPEND = ("metrics", "unweighted", "all")
PAIRED = ("metrics", "paired_difference_vs_main", "all")

# (title, file, make command, rows); a row is (label, key path, format, interval path, in README)
SECTIONS = [
    ("Plan choice on the real 2027 menus (benchmark C)", "step4_decision.json", "make eval-decision", [
        ("Held-out adults", ("people",), count, None, True),
        ("Their households", ("households",), count, None, False),
        ("Lowest premium: given the truly cheapest plan", C + ("lowest_premium", "hit_rate", "value"), pct,
         C + ("lowest_premium", "hit_rate", "ci95"), True),
        ("Lowest premium: mean overpayment (regret)", C + ("lowest_premium", "mean_regret", "value"), usd,
         C + ("lowest_premium", "mean_regret", "ci95"), True),
        ("Same as last year: given the truly cheapest plan", C + ("same_as_last_year", "hit_rate", "value"), pct,
         C + ("same_as_last_year", "hit_rate", "ci95"), True),
        ("Same as last year: mean overpayment", C + ("same_as_last_year", "mean_regret", "value"), usd,
         C + ("same_as_last_year", "mean_regret", "ci95"), True),
        ("TabPFN expected cost: given the truly cheapest plan", C + ("tabpfn_expected", "hit_rate", "value"), pct,
         C + ("tabpfn_expected", "hit_rate", "ci95"), True),
        ("TabPFN expected cost: mean overpayment", C + ("tabpfn_expected", "mean_regret", "value"), usd,
         C + ("tabpfn_expected", "mean_regret", "ci95"), True),
        ("Lowest premium minus TabPFN expected cost, mean overpayment (paired)",
         ("paired_vs_tabpfn_expected", "lowest_premium", "mean_regret_minus_tabpfn"), signed_usd,
         ("paired_vs_tabpfn_expected", "lowest_premium", "ci95"), True),
        ("Same as last year minus TabPFN expected cost (paired)",
         ("paired_vs_tabpfn_expected", "same_as_last_year", "mean_regret_minus_tabpfn"), signed_usd,
         ("paired_vs_tabpfn_expected", "same_as_last_year", "ci95"), True),
        ("Lowest premium: 90th-percentile overpayment", C + ("lowest_premium", "p90_regret", "value"), usd,
         C + ("lowest_premium", "p90_regret", "ci95"), False),
        ("Same as last year: 90th-percentile overpayment", C + ("same_as_last_year", "p90_regret", "value"), usd,
         C + ("same_as_last_year", "p90_regret", "ci95"), False),
    ]),
    ("Plan reader on the held-out plan documents (eval A)", "step3_reader.json", "make eval-reader", [
        ("Held-out plan documents", ("documents",), count, None, True),
        ("Key values scored", READ + ("n",), count, None, True),
        ("gemma4:e4b-it-qat: right", READ + ("accuracy",), pct1, READ + ("accuracy_ci95",), True),
        ("gemma4:e4b-it-qat: wrong", READ + ("wrong_rate",), pct1, None, True),
        ("gemma4:e4b-it-qat: flagged \"check this\"", READ + ("needs_check_rate",), pct1, None, True),
        ("gemma4:e4b-it-qat: right, excluding ambiguous gold values", READ + ("accuracy_excluding_ambiguous",),
         pct1, None, False),
        ("gemma4:e2b-it-qat: right", ("results", "gemma4:e2b-it-qat", "gold_fields", "accuracy"), pct1,
         ("results", "gemma4:e2b-it-qat", "gold_fields", "accuracy_ci95"), True),
        ("gemma4:26b-a4b-it-qat: right", ("results", "gemma4:26b-a4b-it-qat", "gold_fields", "accuracy"), pct1,
         ("results", "gemma4:26b-a4b-it-qat", "gold_fields", "accuracy_ci95"), True),
    ]),
    ("The \"look closer\" check (Step 4b)", "step4b_lazy_safe.json", "make eval-lazy", [
        ("Cases where the lowest premium cost over $250 extra", ("lazy_regret_test", "share_over_250"), pct, None,
         True),
        ("Flagger AUC (TabPFN classifier v2)", FLAG + ("auc", "value"), auc, FLAG + ("auc", "ci95"), True),
        ("Flagger recall", FLAG + ("recall", "value"), pct, FLAG + ("recall", "ci95"), True),
        ("Share of cases flagged", FLAG + ("flag_rate", "value"), pct, FLAG + ("flag_rate", "ci95"), False),
        ("Lazy unless flagged: mean overpayment", LAZY + ("mean_regret", "value"), usd, LAZY + ("mean_regret", "ci95"),
         True),
        ("Lazy unless flagged minus always lowest premium (paired)",
         ("paired", "primary_minus_lowest_premium_mean_regret", "value"), signed_usd,
         ("paired", "primary_minus_lowest_premium_mean_regret", "ci95"), True),
        ("Same policy with perfect flags (reference)", ORACLE + ("mean_regret", "value"), usd,
         ORACLE + ("mean_regret", "ci95"), True),
    ]),
    ("Learning curve: training people needed to get within 5% of full-data loss (Step 2b)",
     "step2b_learning_curve.json", "make eval-spend", [
        ("TabPFN v2, median", CURVE + ("TabPFN v2", "pinball_q50", "smallest_size_within_5pct"), count, None, True),
        ("TabPFN v2, 90th percentile", CURVE + ("TabPFN v2", "pinball_q90", "smallest_size_within_5pct"), count, None,
         True),
        ("Gradient boosting, median",
         CURVE + ("Gradient boosting (quantile)", "pinball_q50", "smallest_size_within_5pct"), count, None, True),
        ("Gradient boosting, 90th percentile",
         CURVE + ("Gradient boosting (quantile)", "pinball_q90", "smallest_size_within_5pct"), count, None, True),
        ("Linear quantile regression, median",
         CURVE + ("Linear quantile regression", "pinball_q50", "smallest_size_within_5pct"), count, None, True),
        ("Linear quantile regression, 90th percentile",
         CURVE + ("Linear quantile regression", "pinball_q90", "smallest_size_within_5pct"), count, None, True),
    ]),
    ("Spend prediction (Step 2, benchmark B)", "step2_spend.json", "make eval-spend", [
        ("Training people", ("split", "train"), count, None, True),
        ("TabPFN v2: median pinball loss per person", SPEND + ("TabPFN v2", "pinball_q50", "value"), usd,
         SPEND + ("TabPFN v2", "pinball_q50", "ci95"), True),
        ("Same as last year: median pinball loss", SPEND + ("Same as last year", "pinball_q50", "value"), usd,
         SPEND + ("Same as last year", "pinball_q50", "ci95"), True),
        ("Same as last year minus TabPFN v2 (paired)", PAIRED + ("Same as last year", "pinball_q50", "value"),
         signed_usd, PAIRED + ("Same as last year", "pinball_q50", "ci95"), True),
        ("Age-sex medians minus TabPFN v2 (paired)", PAIRED + ("Age-sex group quantiles", "pinball_q50", "value"),
         signed_usd, PAIRED + ("Age-sex group quantiles", "pinball_q50", "ci95"), True),
        ("TabPFN v2: actual 2024 spending inside its 10-90% band", SPEND + ("TabPFN v2", "coverage_p10_p90", "value"),
         pct, SPEND + ("TabPFN v2", "coverage_p10_p90", "ci95"), True),
    ]),
]


def get(doc, path):
    for k in path:
        doc = doc[k]
    return doc


def keypath(path) -> str:
    out = ""
    for k in path:
        out += ("." if out else "") + k if re.fullmatch(r"[A-Za-z_]\w*", k) else f'["{k}"]'
    return out


def main() -> None:
    readme = README.read_text()
    lines = ["# Headline numbers", "",
             "Every headline number in README.md, with the result file and key it is read from and the command that",
             "produces it. Generated by `make numbers` (scripts/numbers.py); do not edit by hand. `make numbers` also",
             "fails if README.md shows one of the numbers marked \"README\" differently.", "",
             "Intervals are 95% household-bootstrap intervals stored in the same file.", ""]
    missing = []
    for title, name, make, rows in SECTIONS:
        doc = json.loads((RESULTS / name).read_text())
        if "command" in doc and doc["command"] != make:
            raise SystemExit(f"{name} says it was made by {doc['command']!r}, not {make!r}")
        lines += [f"## {title}", "",
                  f"`{make}` writes [results/{name}](results/{name}) (run {doc.get('run_utc', 'time not recorded')}).",
                  "", "| Number | Value | 95% interval | Key | README |", "|---|---|---|---|---|"]
        for label, path, fmt, ci, in_readme in rows:
            value = fmt(get(doc, path))
            interval = "" if ci is None else " to ".join(fmt(x) for x in get(doc, ci))
            lines.append(f"| {label} | {value} | {interval} | `{keypath(path)}` | {'yes' if in_readme else ''} |")
            if in_readme and value not in readme:
                missing.append(f"{label}: {value} ({name} {keypath(path)})")
        lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)}")
    if missing:
        raise SystemExit("README.md does not show:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
