"""Step 2b: learning curve for TabPFN v2, gradient boosting and linear quantile regression.

Training sizes 100, 200, 400, 800 and the full 2,275 people, 5 random draws per
size (without replacement, seeds recorded), all scored on the same frozen test
set as Step 2. For each size and model: P50 and P90 pinball loss and P10-P90
coverage, averaged over the draws, with a 95% household-bootstrap CI on the
draw-averaged per-person values, the spread across draws, and the paired
difference against TabPFN.

The question, fixed before running: does TabPFN reach its full-data accuracy with
far fewer people than the other methods? A model "reaches" it at the smallest
size whose mean loss is within 5% of that model's own full-data loss.

Run: uv run python -I scripts/spend_learning_curve.py   (or: make eval-spend)
Writes eval/results/step2b_learning_curve.json
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sameaslastyear.spend import (
    SEED, bootstrap_counts, feature_matrix, hist_gb, linear_qr, person_metrics,
    prepare_tabpfn_auth, rounded, summarise, tabpfn_model,
)

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "derived" / "spend_panel28.parquet"
SPLIT = ROOT / "eval" / "spend_split.json"
SPLIT_SHA256 = "d9577f32d9f4a365f1e4f5cd2f88f0ad796dfe87423a4d0862d1076edc4039aa"
OUT = ROOT / "eval" / "results" / "step2b_learning_curve.json"
SIZES = [100, 200, 400, 800]  # plus the full training set
DRAWS = 5
BOOTSTRAP = 1000
WITHIN = 0.05
METRICS = ["pinball_q50", "pinball_q90", "coverage_p10_p90"]
MODELS = {
    "TabPFN v2": lambda seed: tabpfn_model("v2", seed),
    "Gradient boosting (quantile)": lambda seed: hist_gb(seed),
    "Linear quantile regression": lambda seed: linear_qr,
}
MAIN = "TabPFN v2"


def main() -> None:
    started = time.perf_counter()
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == SPLIT_SHA256, "split differs from DECISIONS 12"
    t = pd.read_parquet(TABLE)
    train = t[t["split"] == "train"].reset_index(drop=True)
    test = t[t["split"] == "test"].reset_index(drop=True)
    Xtr_all, Xte = feature_matrix(train), feature_matrix(test)
    ytr_all = np.log1p(train["spend_y2"].to_numpy())
    y = test["spend_y2"].to_numpy()
    sizes = SIZES + [len(train)]
    prepare_tabpfn_auth()

    household = pd.factorize(test["DUID"])[0]
    counts = bootstrap_counts(household.max() + 1, BOOTSTRAP)
    everyone, ones = np.ones(len(test), bool), np.ones(len(test))

    results, draws_log, seconds = {}, [], {m: 0.0 for m in MODELS}
    for i, n in enumerate(sizes):
        per_person = {m: {k: np.zeros(len(test)) for k in METRICS} for m in MODELS}
        draw_means = {m: {k: [] for k in METRICS} for m in MODELS}
        for d in range(DRAWS):
            seed = SEED + 1000 * i + d
            idx = np.random.default_rng(seed).choice(len(train), size=n, replace=False)
            draws_log.append({"size": n, "draw": d, "seed": seed})
            for name, make in MODELS.items():
                t0 = time.perf_counter()
                pred, _ = make(seed)(Xtr_all[idx], ytr_all[idx], Xte, train.iloc[idx], test)
                seconds[name] += time.perf_counter() - t0
                pm = person_metrics(y, np.clip(pred, 0, None))
                for k in METRICS:
                    per_person[name][k] += pm[k] / DRAWS
                    draw_means[name][k].append(float(pm[k].mean()))
        names = list(MODELS)
        values = np.column_stack([per_person[m][k] for m in names for k in METRICS])
        cells = iter(summarise(values, everyone, ones, household, counts))
        others = [m for m in names if m != MAIN]
        diffs = np.column_stack([per_person[m][k] - per_person[MAIN][k]
                                 for m in others for k in METRICS[:2]])
        dcells = iter(summarise(diffs, everyone, ones, household, counts))
        results[str(n)] = {
            m: {k: {**next(cells), "draw_values": draw_means[m][k],
                    "draw_sd": float(np.std(draw_means[m][k], ddof=1))}
                for k in METRICS}
            for m in names
        }
        results[str(n)]["paired_difference_vs_tabpfn"] = {
            m: {k: next(dcells) for k in METRICS[:2]} for m in others
        }
        print(f"n={n}: " + "  ".join(
            f"{m.split()[0]} P50 ${results[str(n)][m]['pinball_q50']['value']:,.0f}" for m in names), flush=True)

    full = str(len(train))
    reaches = {}
    for m in MODELS:
        reaches[m] = {}
        for k in METRICS[:2]:
            target = results[full][m][k]["value"]
            first = next(n for n in sizes if results[str(n)][m][k]["value"] <= target * (1 + WITHIN))
            reaches[m][k] = {
                "full_data_loss": target,
                "smallest_size_within_5pct": first,
                "excess_over_full_by_size_pct": {
                    str(n): 100 * (results[str(n)][m][k]["value"] / target - 1) for n in sizes
                },
            }

    out = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "make eval-spend",
        "question": "Does TabPFN reach its full-data accuracy with far fewer people than the other methods?",
        "rule": f"a model reaches full-data accuracy at the smallest size whose mean loss is within "
                f"{WITHIN:.0%} of its own full-data loss",
        "sizes": sizes,
        "draws_per_size": DRAWS,
        "sampling": "people drawn without replacement from the frozen training set; at the full size every "
                    "draw holds all people and only model seeds differ",
        "draw_seeds": draws_log,
        "test": {"people": len(test), "households": int(test["DUID"].nunique()),
                 "split_sha256": SPLIT_SHA256},
        "ci": "95% household bootstrap (1,000 resamples) of the draw-averaged per-person value",
        "results": results,
        "reaches_full_data_accuracy": reaches,
        "runtime_seconds": {**{m: round(s, 1) for m, s in seconds.items()},
                            "total": round(time.perf_counter() - started, 1)},
    }
    OUT.write_text(json.dumps(rounded(out), indent=1) + "\n")
    for m, r in reaches.items():
        print(f"{m}: P50 within 5% of full at n={r['pinball_q50']['smallest_size_within_5pct']}, "
              f"P90 at n={r['pinball_q90']['smallest_size_within_5pct']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['runtime_seconds']['total']}s")


if __name__ == "__main__":
    main()
