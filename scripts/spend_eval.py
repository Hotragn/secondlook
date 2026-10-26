"""Step 2, benchmark B: predict 2024 spending quantiles from 2023 and score them.

Models (DECISIONS 13): TabPFN v3.5 when the Prior Labs license check passes for
the token in .env, otherwise v2 as the main TabPFN (v2 is always reported), and
four baselines. Scored on the frozen test split (eval/spend_split.json):
pinball loss, P10-P90 coverage and width, overall, by year-1 spending tier and
for the 24-month employer-coverage subgroup. 95% CIs come from resampling test
households; headline metrics are also reported LONGWT-weighted.

Run: uv run python -I scripts/spend_eval.py   (or: make eval-spend)
Writes eval/results/step2_spend.json and data/derived/step2_test_predictions.parquet.
"""

import hashlib
import json
import os
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "derived" / "spend_panel28.parquet"
SPLIT = ROOT / "eval" / "spend_split.json"
SPLIT_SHA256 = "d9577f32d9f4a365f1e4f5cd2f88f0ad796dfe87423a4d0862d1076edc4039aa"
OUT = ROOT / "eval" / "results" / "step2_spend.json"
PREDICTIONS = ROOT / "data" / "derived" / "step2_test_predictions.parquet"
SEED = 20261026
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
BOOTSTRAP = 1000

FEATURES = [
    "age", "female", "region_midwest", "region_south", "region_west",
    "self_rated_health", "self_rated_mental_health",
    "high_bp", "coronary_hd", "angina", "heart_attack", "other_heart", "stroke",
    "emphysema", "high_chol", "cancer", "diabetes", "arthritis", "asthma", "n_conditions",
    "office_visits", "outpatient_visits", "er_visits", "inpatient_stays", "rx_fills", "spend_y1",
]
LOG_FEATURES = ["office_visits", "outpatient_visits", "er_visits", "inpatient_stays", "rx_fills", "spend_y1"]
HEADLINE = ["pinball_q10", "pinball_q50", "pinball_q90", "coverage_p10_p90"]


# ---------------------------------------------------------------- models

def prepare_tabpfn_auth() -> None:
    """Expose only TABPFN_TOKEN from .env, never open a browser, never cache the token."""
    token = dotenv_values(ROOT / ".env").get("TABPFN_TOKEN")
    if token:
        os.environ["TABPFN_TOKEN"] = token.strip()
    os.environ["TABPFN_NO_BROWSER"] = "1"
    import tabpfn.browser_auth as auth

    auth.save_token = lambda token: None  # the library would write it to ~/.cache in plain text


def tabpfn_model(version: str):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    v = {"v2": ModelVersion.V2, "v3.5": ModelVersion.V3_5}[version]

    def fit_predict(Xtr, ytr, Xte, _train, _test):
        reg = TabPFNRegressor.create_default_for_version(v, random_state=SEED)
        reg.fit(Xtr, ytr)
        q = reg.predict(Xte, output_type="quantiles", quantiles=QUANTILES)
        return np.expm1(np.column_stack(q)), str(getattr(reg, "devices_", ""))

    return fit_predict


def same_as_last_year(_Xtr, _ytr, _Xte, _train, test):
    return np.repeat(test["spend_y1"].to_numpy()[:, None], len(QUANTILES), axis=1), "cpu"


def age_sex_quantiles(_Xtr, _ytr, _Xte, train, test):
    groups = train.groupby(["age_band", "female"])["spend_y2"]
    table = {k: np.quantile(v.to_numpy(), QUANTILES) for k, v in groups}
    return np.vstack([table[k] for k in zip(test["age_band"], test["female"])]), "cpu"


def per_quantile(make):
    def fit_predict(Xtr, ytr, Xte, _train, _test):
        cols = []
        for q in QUANTILES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cols.append(make(q).fit(Xtr, ytr).predict(Xte))
        return np.expm1(np.sort(np.column_stack(cols), axis=1)), "cpu"

    return fit_predict


hist_gb = per_quantile(
    lambda q: HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=SEED)
)
linear_qr = per_quantile(
    lambda q: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        QuantileRegressor(quantile=q, alpha=0.0, solver="highs"),
    )
)


# ---------------------------------------------------------------- metrics

def person_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for j, q in enumerate(QUANTILES):
        diff = y - pred[:, j]
        out[f"pinball_q{round(q * 100)}"] = np.maximum(q * diff, (q - 1) * diff)
    lo, hi = pred[:, QUANTILES.index(0.1)], pred[:, QUANTILES.index(0.9)]
    out["coverage_p10_p90"] = ((y >= lo) & (y <= hi)).astype(float)
    out["width_p10_p90"] = hi - lo
    return out


def bootstrap_counts(n_households: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_households, size=(BOOTSTRAP, n_households))
    return np.stack([np.bincount(row, minlength=n_households) for row in draws])


def summarise(values: np.ndarray, mask: np.ndarray, weights: np.ndarray,
              household: np.ndarray, counts: np.ndarray) -> list[dict]:
    """Mean of each column over masked people, with a household-bootstrap 95% CI."""
    w = weights * mask
    sums = np.zeros((counts.shape[1], values.shape[1]))
    np.add.at(sums, household, values * w[:, None])
    totals = np.bincount(household, weights=w, minlength=counts.shape[1])
    point = sums.sum(0) / totals.sum()
    boot = (counts @ sums) / (counts @ totals)[:, None]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return [{"value": float(p), "ci95": [float(a), float(b)]} for p, a, b in zip(point, lo, hi)]


def rounded(d):
    if isinstance(d, dict):
        return {k: rounded(v) for k, v in d.items()}
    if isinstance(d, list):
        return [rounded(v) for v in d]
    if isinstance(d, float):
        return round(d, 4) if abs(d) < 10 else round(d, 1)
    return d


def hardware() -> dict:
    import torch

    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    mem = sysctl("hw.memsize")
    return {
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_gb": round(int(mem) / 2**30, 1) if mem.isdigit() else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
    }


# ---------------------------------------------------------------- main

def main() -> None:
    started = time.perf_counter()
    digest = hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    assert digest == SPLIT_SHA256, "eval/spend_split.json differs from the frozen split in DECISIONS 12"

    t = pd.read_parquet(TABLE)
    train, test = t[t["split"] == "train"].reset_index(drop=True), t[t["split"] == "test"].reset_index(drop=True)

    def matrix(df):
        X = df[FEATURES].astype(float).copy()
        X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
        return X.to_numpy()

    Xtr, Xte = matrix(train), matrix(test)
    ytr = np.log1p(train["spend_y2"].to_numpy())
    y = test["spend_y2"].to_numpy()

    prepare_tabpfn_auth()
    models = {}
    tabpfn_status = {}
    for version in ["v3.5", "v2"]:
        models[f"TabPFN {version}"] = tabpfn_model(version)
    models.update({
        "Same as last year": same_as_last_year,
        "Age-sex group quantiles": age_sex_quantiles,
        "Gradient boosting (quantile)": hist_gb,
        "Linear quantile regression": linear_qr,
    })

    preds, seconds, devices = {}, {}, {}
    for name, fit_predict in models.items():
        t0 = time.perf_counter()
        try:
            p, device = fit_predict(Xtr, ytr, Xte, train, test)
        except Exception as exc:  # v3.5 refuses until the account accepts its license
            if not name.startswith("TabPFN"):
                raise
            tabpfn_status[name] = f"not run: {type(exc).__name__}: {str(exc).splitlines()[0]}"
            print(f"{name}: {tabpfn_status[name]}", flush=True)
            continue
        preds[name] = np.clip(p, 0, None)
        seconds[name] = round(time.perf_counter() - t0, 2)
        devices[name] = device
        if name.startswith("TabPFN"):
            tabpfn_status[name] = "ran"
        print(f"{name}: {seconds[name]}s", flush=True)

    main_model = "TabPFN v3.5" if "TabPFN v3.5" in preds else "TabPFN v2"

    # Per-person metric matrix: columns = model x metric.
    metric_names = list(person_metrics(y, preds[main_model]))
    per_model = {m: person_metrics(y, p) for m, p in preds.items()}
    names = list(preds)
    values = np.column_stack([per_model[m][k] for m in names for k in metric_names])
    paired_metrics = ["pinball_q10", "pinball_q50", "pinball_q90"]
    others = [m for m in names if m != main_model]
    diffs = np.column_stack([
        per_model[m][k] - per_model[main_model][k] for m in others for k in paired_metrics
    ])

    household = pd.factorize(test["DUID"])[0]
    counts = bootstrap_counts(household.max() + 1)
    ones, weights = np.ones(len(test)), test["LONGWT"].to_numpy()
    slices = {
        "all": np.ones(len(test), bool),
        **{f"tier_{k}": (test["tier"] == k).to_numpy() for k in ["zero", "low", "mid", "high"]},
        "employer_coverage_all_24_months": test["esi_all_24_months"].eq(1).to_numpy(),
    }

    t0 = time.perf_counter()
    unweighted, paired = {}, {}
    for s, mask in slices.items():
        cells = iter(summarise(values, mask, ones, household, counts))
        unweighted[s] = {m: {k: next(cells) for k in metric_names} for m in names}
        cells = iter(summarise(diffs, mask, ones, household, counts))
        paired[s] = {m: {k: next(cells) for k in paired_metrics} for m in others}
    cells = iter(summarise(values, slices["all"], weights, household, counts))
    weighted = {m: {k: c for k, c in ((k, next(cells)) for k in metric_names) if k in HEADLINE} for m in names}
    bootstrap_s = round(time.perf_counter() - t0, 2)

    out = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "make eval-spend",
        "hardware": hardware(),
        "population": {
            "n": len(t),
            "filter": "HC-257 Panel 28; YEARIND = 1; LONGWT > 0; AGEY1X and AGEY2X in 18-64; "
                      "INSCOVY1 = INSCOVY2 = 1 (any private coverage both years)",
        },
        "split": {"file": str(SPLIT.relative_to(ROOT)), "sha256": digest,
                  "train": len(train), "test": len(test),
                  "test_households": int(test["DUID"].nunique())},
        "tabpfn": {
            "main_version": main_model.split()[-1],
            "status": tabpfn_status,
            "context_rows": len(train),
            "context_selection": "none: all training rows fit in TabPFN's context",
            "device": {m: d for m, d in devices.items() if m.startswith("TabPFN")},
        },
        "features": FEATURES,
        "target": "TOTEXPY2, 2024 total health care spending in dollars; learned models fit log1p and map back with expm1",
        "quantiles": QUANTILES,
        "bootstrap": {"resamples": BOOTSTRAP, "unit": "test household", "interval": "percentile 2.5-97.5",
                      "note": "ignores MEPS strata and PSUs, so CIs are approximate"},
        "slice_sizes": {s: int(m.sum()) for s, m in slices.items()},
        "runtime_seconds": {"fit_predict": seconds, "bootstrap": bootstrap_s},
        "metrics": {
            "unweighted": unweighted,
            "weighted_longwt": {"all": weighted},
            "paired_difference_vs_main": {
                "definition": f"model minus {main_model}, per person; negative means the model has lower loss",
                **paired,
            },
        },
    }
    out["runtime_seconds"]["total"] = round(time.perf_counter() - started, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rounded(out), indent=1) + "\n")

    pred_cols = {"DUPERSID": test["DUPERSID"], "spend_y2": y}
    for m, p in preds.items():
        for j, q in enumerate(QUANTILES):
            pred_cols[f"{m} q{round(q * 100)}"] = p[:, j]
    pd.DataFrame(pred_cols).to_parquet(PREDICTIONS, index=False)

    print(f"\nmain TabPFN: {main_model}")
    print(f"{'model':30} {'q10':>8} {'q50':>8} {'q90':>8} {'cover':>6} {'width':>9}")
    for m in names:
        r = unweighted["all"][m]
        print(f"{m:30} {r['pinball_q10']['value']:8.0f} {r['pinball_q50']['value']:8.0f} "
              f"{r['pinball_q90']['value']:8.0f} {r['coverage_p10_p90']['value']:6.3f} {r['width_p10_p90']['value']:9.0f}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['runtime_seconds']['total']}s")


if __name__ == "__main__":
    main()
