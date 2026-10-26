"""Step 4b: the lazy-safe check (pre-registered in DECISIONS 25).

Who does the lazy rule ("take the lowest premium") cost real money? For every
person and 2027 menu, lazy_regret = true cost of the lowest-premium plan minus
the cheapest plan's true cost (benchmark C pricing). Classifiers trained only on
the 2,275 training people (x 3 menus) flag rows where lazy_regret > $250 (also
> $500). Thresholds give a 20% flag rate in 5-fold household-grouped
cross-validation on training rows. The test set is scored once. The policy
"lazy unless flagged" takes the lowest premium except for flagged rows, which get
benchmark C's TabPFN expected-cost pick.

Run: uv run python -I scripts/lazy_safe_eval.py [--train-only [--skip-tabpfn]]   (or: make eval-lazy)
--train-only prints cross-validated training metrics and never reads test labels.
Writes eval/results/step4b_lazy_safe.json
"""

import argparse
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sameaslastyear import menus
from sameaslastyear.scenarios import QUANTILES, SEED, choices, cost_matrix, household_counts, load_events, regret_metrics
from sameaslastyear.spend import feature_matrix, prepare_tabpfn_auth

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "derived" / "spend_panel28.parquet"
PREDICTIONS = ROOT / "data" / "derived" / "step2_test_predictions.parquet"
SPLIT = ROOT / "eval" / "spend_split.json"
SPLIT_SHA256 = "d9577f32d9f4a365f1e4f5cd2f88f0ad796dfe87423a4d0862d1076edc4039aa"
OUT = ROOT / "eval" / "results" / "step4b_lazy_safe.json"
LABELS = {"at_risk_250": 250.0, "at_risk_500": 500.0}
FLAG_RATE = 0.20
FOLDS = 5
BOOTSTRAP = 1000


def tabpfn():
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, random_state=SEED)


LEARNERS = {
    "TabPFN classifier v2": tabpfn,
    "Logistic regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                 LogisticRegression(C=1.0, max_iter=2000)),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
}
RULES = ["Year-1 spend rule", "Any chronic condition"]


def rows(df: pd.DataFrame, n_menus: int) -> np.ndarray:
    """Year-1 features repeated for each menu, plus menu dummies."""
    X = feature_matrix(df)
    return np.vstack([np.hstack([X, np.tile(np.eye(n_menus)[m], (len(df), 1))]) for m in range(n_menus)])


def rule_scores(df: pd.DataFrame, n_menus: int) -> dict[str, np.ndarray]:
    return {"Year-1 spend rule": np.tile(df["spend_y1"].to_numpy(), n_menus),
            "Any chronic condition": np.tile((df["n_conditions"] > 0).astype(float).to_numpy(), n_menus)}


def fit_scores(name: str, Xtr, ytr, Xte) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LEARNERS[name]().fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]


def oof_scores(name: str, X, y, groups) -> np.ndarray:
    out = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        out[va] = fit_scores(name, X[tr], y[tr], X[va])
    return out


def classification(y, score, threshold, household, counts) -> dict:
    flag = score > threshold
    w = counts[:, household]  # bootstrap weight of each row = count of its household
    def stats(wt):
        tp = (wt * (flag & (y == 1))).sum()
        return {"auc": roc_auc_score(y, score, sample_weight=wt) if len(set(y[wt > 0])) > 1 else np.nan,
                "pr_auc": average_precision_score(y, score, sample_weight=wt),
                "recall": tp / max((wt * (y == 1)).sum(), 1e-9),
                "precision": tp / max((wt * flag).sum(), 1e-9),
                "flag_rate": (wt * flag).sum() / wt.sum()}
    point = stats(np.ones(len(y)))
    boot = [stats(wt) for wt in w]
    out = {k: {"value": float(v), "ci95": [float(np.nanpercentile([b[k] for b in boot], 2.5)),
                                           float(np.nanpercentile([b[k] for b in boot], 97.5))]}
           for k, v in point.items()}
    prob = score if score.max() <= 1 else None
    if prob is not None:
        out["brier"] = float(np.mean((prob - y) ** 2))
        edges = np.unique(np.quantile(prob, np.linspace(0, 1, 6)))
        bins = np.clip(np.digitize(prob, edges[1:-1]), 0, len(edges) - 2)
        out["calibration"] = [{"mean_predicted": float(prob[bins == b].mean()), "observed": float(y[bins == b].mean()),
                               "rows": int((bins == b).sum())} for b in range(len(edges) - 1) if (bins == b).any()]
    out["threshold"] = float(threshold)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--skip-tabpfn", action="store_true", help="dry runs only; requires --train-only")
    args = ap.parse_args()
    assert args.train_only or not args.skip_tabpfn, "--skip-tabpfn is only for --train-only dry runs"
    if args.skip_tabpfn:
        LEARNERS.pop("TabPFN classifier v2")
    started = time.perf_counter()
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == SPLIT_SHA256, "spend split changed"
    t = pd.read_parquet(TABLE)
    train = t[t["split"] == "train"].reset_index(drop=True)
    test = t[t["split"] == "test"].reset_index(drop=True)
    plans_by_sponsor = menus.load()
    sponsors = list(plans_by_sponsor)
    k = len(sponsors)
    ev24 = load_events(2024, set(t["DUPERSID"]))

    # Lazy regret per row (menu-major order: all people for menu 0, then menu 1, ...).
    truth_tr, lazy_tr = [], []
    for s in sponsors:
        c = cost_matrix(plans_by_sponsor[s], ev24, list(train["DUPERSID"]))
        lp = min(range(c.shape[1]), key=lambda j: (plans_by_sponsor[s][j].premium_monthly, plans_by_sponsor[s][j].deductible))
        truth_tr.append(c)
        lazy_tr.append(c[:, lp] - c.min(1))
    lazy_tr = np.concatenate(lazy_tr)
    Xtr = rows(train, k)
    groups = np.tile(train["DUID"].to_numpy(), k)
    prepare_tabpfn_auth()

    thresholds, cv = {}, {}
    for label, cut in LABELS.items():
        y = (lazy_tr > cut).astype(int)
        cv[label] = {"positive_rate": float(y.mean())}
        thresholds[label] = {}
        for name in LEARNERS:
            t0 = time.perf_counter()
            oof = oof_scores(name, Xtr, y, groups)
            thresholds[label][name] = float(np.quantile(oof, 1 - FLAG_RATE))
            cv[label][name] = {"oof_auc": float(roc_auc_score(y, oof)), "oof_pr_auc": float(average_precision_score(y, oof)),
                               "seconds": round(time.perf_counter() - t0, 1)}
            print(f"[train CV] {label} {name}: AUC {cv[label][name]['oof_auc']:.3f} PR-AUC {cv[label][name]['oof_pr_auc']:.3f}", flush=True)
        rs = rule_scores(train, k)
        thresholds[label]["Year-1 spend rule"] = float(np.quantile(rs["Year-1 spend rule"], 1 - FLAG_RATE))
        thresholds[label]["Any chronic condition"] = 0.5
        for name in RULES:
            cv[label][name] = {"train_auc": float(roc_auc_score(y, rs[name]))}
    if args.train_only:
        print(json.dumps({"thresholds": thresholds, "cv": cv}, indent=1))
        return

    # ---- the one test scoring ----
    ev23 = load_events(2023, set(test["DUPERSID"]))
    pred = pd.read_parquet(PREDICTIONS).set_index("DUPERSID").loc[test["DUPERSID"]]
    q = pred[[f"TabPFN v2 {x}" for x in QUANTILES]].to_numpy()
    regret_of, lazy_te = {}, []
    for s, c_tr in zip(sponsors, truth_tr):
        plans = plans_by_sponsor[s]
        truth = cost_matrix(plans, ev24, list(test["DUPERSID"]))
        last = cost_matrix(plans, ev23, list(test["DUPERSID"]))
        pick = choices(plans, truth, last, c_tr, q, train["spend_y2"].to_numpy())
        best = truth.min(1)
        for strat in ("lowest_premium", "same_as_last_year", "tabpfn_expected"):
            regret_of.setdefault(strat, []).append(truth[np.arange(len(test)), pick[strat]] - best)
        lazy_te.append(truth[np.arange(len(test)), pick["lowest_premium"]] - best)
    regret_of = {s: np.concatenate(v) for s, v in regret_of.items()}
    lazy_te = np.concatenate(lazy_te)
    Xte = rows(test, k)
    household = np.tile(pd.factorize(test["DUID"])[0], k)
    counts = household_counts(pd.factorize(test["DUID"])[0], BOOTSTRAP)

    detection, flags = {}, {}
    for label, cut in LABELS.items():
        y = (lazy_tr > cut).astype(int)
        y_te = (lazy_te > cut).astype(int)
        detection[label] = {"test_positive_rate": float(y_te.mean())}
        rs_te = rule_scores(test, k)
        for name in [*LEARNERS, *RULES]:
            score = fit_scores(name, Xtr, y, Xte) if name in LEARNERS else rs_te[name]
            detection[label][name] = classification(y_te, score, thresholds[label][name], household, counts)
            flags[(label, name)] = score > thresholds[label][name]
            print(f"[test] {label} {name}: AUC {detection[label][name]['auc']['value']:.3f}", flush=True)
        flags[(label, "Oracle flags (reference)")] = y_te.astype(bool)

    # Policy: lazy unless flagged; flagged rows get the TabPFN expected-cost pick.
    policies = {"Lowest premium (always lazy)": regret_of["lowest_premium"],
                "Same as last year": regret_of["same_as_last_year"],
                "TabPFN expected cost (always)": regret_of["tabpfn_expected"]}
    for (label, name), f in flags.items():
        policies[f"Lazy unless flagged: {name} ({label})"] = np.where(f, regret_of["tabpfn_expected"], regret_of["lowest_premium"])
    policy_res = {p: regret_metrics(r, r < 0.5, household, counts) for p, r in policies.items()}
    primary = "Lazy unless flagged: TabPFN classifier v2 (at_risk_250)"
    base = policies["Lowest premium (always lazy)"]
    d = policies[primary] - base
    paired = {"primary_minus_lowest_premium_mean_regret": {
        "value": float(d.mean()),
        "ci95": [float(v) for v in np.percentile((counts @ np.bincount(household, weights=d, minlength=counts.shape[1]))
                                                 / (counts @ np.bincount(household, minlength=counts.shape[1])), [2.5, 97.5])]}}

    det = detection["at_risk_250"]["TabPFN classifier v2"]
    pr, lp = policy_res[primary], policy_res["Lowest premium (always lazy)"]
    orc = policy_res["Lazy unless flagged: Oracle flags (reference) (at_risk_250)"]
    headline = (
        f"Taking the lowest premium cost more than $250 extra in {detection['at_risk_250']['test_positive_rate']:.0%} of "
        f"person-menu cases. A TabPFN classifier trained on other people spotted them with AUC {det['auc']['value']:.2f} "
        f"(95% CI {det['auc']['ci95'][0]:.2f}-{det['auc']['ci95'][1]:.2f}); flagging its top 20% and giving them TabPFN's "
        f"expected-cost pick gave mean regret ${pr['mean_regret']['value']:,.0f} against ${lp['mean_regret']['value']:,.0f} "
        f"for always taking the lowest premium (paired difference ${paired['primary_minus_lowest_premium_mean_regret']['value']:+,.0f}, "
        f"95% CI {paired['primary_minus_lowest_premium_mean_regret']['ci95'][0]:+,.0f} to "
        f"{paired['primary_minus_lowest_premium_mean_regret']['ci95'][1]:+,.0f}). With perfect flags the same policy would "
        f"reach ${orc['mean_regret']['value']:,.0f}."
    )
    out = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "make eval-lazy",
        "preregistration": "DECISIONS 25 (commit 3d01df7, before scoring)",
        "headline": headline,
        "rows": {"train": int(len(lazy_tr)), "test": int(len(lazy_te)), "menus": sponsors},
        "lazy_regret_test": {"mean": float(lazy_te.mean()), "p90": float(np.percentile(lazy_te, 90)),
                             "share_over_250": float((lazy_te > 250).mean()), "share_over_500": float((lazy_te > 500).mean())},
        "thresholds_from_train_cv": thresholds,
        "train_cross_validation": cv,
        "detection_test": detection,
        "policies_test": policy_res,
        "paired": paired,
        "notes": ["TabPFN classifier v2 ran after the Step 3 reader run finished (GPU free).",
                  "Benchmark C approximations apply to the regret labels (scripts/decision_eval.py)."],
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(headline)
    print(f"wrote {OUT.relative_to(ROOT)} in {out['runtime_seconds']}s")


if __name__ == "__main__":
    main()
