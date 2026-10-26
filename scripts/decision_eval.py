"""Step 4, benchmark C: which plan-choosing strategy saves people money?

For each of the 758 held-out Panel 28 people (Step 2 test split) and each real 2027
menu (Washington PEBB, Oregon PEBB, Tennessee state; self-only coverage), the true
cost of every plan is its employee premium plus what the person's actual 2024 MEPS
events cost under that plan's rules, less any employer HSA deposit. Strategies
pick a plan; regret is the picked plan's true cost minus the cheapest plan's.

Strategies
  lowest premium       cheapest employee premium (ties: lower deductible)
  lowest deductible    smallest deductible (ties: lower premium)
  same as last year    the plan that would have been cheapest for the person's 2023 events
  TabPFN expected      lowest mean cost over scenarios at the person's predicted 10th, 25th,
                       50th, 75th and 90th spending percentiles (TabPFN v2, Step 2)
  TabPFN risk-capped   lowest expected cost among plans whose 90th-percentile scenario cost
                       is within $1,000 of the lowest 90th-percentile scenario cost
  oracle               the truly cheapest plan (the reference)

A scenario at a predicted spending level is the average plan cost of the 5 training
people whose actual 2024 spending is nearest to it, run through each plan's rules
(their real events, not a single total).

Approximations, also written to the JSON:
  - Event costs are MEPS total payments from all sources, used as the allowed amount.
  - Office visits: general checkups, immunizations and well-child visits are preventive
    ($0); family, general and internal medicine, pediatrics, geriatrics and osteopathy
    doctors, nurse practitioners, physician assistants, midwives and mental-health visits
    use the primary-care row; other providers use the specialist row. Vision exams, laser
    eye surgery and dentists are excluded (not medical-plan costs).
  - Outpatient department visits with surgery use the outpatient surgery facility row;
    other outpatient visits use the specialist row.
  - Each event takes one row's cost sharing on its full payment (no separate imaging,
    lab or physician lines); ER visits that lead to admission keep coinsurance but
    waive the ER copay.
  - Prescriptions: fills costing up to $50 use the generic row, others the preferred
    brand row (MEPS has no generic/brand flag).
  - Event months order the deductible; a missing month counts as June.
  - Dental, home health, vision and other medical spending are not modelled.

Run: uv run python -I scripts/decision_eval.py   (or: make eval-decision)
Writes eval/results/step4_decision.json
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sameaslastyear import menus
from sameaslastyear.scenarios import (
    DONORS, GENERIC_MAX, QUANTILES, RISK_CAP, choices, cost_matrix, household_counts, load_events, regret_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "derived" / "spend_panel28.parquet"
PREDICTIONS = ROOT / "data" / "derived" / "step2_test_predictions.parquet"
SPLIT = ROOT / "eval" / "spend_split.json"
SPLIT_SHA256 = "d9577f32d9f4a365f1e4f5cd2f88f0ad796dfe87423a4d0862d1076edc4039aa"
OUT = ROOT / "eval" / "results" / "step4_decision.json"
STRATEGIES = ["lowest_premium", "lowest_deductible", "same_as_last_year", "tabpfn_expected",
              "tabpfn_risk_capped", "oracle"]


def main() -> None:
    started = time.perf_counter()
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == SPLIT_SHA256, "spend split changed"
    t = pd.read_parquet(TABLE)
    test = t[t["split"] == "test"].reset_index(drop=True)
    train = t[t["split"] == "train"].reset_index(drop=True)
    pred = pd.read_parquet(PREDICTIONS).set_index("DUPERSID").loc[test["DUPERSID"]]
    q = pred[[f"TabPFN v2 {k}" for k in QUANTILES]].to_numpy()

    ev24 = load_events(2024, set(t["DUPERSID"]))
    ev23 = load_events(2023, set(test["DUPERSID"]))
    modelled = np.array([sum(e.allowed for e in ev24[i]) for i in test["DUPERSID"]])
    coverage = float(modelled.sum() / test["spend_y2"].sum())

    household = pd.factorize(test["DUID"])[0]
    counts = household_counts(household)
    plans_by_sponsor = menus.load()
    results, pooled = {}, {s: {"regret": [], "hit": []} for s in STRATEGIES}
    oracle_share = {}
    for sponsor, plans in plans_by_sponsor.items():
        truth = cost_matrix(plans, ev24, list(test["DUPERSID"]))
        last_year = cost_matrix(plans, ev23, list(test["DUPERSID"]))
        donor_cost = cost_matrix(plans, ev24, list(train["DUPERSID"]))
        choice = choices(plans, truth, last_year, donor_cost, q, train["spend_y2"].to_numpy())
        n = len(test)
        best = truth.min(1)
        res = {"plans": [{"plan_id": p.plan_id, "name": p.name, "premium_monthly": p.premium_monthly,
                          "deductible": p.deductible, "oop": p.oop, "employer_hsa": p.employer_hsa} for p in plans]}
        for s, c in choice.items():
            regret = truth[np.arange(n), c] - best
            hit = regret < 0.5
            pooled[s]["regret"].append(regret)
            pooled[s]["hit"].append(hit)
            res[s] = {**regret_metrics(regret, hit, household, counts),
                      "plans_chosen": {plans[k].name: int((c == k).sum()) for k in np.unique(c)}}
        res["mean_true_cost_of_cheapest_plan"] = float(best.mean())
        oracle_share[sponsor] = {plans[k].name: int((choice["oracle"] == k).sum()) for k in np.unique(choice["oracle"])}
        results[sponsor] = res
        print(f"{sponsor}: " + ", ".join(f"{s} hit {res[s]['hit_rate']['value']:.0%} regret ${res[s]['mean_regret']['value']:,.0f}"
                                         for s in STRATEGIES), flush=True)

    # Pooled: each person faces all three menus; resample households, keeping their three decisions.
    pooled_res = {}
    h3, c3 = np.tile(household, 3), counts
    for s in STRATEGIES:
        pooled_res[s] = regret_metrics(np.concatenate(pooled[s]["regret"]), np.concatenate(pooled[s]["hit"]), h3, c3)
    # Paired regret differences against TabPFN expected, and results by 2023 spending tier.
    tab = np.concatenate(pooled["tabpfn_expected"]["regret"])
    n_h = c3.shape[1]
    hs = np.bincount(h3, minlength=n_h)
    paired = {}
    for s in STRATEGIES:
        if s in ("tabpfn_expected", "oracle"):
            continue
        d = np.concatenate(pooled[s]["regret"]) - tab
        boot = (c3 @ np.bincount(h3, weights=d, minlength=n_h)) / (c3 @ hs)
        paired[s] = {"mean_regret_minus_tabpfn": float(d.mean()),
                     "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}
    tiers = np.tile(test["tier"].to_numpy(), 3)
    by_tier = {}
    for tier in ["zero", "low", "mid", "high"]:
        m = tiers == tier
        by_tier[tier] = {"decisions": int(m.sum()), **{
            s: {"hit_rate": float(np.concatenate(pooled[s]["hit"])[m].mean()),
                "mean_regret": float(np.concatenate(pooled[s]["regret"])[m].mean())} for s in STRATEGIES}}
    p = pooled_res
    headline = (
        f"Across {len(test)} held-out MEPS adults and {len(results)} real 2027 employer menus, choosing the plan with the "
        f"lowest TabPFN-expected cost picked the truly cheapest plan {p['tabpfn_expected']['hit_rate']['value']:.0%} of the time, "
        f"against {p['same_as_last_year']['hit_rate']['value']:.0%} for picking what would have been best last year and "
        f"{p['lowest_premium']['hit_rate']['value']:.0%} for the lowest premium; mean regret was "
        f"${p['tabpfn_expected']['mean_regret']['value']:,.0f} versus ${p['same_as_last_year']['mean_regret']['value']:,.0f} "
        f"and ${p['lowest_premium']['mean_regret']['value']:,.0f}."
    )
    out = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "make eval-decision",
        "headline": headline,
        "people": len(test), "households": int(test["DUID"].nunique()),
        "coverage": "self-only; every plan in a menu is assumed available to everyone",
        "menus": "data/menus/menus_2027.json (quotes verified on load by sameaslastyear/menus.py)",
        "tabpfn": "TabPFN v2 quantiles from Step 2 (eval/results/step2_spend.json), trained on the 2,275 training people",
        "scenario": f"mean plan cost of the {DONORS} training people with the nearest actual 2024 spending to each predicted "
                    "percentile; expected cost = mean over the 10th/25th/50th/75th/90th percentile scenarios (an approximation "
                    "of the predicted distribution by five points)",
        "risk_cap": f"plans whose 90th-percentile scenario cost is within ${RISK_CAP:,.0f} of the lowest",
        "modelled_share_of_2024_spending": coverage,
        "approximations": [
            "Event costs are MEPS total payments from all sources, used as the allowed amount.",
            "Office visits are mapped to the preventive, primary care or specialist row by visit type and provider; vision "
            "exams, laser eye surgery and dentist visits are excluded.",
            "Outpatient visits with surgery use the outpatient surgery facility row, others the specialist row.",
            "Each event takes one row's cost sharing on its full payment; ER visits leading to admission waive the ER copay.",
            f"Prescription fills costing up to ${GENERIC_MAX:.0f} use the generic row, others the preferred brand row.",
            "A missing event month counts as June when ordering the deductible.",
            "Dental, home health, vision and other medical spending are not modelled (dental is not a medical-plan cost).",
            "Employer HSA deposits are counted as cash; employee HSA tax savings are not included.",
        ],
        "oracle_plan_counts": oracle_share,
        "by_menu": results,
        "pooled": pooled_res,
        "paired_vs_tabpfn_expected": paired,
        "pooled_by_2023_spending_tier": by_tier,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(headline)
    print(f"modelled share of 2024 spending: {coverage:.1%}; wrote {OUT.relative_to(ROOT)} in {out['runtime_seconds']}s")


if __name__ == "__main__":
    main()
