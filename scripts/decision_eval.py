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

from sameaslastyear import costs, menus
from sameaslastyear.costs import Event

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "meps"
TABLE = ROOT / "data" / "derived" / "spend_panel28.parquet"
PREDICTIONS = ROOT / "data" / "derived" / "step2_test_predictions.parquet"
SPLIT = ROOT / "eval" / "spend_split.json"
SPLIT_SHA256 = "d9577f32d9f4a365f1e4f5cd2f88f0ad796dfe87423a4d0862d1076edc4039aa"
OUT = ROOT / "eval" / "results" / "step4_decision.json"
SEED = 20261026
BOOTSTRAP = 1000
DONORS = 5
RISK_CAP = 1000.0
QUANTILES = ["q10", "q25", "q50", "q75", "q90"]
GENERIC_MAX = 50.0
PCP_DOCTORS = {6, 8, 10, 14, 21, 24}  # family, general, geriatrics, internal medicine, osteopathy, pediatrics
PCP_PEOPLE = {7, 8, 11, 6, 13, 14}  # midwife, nurse/NP, PA, MH counselor, psychologist, social worker
FILES = {2024: dict(ob="h254g", op="h254f", er="h254e", ip="h254d", rx="h254a"),
         2023: dict(ob="h248g", op="h248f", er="h248e", ip="h248d", rx="h248a")}
STRATEGIES = ["lowest_premium", "lowest_deductible", "same_as_last_year", "tabpfn_expected",
              "tabpfn_risk_capped", "oracle"]


def month(s: pd.Series) -> pd.Series:
    return s.where(s.between(1, 12), 6).astype(int)


def office_category(df: pd.DataFrame) -> pd.Series:
    cat = pd.Series("specialist", index=df.index)
    cat[df["SEEDOC_M18"].eq(1) & df["DRSPLTY_M18"].isin(PCP_DOCTORS)] = "primary_care"
    cat[df["SEEDOC_M18"].ne(1) & df["MEDPTYPE_M18"].isin(PCP_PEOPLE)] = "primary_care"
    cat[df["VSTCTGRY"].eq(4)] = "primary_care"
    cat[df["VSTCTGRY"].isin([1, 6, 9])] = "preventive"
    cat[df["VSTCTGRY"].isin([7, 10]) | df["MEDPTYPE_M18"].isin([3, 9])] = "excluded"
    return cat


def load_events(year: int, people: set[str]) -> dict[str, list[Event]]:
    f, yy = FILES[year], str(year)[2:]
    frames = []
    ob = pd.read_parquet(RAW / f"{f['ob']}.parquet", columns=["DUPERSID", "OBDATEMM", "VSTCTGRY", "SEEDOC_M18",
                                                              "DRSPLTY_M18", "MEDPTYPE_M18", f"OBXP{yy}X"])
    frames.append(pd.DataFrame({"id": ob["DUPERSID"], "cat": office_category(ob), "allowed": ob[f"OBXP{yy}X"],
                                "month": month(ob["OBDATEMM"]), "nights": 0}))
    op = pd.read_parquet(RAW / f"{f['op']}.parquet", columns=["DUPERSID", "OPDATEMM", "SURGPROC", f"OPXP{yy}X"])
    frames.append(pd.DataFrame({"id": op["DUPERSID"], "allowed": op[f"OPXP{yy}X"], "month": month(op["OPDATEMM"]),
                                "cat": np.where(op["SURGPROC"].eq(1), "outpatient_facility", "specialist"), "nights": 0}))
    er = pd.read_parquet(RAW / f"{f['er']}.parquet", columns=["DUPERSID", "ERDATEMM", "ERHEVIDX", f"ERXP{yy}X"])
    admitted = er["ERHEVIDX"].astype(str).str.strip().ne("-1")
    frames.append(pd.DataFrame({"id": er["DUPERSID"], "allowed": er[f"ERXP{yy}X"], "month": month(er["ERDATEMM"]),
                                "cat": np.where(admitted, "emergency_room_admitted", "emergency_room"), "nights": 0}))
    ip = pd.read_parquet(RAW / f"{f['ip']}.parquet", columns=["DUPERSID", "IPBEGMM", "NUMNIGHX", f"IPXP{yy}X"])
    frames.append(pd.DataFrame({"id": ip["DUPERSID"], "cat": "inpatient_facility", "allowed": ip[f"IPXP{yy}X"],
                                "month": month(ip["IPBEGMM"]), "nights": ip["NUMNIGHX"].clip(lower=1).astype(int)}))
    rx = pd.read_parquet(RAW / f"{f['rx']}.parquet", columns=["DUPERSID", "RXBEGMM", f"RXXP{yy}X"])
    frames.append(pd.DataFrame({"id": rx["DUPERSID"], "allowed": rx[f"RXXP{yy}X"], "month": month(rx["RXBEGMM"]),
                                "cat": np.where(rx[f"RXXP{yy}X"] <= GENERIC_MAX, "generic_drugs", "preferred_brand_drugs"),
                                "nights": 0}))
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[ev["id"].isin(people) & ev["allowed"].gt(0) & ev["cat"].ne("excluded")]
    out = {p: [] for p in people}
    for row in ev.itertuples(index=False):
        out[row.id].append(Event(row.cat, float(row.allowed), int(row.month), int(row.nights)))
    return out


def cost_matrix(plans: list, events: dict, ids: list[str]) -> np.ndarray:
    return np.array([[costs.annual_cost(p, events[i]) for p in plans] for i in ids])


def household_counts(households: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n = households.max() + 1
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    return np.stack([np.bincount(d, minlength=n) for d in draws])


def metrics(regret: np.ndarray, hit: np.ndarray, household: np.ndarray, counts: np.ndarray) -> dict:
    """Mean regret, P90 regret and hit rate with household-bootstrap 95% CIs (rows = people)."""
    n_h = counts.shape[1]
    hs = np.bincount(household, minlength=n_h)
    out = {}
    for name, x in (("hit_rate", hit.astype(float)), ("mean_regret", regret)):
        sums = np.bincount(household, weights=x, minlength=n_h)
        boot = (counts @ sums) / (counts @ hs)
        out[name] = {"value": float(x.mean()), "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}
    order = np.argsort(household, kind="stable")
    by_h = np.split(regret[order], np.cumsum(hs)[:-1])
    p90 = []
    for c in counts[:200]:  # P90 needs the resampled people themselves; 200 resamples keep it fast
        sample = np.concatenate([np.repeat(by_h[h], k) for h, k in enumerate(c) if k])
        p90.append(np.percentile(sample, 90))
    out["p90_regret"] = {"value": float(np.percentile(regret, 90)),
                         "ci95": [float(v) for v in np.percentile(p90, [2.5, 97.5])], "resamples": 200}
    return out


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

    # Donors: for each predicted percentile, the 5 training people with the nearest actual 2024 spending.
    order = np.argsort(train["spend_y2"].to_numpy(), kind="stable")
    spend_sorted = train["spend_y2"].to_numpy()[order]

    def donors(v: float) -> np.ndarray:
        i = np.searchsorted(spend_sorted, v)
        lo, hi = max(0, i - DONORS), min(len(order), i + DONORS)
        window = np.arange(lo, hi)
        nearest = window[np.argsort(np.abs(spend_sorted[window] - v), kind="stable")[:DONORS]]
        return order[nearest]

    household = pd.factorize(test["DUID"])[0]
    counts = household_counts(household)
    plans_by_sponsor = menus.load()
    results, pooled = {}, {s: {"regret": [], "hit": []} for s in STRATEGIES}
    oracle_share = {}
    for sponsor, plans in plans_by_sponsor.items():
        truth = cost_matrix(plans, ev24, list(test["DUPERSID"]))
        last_year = cost_matrix(plans, ev23, list(test["DUPERSID"]))
        donor_cost = cost_matrix(plans, ev24, list(train["DUPERSID"]))
        scen = np.stack([np.stack([donor_cost[donors(v)].mean(0) for v in q[:, k]]) for k in range(len(QUANTILES))])
        expected, p90 = scen.mean(0), scen[-1]
        premium = np.array([p.premium_monthly for p in plans])
        deductible = np.array([p.deductible for p in plans])
        n = len(test)
        capped = np.where(p90 <= p90.min(1, keepdims=True) + RISK_CAP, expected, np.inf)
        choice = {
            "lowest_premium": np.full(n, np.lexsort((deductible, premium))[0]),
            "lowest_deductible": np.full(n, np.lexsort((premium, deductible))[0]),
            "same_as_last_year": last_year.argmin(1),
            "tabpfn_expected": expected.argmin(1),
            "tabpfn_risk_capped": capped.argmin(1),
            "oracle": truth.argmin(1),
        }
        best = truth.min(1)
        res = {"plans": [{"plan_id": p.plan_id, "name": p.name, "premium_monthly": p.premium_monthly,
                          "deductible": p.deductible, "oop": p.oop, "employer_hsa": p.employer_hsa} for p in plans]}
        for s, c in choice.items():
            regret = truth[np.arange(n), c] - best
            hit = regret < 0.5
            pooled[s]["regret"].append(regret)
            pooled[s]["hit"].append(hit)
            res[s] = {**metrics(regret, hit, household, counts),
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
        pooled_res[s] = metrics(np.concatenate(pooled[s]["regret"]), np.concatenate(pooled[s]["hit"]), h3, c3)
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
