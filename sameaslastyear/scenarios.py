"""Scenario engine shared by benchmark C (Step 4) and the lazy-safe check (Step 4b).

MEPS event files become per-person Event lists (approximations listed in
scripts/decision_eval.py), every plan is priced for every person, and the
strategies pick plans. TabPFN scenarios use real training people's events: for
each predicted spending percentile, the mean plan cost of the 5 training people
whose actual 2024 spending is nearest to it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sameaslastyear import costs
from sameaslastyear.costs import Event

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "meps"
SEED = 20261026
DONORS = 5
RISK_CAP = 1000.0
QUANTILES = ["q10", "q25", "q50", "q75", "q90"]
GENERIC_MAX = 50.0
PCP_DOCTORS = {6, 8, 10, 14, 21, 24}  # family, general, geriatrics, internal medicine, osteopathy, pediatrics
PCP_PEOPLE = {7, 8, 11, 6, 13, 14}  # midwife, nurse/NP, PA, MH counselor, psychologist, social worker
FILES = {2024: dict(ob="h254g", op="h254f", er="h254e", ip="h254d", rx="h254a"),
         2023: dict(ob="h248g", op="h248f", er="h248e", ip="h248d", rx="h248a")}


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
    """Annual cost (premiums + cost sharing - employer HSA) of every plan for every person."""
    return np.array([[costs.annual_cost(p, events[i]) for p in plans] for i in ids])


def donor_finder(train_spend: np.ndarray):
    order = np.argsort(train_spend, kind="stable")
    spend_sorted = train_spend[order]

    def donors(v: float) -> np.ndarray:
        i = np.searchsorted(spend_sorted, v)
        lo, hi = max(0, i - DONORS), min(len(order), i + DONORS)
        window = np.arange(lo, hi)
        nearest = window[np.argsort(np.abs(spend_sorted[window] - v), kind="stable")[:DONORS]]
        return order[nearest]

    return donors


def choices(plans: list, truth: np.ndarray, last_year: np.ndarray, donor_cost: np.ndarray,
            quantiles: np.ndarray, train_spend: np.ndarray) -> dict[str, np.ndarray]:
    """Plan index picked by each benchmark C strategy, per person (rows of truth)."""
    donors = donor_finder(train_spend)
    scen = np.stack([np.stack([donor_cost[donors(v)].mean(0) for v in quantiles[:, k]])
                     for k in range(quantiles.shape[1])])
    expected, p90 = scen.mean(0), scen[-1]
    premium = np.array([p.premium_monthly for p in plans])
    deductible = np.array([p.deductible for p in plans])
    n = truth.shape[0]
    capped = np.where(p90 <= p90.min(1, keepdims=True) + RISK_CAP, expected, np.inf)
    return {
        "lowest_premium": np.full(n, np.lexsort((deductible, premium))[0]),
        "lowest_deductible": np.full(n, np.lexsort((premium, deductible))[0]),
        "same_as_last_year": last_year.argmin(1),
        "tabpfn_expected": expected.argmin(1),
        "tabpfn_risk_capped": capped.argmin(1),
        "oracle": truth.argmin(1),
    }


def household_counts(households: np.ndarray, resamples: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n = households.max() + 1
    draws = rng.integers(0, n, size=(resamples, n))
    return np.stack([np.bincount(d, minlength=n) for d in draws])


def boot_mean(x: np.ndarray, household: np.ndarray, counts: np.ndarray) -> dict:
    n_h = counts.shape[1]
    sums = np.bincount(household, weights=x, minlength=n_h)
    boot = (counts @ sums) / (counts @ np.bincount(household, minlength=n_h))
    return {"value": float(x.mean()), "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}


def regret_metrics(regret: np.ndarray, hit: np.ndarray, household: np.ndarray, counts: np.ndarray) -> dict:
    """Mean regret, P90 regret and hit rate with household-bootstrap 95% CIs (rows = decisions)."""
    out = {"hit_rate": boot_mean(hit.astype(float), household, counts),
           "mean_regret": boot_mean(regret, household, counts)}
    hs = np.bincount(household, minlength=counts.shape[1])
    order = np.argsort(household, kind="stable")
    by_h = np.split(regret[order], np.cumsum(hs)[:-1])
    p90 = []
    for c in counts[:200]:  # P90 needs the resampled decisions themselves; 200 resamples keep it fast
        sample = np.concatenate([np.repeat(by_h[h], k) for h, k in enumerate(c) if k])
        p90.append(np.percentile(sample, 90))
    out["p90_regret"] = {"value": float(np.percentile(regret, 90)),
                         "ci95": [float(v) for v in np.percentile(p90, [2.5, 97.5])], "resamples": 200}
    return out
