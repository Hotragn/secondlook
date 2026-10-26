"""Build the Step 2 spend table from MEPS HC-257 and freeze the person split.

Population (DECISIONS 11): Panel 28 people present in both years (YEARIND = 1)
with a positive longitudinal weight, aged 18-64 at the end of 2023 and of 2024,
with any private coverage in both years (INSCOVY1 = INSCOVY2 = 1).

Features are year-1 (2023) values; the target is year-2 (2024) total spending.
Year-2 component totals, counts and out-of-pocket spending are kept for Step 4.
MEPS negative codes (inapplicable, refused, don't know) become missing.

Split: 25% test, households kept whole (all members on one side), stratified by
year-1 spending tier, fixed seed. The split file is written once; later runs
check that the population still matches it.

Run: uv run python -I scripts/spend_data.py
Writes data/derived/spend_panel28.parquet (local only) and eval/spend_split.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "meps" / "h257.parquet"
TABLE = ROOT / "data" / "derived" / "spend_panel28.parquet"
SPLIT = ROOT / "eval" / "spend_split.json"
SEED = 20261026
MONTHS = ["JA", "FE", "MA", "AP", "MY", "JU", "JL", "AU", "SE", "OC", "NO", "DE"]

# Year-1 priority-condition flags asked of adults (1 = yes, 2 = no).
CONDITIONS = {
    "high_bp": "HIBPDXY1", "coronary_hd": "CHDDXY1", "angina": "ANGIDXY1",
    "heart_attack": "MIDXY1", "other_heart": "OHRTDXY1", "stroke": "STRKDXY1",
    "emphysema": "EMPHDXY1", "high_chol": "CHOLDXY1", "cancer": "CANCERY1",
    "diabetes": "DIABDXY1_M18", "arthritis": "ARTHDXY1", "asthma": "ASTHDXY1",
}
UTILISATION = {
    "office_visits": "OBTOTVY1", "outpatient_visits": "OPTOTVY1",
    "er_visits": "ERTOTY1", "inpatient_stays": "IPDISY1", "rx_fills": "RXTOTY1",
}
COMPONENTS = {
    "office": "OBVEXP", "outpatient": "OPTEXP", "er": "ERTEXP",
    "inpatient": "IPTEXP", "rx": "RXEXP",
}
COUNTS_Y2 = {
    "office_visits_y2": "OBTOTVY2", "outpatient_visits_y2": "OPTOTVY2",
    "er_visits_y2": "ERTOTY2", "inpatient_stays_y2": "IPDISY2", "rx_fills_y2": "RXTOTY2",
}
FEATURES = (
    ["age", "female", "region_midwest", "region_south", "region_west",
     "self_rated_health", "self_rated_mental_health"]
    + list(CONDITIONS) + ["n_conditions"] + list(UTILISATION) + ["spend_y1"]
)


def valid(s: pd.Series) -> pd.Series:
    """MEPS reserved codes are negative; treat them as missing."""
    return s.where(s >= 0)


def latest_year1(d: pd.DataFrame, stem: str) -> pd.Series:
    """Round 2 is the last interview fully inside 2023; fall back to round 1."""
    return valid(d[f"{stem}2"]).fillna(valid(d[f"{stem}1"])).where(lambda s: s > 0)


def build() -> tuple[pd.DataFrame, list[dict]]:
    d = pd.read_parquet(SRC)
    funnel = [{"step": "HC-257 Panel 28 persons", "n": len(d)}]
    rules = [
        ("present in both 2023 and 2024 (YEARIND = 1)", d["YEARIND"].eq(1)),
        ("positive longitudinal weight (LONGWT > 0)", d["LONGWT"].gt(0)),
        ("aged 18-64 at the end of 2023 and of 2024 (AGEY1X, AGEY2X)",
         d["AGEY1X"].between(18, 64) & d["AGEY2X"].between(18, 64)),
        ("any private coverage in 2023 and in 2024 (INSCOVY1 = INSCOVY2 = 1)",
         d["INSCOVY1"].eq(1) & d["INSCOVY2"].eq(1)),
    ]
    keep = pd.Series(True, index=d.index)
    for label, rule in rules:
        keep &= rule
        funnel.append({"step": label, "n": int(keep.sum())})
    d = d[keep].reset_index(drop=True)

    t = pd.DataFrame({
        "DUPERSID": d["DUPERSID"], "DUID": d["DUID"].astype(str),
        "LONGWT": d["LONGWT"], "VARSTR": d["VARSTR"], "VARPSU": d["VARPSU"],
        "age": d["AGEY1X"], "female": d["SEX"].eq(2).astype(int),
        "region_midwest": d["REGIONY1"].eq(2).astype(int),
        "region_south": d["REGIONY1"].eq(3).astype(int),
        "region_west": d["REGIONY1"].eq(4).astype(int),
        "self_rated_health": latest_year1(d, "RTHLTH"),
        "self_rated_mental_health": latest_year1(d, "MNHLTH"),
    })
    for name, var in CONDITIONS.items():
        t[name] = valid(d[var]).map({1: 1.0, 2: 0.0})
    t["n_conditions"] = t[list(CONDITIONS)].fillna(0).sum(axis=1)
    for name, var in UTILISATION.items():
        t[name] = valid(d[var])
    t["spend_y1"] = valid(d["TOTEXPY1"])
    t["spend_y2"] = valid(d["TOTEXPY2"])
    for name, stem in COMPONENTS.items():
        t[f"{name}_y1"] = valid(d[f"{stem}Y1"])
        t[f"{name}_y2"] = valid(d[f"{stem}Y2"])
    for name, var in COUNTS_Y2.items():
        t[name] = valid(d[var])
    t["oop_y1"] = valid(d["TOTSLFY1"])
    t["oop_y2"] = valid(d["TOTSLFY2"])
    t["esi_all_24_months"] = np.logical_and.reduce(
        [d[f"PEG{m}Y{y}"].eq(1) for y in (1, 2) for m in MONTHS]
    ).astype(int)
    t["age_band"] = pd.cut(t["age"], [17, 34, 49, 64], labels=["18-34", "35-49", "50-64"]).astype(str)
    return t, funnel


def add_tiers(t: pd.DataFrame) -> dict:
    positive = t.loc[t["spend_y1"] > 0, "spend_y1"]
    low_max, mid_max = (float(x) for x in positive.quantile([1 / 3, 2 / 3]))
    t["tier"] = np.select(
        [t["spend_y1"].eq(0), t["spend_y1"] <= low_max, t["spend_y1"] <= mid_max],
        ["zero", "low", "mid"], default="high",
    )
    return {"zero": "spend_y1 = 0", "low": f"0 < spend_y1 <= {low_max:.0f}",
            "mid": f"{low_max:.0f} < spend_y1 <= {mid_max:.0f}", "high": f"spend_y1 > {mid_max:.0f}"}


def main() -> None:
    t, funnel = build()
    assert t["spend_y1"].notna().all() and t["spend_y2"].notna().all()
    tiers = add_tiers(t)

    if SPLIT.exists():
        split = json.loads(SPLIT.read_text())
        frozen = set(split["train_ids"]) | set(split["test_ids"])
        assert frozen == set(t["DUPERSID"]), "population no longer matches the frozen split"
        print(f"split exists; population matches ({len(frozen)} people)")
    else:
        folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        _, test_idx = next(folds.split(t, t["tier"], groups=t["DUID"]))
        is_test = t.index.isin(test_idx)
        split = {
            "rule": "25% test by household (StratifiedGroupKFold, 4 folds, fold 0), "
                    "stratified by year-1 spending tier",
            "seed": SEED,
            "population_funnel": funnel,
            "tiers": tiers,
            "counts": {
                "train": int((~is_test).sum()), "test": int(is_test.sum()),
                "train_households": int(t.loc[~is_test, "DUID"].nunique()),
                "test_households": int(t.loc[is_test, "DUID"].nunique()),
            },
            "test_ids": sorted(t.loc[is_test, "DUPERSID"]),
            "train_ids": sorted(t.loc[~is_test, "DUPERSID"]),
        }
        SPLIT.parent.mkdir(parents=True, exist_ok=True)
        SPLIT.write_text(json.dumps(split, indent=1) + "\n")
        print(f"wrote {SPLIT.relative_to(ROOT)}")

    t["split"] = np.where(t["DUPERSID"].isin(split["test_ids"]), "test", "train")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(TABLE, index=False)
    for f in funnel:
        print(f"  {f['n']:>6,}  {f['step']}")
    print(f"tiers: {tiers}")
    print(f"split counts: {split['counts']}")
    print(t.groupby(["split", "tier"]).size().unstack(fill_value=0).to_string())
    print(f"wrote {TABLE.relative_to(ROOT)} ({len(t)} rows, {len(FEATURES)} features)")
    print(f"split sha256 {hashlib.sha256(SPLIT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
