"""Build docs/data/secondlook.json, everything the static site shows (Step 5).

Nothing is typed in by hand except question wording:
- **Menus:** the three real 2027 menus (sameaslastyear.menus). Each plan fact carries its source line and page.
  The line is the reader's quote (e4b first, then the other two models) when the parsed value equals the gold value.
  Five facts the reader got wrong use the correct line from the PDF instead, marked "transcribed".
  Every line is checked against the PDF text.
- **People:** three real MEPS test-set people (DECISIONS 28).
  - Typical year: each plan priced on the real 2024 events of the 5 training people whose 2024 spending is nearest
    this person's predicted median (Step 2 TabPFN quantiles).
  - Bad year: the same at the predicted 90th percentile.
  - Flag: the Step 4b TabPFN classifier at its training threshold.
  - Also what 2024 actually cost them, and the plan their 2023 care pointed to.
- **Five answers:** every combination (3^5 = 243) for each menu, from a five-answer TabPFN regressor and classifier
  trained on the training people only. Each answer maps to the median value among training people with that answer.
- **How sure:** numbers copied from eval/results/*.json.

Run: uv run python -I scripts/build_app_data.py   (or: make app-data)
"""

import csv
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from sameaslastyear import menus as menu_loader
from sameaslastyear import reader
from sameaslastyear.scenarios import cost_matrix, donor_finder, load_events
from sameaslastyear.spend import FEATURES, QUANTILES, SEED, feature_matrix, prepare_tabpfn_auth

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "secondlook.json"
RESULTS = ROOT / "eval" / "results"
PEOPLE = {
    "2810297103": "Healthy single adult",
    "2810385102": "Ongoing prescriptions",
    "2810035101": "High spender",
}
MODELS = ["gemma4_e4b-it-qat", "gemma4_26b-a4b-it-qat", "gemma4_e2b-it-qat"]
FACTS = [("deductible_individual", "Deductible"), ("oop_individual", "Out-of-pocket limit"),
         ("primary_care", "Primary care visit"), ("specialist", "Specialist visit"),
         ("emergency_room", "Emergency room"), ("inpatient_facility", "Hospital stay"), ("generic_drugs", "Generic drugs")]
# Lines for the facts the reader got wrong (Step 3 failure log); verified against the PDF below.
TRANSCRIBED = {
    ("wa-pebb-ump-classic-2027", "emergency_room"): "15% coinsurance after $75 copay / visit",
    ("wa-pebb-ump-select-2027", "emergency_room"): "20% coinsurance after $75 copay / visit",
    ("tn-premier-ppo-2027", "oop_individual"): "$2,400/ $3,600/ $7,200 employee only",
    ("tn-standard-ppo-2027", "oop_individual"): "$2,400/ $4,400/ $8,800 employee only",
    ("tn-cdhp-2027", "oop_individual"): "$2,400/ $2,800/ $5,600 employee only",
}
TN_OOP_NOTE = "Middle figure: the in-network limit for everything except specialty drugs, which have their own $2,400 limit."
QUESTIONS = [
    ("age", "How old are you?", ["18 to 34", "35 to 49", "50 to 64"]),
    ("health", "How is your health in general?", ["Excellent or very good", "Good", "Fair or poor"]),
    ("conditions", "Ongoing conditions you are treated for (diabetes, high blood pressure, asthma and so on)",
     ["None", "One", "Two or more"]),
    ("prescriptions", "Prescriptions you filled last year", ["None", "A few (1 to 11)", "Monthly or more (12+)"]),
    ("urgent", "Last year, did you have", ["Neither", "An ER visit", "A hospital stay"]),
]
FIVE = ["age", "self_rated_health", "n_conditions", "rx_fills", "er_visits", "inpatient_stays"]


def manifest() -> dict:
    with (ROOT / "data" / "sbc" / "manifest.csv").open() as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def source(doc: str, quote: str, rows: dict, provenance: str, note: str | None = None) -> dict:
    """Page, the quote and a little text either side, from the PDF's own extracted text."""
    pages = [reader.normalize(p) for p in reader.load_pages(ROOT / rows[doc]["path"], max_pages=200)]
    q = reader.normalize(quote)
    for i, text in enumerate(pages):
        at = text.find(q)
        if at >= 0:
            before = text[max(0, at - 110): at]
            after = text[at + len(q): at + len(q) + 110]
            return {"doc": doc, "page": i + 1, "quote": q, "before": before.split(" ", 1)[-1] if len(before) == 110 else before,
                    "after": after.rsplit(" ", 1)[0] if len(after) == 110 else after, "provenance": provenance,
                    "url": rows[doc]["url"], "title": f"{rows[doc]['plan']} ({rows[doc]['sponsor']}, {rows[doc]['year']})",
                    **({"note": note} if note else {})}
    raise ValueError(f"source line not found in {doc}: {quote!r}")


def cost_text(v: dict) -> str:
    if v.get("not_covered"):
        return "Not covered"
    parts = []
    if v.get("copay") is not None:
        parts.append("No charge" if v["copay"] == 0 and v.get("coinsurance") is None else f"${v['copay']:,.0f}")
    if v.get("coinsurance") is not None:
        parts.append(f"{v['coinsurance'] * 100:g}%")
    return " then ".join(parts)


def same(pred, gold) -> bool:
    if pred is None:
        return False
    if isinstance(gold, dict):
        return all((pred.get(k) is None) == (gold.get(k) is None)
                   and (pred.get(k) is None or abs(pred[k] - gold[k]) < 0.005) for k in ("copay", "coinsurance"))
    return abs(float(pred) - float(gold)) < 0.5


def plan_facts(plan_id: str, spec: dict, plan, gold: dict, rows: dict) -> list[dict]:
    checked = {m: reader.check_fields(ROOT / rows[plan_id]["path"],
                                      json.loads((RESULTS / "step3_raw" / "heldout" / m / f"{plan_id}.json").read_text()))
               for m in MODELS}
    p = spec["premium"]
    prem = source(p["doc"], p["quote"], rows, "rate sheet")
    if "share" in p:
        prem["second"] = source(p["share"]["doc"], p["share"]["quote"], rows, "enrollment guide")
    facts = [{"key": "premium", "label": "Monthly premium", "value": f"${plan.premium_monthly:,.2f}", "source": prem}]
    for key, label in FACTS:
        g = gold[key]
        value = f"${g:,.0f}" if not isinstance(g, dict) else cost_text(g)
        hit = next((m for m in MODELS if checked[m][key]["status"] == "verified" and same(checked[m][key]["value"], g)), None)
        if hit:
            src = source(plan_id, checked[hit][key]["quote"], rows, f"read by {hit.replace('_', ':', 1)}")
        else:
            src = source(plan_id, TRANSCRIBED[(plan_id, key)], rows, "transcribed (the reader got this one wrong)",
                         TN_OOP_NOTE if plan_id.startswith("tn-") else None)
        facts.append({"key": key, "label": label, "value": value, "source": src})
    if plan.hsa_eligible:
        h = spec["hsa"]
        facts.append({"key": "hsa", "label": "Employer HSA deposit", "value": f"${plan.employer_hsa:,.2f}",
                      "source": source(h["doc"], h["quote"], rows, "plan document")})
    return facts


def tabpfn_regressor():
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, random_state=SEED)


def tabpfn_classifier():
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, random_state=SEED)


def five_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[FIVE].astype(float).copy()
    for c in ("rx_fills", "er_visits", "inpatient_stays"):
        X[c] = np.log1p(X[c])
    return X.to_numpy()


def answer_values(train: pd.DataFrame) -> dict:
    """Each answer's feature value = the median among training people who would give that answer."""
    med = lambda s: float(np.median(s)) if len(s) else 0.0
    t = train
    return {
        "age": [med(t.age[t.age <= 34]), med(t.age[t.age.between(35, 49)]), med(t.age[t.age >= 50])],
        "health": [med(t.self_rated_health[t.self_rated_health <= 2]), 3.0, med(t.self_rated_health[t.self_rated_health >= 4])],
        "conditions": [0.0, 1.0, med(t.n_conditions[t.n_conditions >= 2])],
        "prescriptions": [0.0, med(t.rx_fills[t.rx_fills.between(1, 11)]), med(t.rx_fills[t.rx_fills >= 12])],
        "urgent": [(0.0, 0.0), (med(t.er_visits[(t.er_visits >= 1) & (t.inpatient_stays == 0)]), 0.0),
                   (med(t.er_visits[t.inpatient_stays >= 1]), med(t.inpatient_stays[t.inpatient_stays >= 1]))],
    }


def main() -> None:
    warnings.simplefilter("ignore")
    rows = manifest()
    gold = json.loads((ROOT / "eval" / "gold" / "heldout_gold.json").read_text())["documents"]
    spec = json.loads((ROOT / "data" / "menus" / "menus_2027.json").read_text())["sponsors"]
    plans_by = menu_loader.load()
    t = pd.read_parquet(ROOT / "data" / "derived" / "spend_panel28.parquet")
    train = t[t.split == "train"].reset_index(drop=True)
    test = t[t.split == "test"].reset_index(drop=True)
    pred = pd.read_parquet(ROOT / "data" / "derived" / "step2_test_predictions.parquet").set_index("DUPERSID")
    people = test.set_index("DUPERSID").loc[list(PEOPLE)]
    ev24 = load_events(2024, set(train.DUPERSID) | set(PEOPLE))
    ev23 = load_events(2023, set(PEOPLE))
    donors = donor_finder(train.spend_y2.to_numpy())
    lazy = json.loads((RESULTS / "step4b_lazy_safe.json").read_text())
    threshold = lazy["thresholds_from_train_cv"]["at_risk_250"]["TabPFN classifier v2"]
    prepare_tabpfn_auth()

    sponsors = list(plans_by)
    k = len(sponsors)
    menus, donor_cost, lazy_tr = {}, {}, []
    for s in sponsors:
        plans = plans_by[s]
        donor_cost[s] = cost_matrix(plans, ev24, list(train.DUPERSID))
        lp = min(range(len(plans)), key=lambda j: (plans[j].premium_monthly, plans[j].deductible))
        lazy_tr.append(donor_cost[s][:, lp] - donor_cost[s].min(1))
        menus[s] = {"name": spec[s]["name"], "lowest_premium": plans[lp].plan_id, "plans": [
            {"id": p.plan_id, "name": p.name, "premium_monthly": p.premium_monthly, "deductible": p.deductible,
             "hsa": p.hsa_eligible, "facts": plan_facts(p.plan_id, spec[s]["plans"][p.plan_id], p, gold[p.plan_id]["fields"], rows)}
            for p in plans]}
        print(f"menu {s}: {len(plans)} plans, facts with sources", flush=True)
    y_tr = (np.concatenate(lazy_tr) > 250).astype(int)
    onehot = lambda n, m: np.tile(np.eye(k)[m], (n, 1))

    def scenario(s: str, v: float) -> list[float]:
        return [round(float(x), 2) for x in donor_cost[s][donors(v)].mean(0)]

    # People: the full Step 2 / 4b models.
    Xtr = np.vstack([np.hstack([feature_matrix(train), onehot(len(train), m)]) for m in range(k)])
    clf = tabpfn_classifier().fit(Xtr, y_tr)
    out_people = []
    for pid, label in PEOPLE.items():
        row = people.loc[[pid]].reset_index()
        q = pred.loc[pid, [f"TabPFN v2 {x}" for x in ("q10", "q25", "q50", "q75", "q90")]].to_numpy()
        per_menu = {}
        for m, s in enumerate(sponsors):
            score = float(clf.predict_proba(np.hstack([feature_matrix(row), onehot(1, m)]))[0, 1])
            actual = cost_matrix(plans_by[s], ev24, [pid])[0]
            last = cost_matrix(plans_by[s], ev23, [pid])[0]
            per_menu[s] = {"typical": scenario(s, q[2]), "bad": scenario(s, q[4]), "flag": score > threshold,
                           "risk_score": round(score, 3), "actual_2024": [round(float(x), 2) for x in actual],
                           "last_year_pick": plans_by[s][int(last.argmin())].plan_id}
        r = people.loc[pid]
        out_people.append({"id": pid, "label": label, "age": int(r.age), "sex": "female" if r.female else "male",
                           "health": int(r.self_rated_health), "conditions": int(r.n_conditions),
                           "rx_fills_2023": int(r.rx_fills), "spend_2023": round(float(r.spend_y1)),
                           "spend_2024": round(float(r.spend_y2)), "predicted_typical_spend": round(float(q[2])),
                           "predicted_bad_spend": round(float(q[4])), "menus": per_menu})
        print(f"person {pid}: flags {[per_menu[s]['flag'] for s in sponsors]}", flush=True)

    # Five answers: models trained on the training people only.
    values = answer_values(train)
    reg = tabpfn_regressor().fit(five_matrix(train), np.log1p(train.spend_y2.to_numpy()))
    X5 = np.vstack([np.hstack([five_matrix(train), onehot(len(train), m)]) for m in range(k)])
    groups = np.tile(train.DUID.to_numpy(), k)
    oof = np.zeros(len(y_tr))
    for tr, va in GroupKFold(n_splits=5).split(X5, y_tr, groups):
        oof[va] = tabpfn_classifier().fit(X5[tr], y_tr[tr]).predict_proba(X5[va])[:, 1]
    five_threshold = float(np.quantile(oof, 0.8))
    clf5 = tabpfn_classifier().fit(X5, y_tr)
    # The five-answer flagger is a new model, so it gets its own one-time test AUC.
    lazy_te = []
    for s in sponsors:
        c = cost_matrix(plans_by[s], load_events(2024, set(test.DUPERSID)), list(test.DUPERSID))
        lp = [p.plan_id for p in plans_by[s]].index(menus[s]["lowest_premium"])
        lazy_te.append(c[:, lp] - c.min(1))
    y_te = (np.concatenate(lazy_te) > 250).astype(int)
    X5_te = np.vstack([np.hstack([five_matrix(test), onehot(len(test), m)]) for m in range(k)])
    five_auc = float(roc_auc_score(y_te, clf5.predict_proba(X5_te)[:, 1]))

    combos = list(itertools.product(range(3), repeat=5))
    grid_rows = []
    for c in combos:
        er, ip = values["urgent"][c[4]]
        grid_rows.append({"age": values["age"][c[0]], "self_rated_health": values["health"][c[1]],
                          "n_conditions": values["conditions"][c[2]], "rx_fills": values["prescriptions"][c[3]],
                          "er_visits": er, "inpatient_stays": ip})
    G = pd.DataFrame(grid_rows)
    qs = np.expm1(np.column_stack(reg.predict(five_matrix(G), output_type="quantiles", quantiles=QUANTILES)))
    grid = {}
    for m, s in enumerate(sponsors):
        scores = clf5.predict_proba(np.hstack([five_matrix(G), onehot(len(G), m)]))[:, 1]
        grid[s] = {"".join(map(str, c)): {"typical": scenario(s, qs[i, 2]), "bad": scenario(s, qs[i, 4]),
                                          "flag": bool(scores[i] > five_threshold),
                                          "typical_spend": round(float(qs[i, 2])), "bad_spend": round(float(qs[i, 4]))}
                   for i, c in enumerate(combos)}
    print(f"grid: {len(combos)} answer sets x {k} menus; five-answer flagger test AUC {five_auc:.3f}", flush=True)

    # How sure: straight from the eval JSON files.
    d4 = json.loads((RESULTS / "step4_decision.json").read_text())
    r3 = json.loads((RESULTS / "step3_reader.json").read_text())["results"]["gemma4:e4b-it-qat"]["gold_fields"]
    det = lazy["detection_test"]["at_risk_250"]["TabPFN classifier v2"]
    trust = {
        "people": d4["people"], "menus": len(d4["by_menu"]),
        "lowest_premium_hit": d4["pooled"]["lowest_premium"]["hit_rate"]["value"],
        "same_as_last_year_hit": d4["pooled"]["same_as_last_year"]["hit_rate"]["value"],
        "lowest_premium_regret": d4["pooled"]["lowest_premium"]["mean_regret"]["value"],
        "same_as_last_year_regret": d4["pooled"]["same_as_last_year"]["mean_regret"]["value"],
        "lazy_costly_share": lazy["lazy_regret_test"]["share_over_250"],
        "flagger_auc": det["auc"]["value"], "flagger_auc_ci": det["auc"]["ci95"], "flagger_recall": det["recall"]["value"],
        "five_answer_flagger_auc": five_auc,
        "reader_accuracy": r3["accuracy"], "reader_accuracy_clear": r3["accuracy_excluding_ambiguous"],
        "reader_wrong": r3["wrong_rate"], "reader_needs_check": r3["needs_check_rate"], "reader_values": r3["n"],
        "approximations": d4["approximations"],
        "sources": ["eval/results/step4_decision.json", "eval/results/step4b_lazy_safe.json", "eval/results/step3_reader.json"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"menus": menus, "order": sponsors, "people": out_people,
                               "questions": [{"key": q, "text": t_, "options": o} for q, t_, o in QUESTIONS],
                               "answer_values": values, "grid": grid, "trust": trust,
                               "five_answer_threshold": five_threshold}, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
