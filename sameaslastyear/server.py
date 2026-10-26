"""Local mode (make app): the same UI as docs/, with live work done on this computer.

- POST /api/answers runs the five-answer TabPFN regressor and classifier on the CPU, trained on the
  training people only, and prices every plan in the chosen 2027 menu on donors' real 2024 events.
- POST /api/read reads an uploaded SBC PDF with the e4b reader through the local Ollama server.
  The file is written to a temporary directory and deleted afterwards. Nothing is sent anywhere.

Run: make app   (serves http://127.0.0.1:8765)
"""

import json
import os
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, File, UploadFile  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from sameaslastyear import menus as menu_loader  # noqa: E402
from sameaslastyear import reader  # noqa: E402
from sameaslastyear.scenarios import cost_matrix, donor_finder, load_events  # noqa: E402
from sameaslastyear.spend import QUANTILES, SEED, prepare_tabpfn_auth  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
READER_MODEL = "gemma4:e4b-it-qat"
FIVE = ["age", "self_rated_health", "n_conditions", "rx_fills", "er_visits", "inpatient_stays"]
LABELS = {
    "plan_name": "Plan name", "network_type": "Plan type", "deductible_individual": "Deductible (one person)",
    "deductible_family": "Deductible (family)", "deductible_family_rule": "Family deductible works per person",
    "services_before_deductible": "Some care covered before the deductible", "oop_individual": "Out-of-pocket limit (one person)",
    "oop_family": "Out-of-pocket limit (family)", "hsa": "Works with a health savings account",
    **{k: v for k, v in reader.SERVICE_ROWS.items()},
    "peg_total": "Peg (having a baby) would pay", "joe_total": "Joe (diabetes) would pay", "mia_total": "Mia (fracture) would pay",
}


def five_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[FIVE].astype(float).copy()
    for c in ("rx_fills", "er_visits", "inpatient_stays"):
        X[c] = np.log1p(X[c])
    return X.to_numpy()


def display(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, dict):
        if v.get("not_covered"):
            return "Not covered"
        parts = []
        if v.get("copay") is not None:
            parts.append("No charge" if v["copay"] == 0 and v.get("coinsurance") is None else f"${v['copay']:,.0f}")
        if v.get("coinsurance") is not None:
            parts.append(f"{v['coinsurance'] * 100:g}%")
        return " then ".join(parts)
    if isinstance(v, float):
        return f"${v:,.0f}"
    return str(v)


class Model:
    """Five-answer models and donor costs, built once at start-up on the CPU."""

    def __init__(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        warnings.simplefilter("ignore")
        prepare_tabpfn_auth()
        self.data = json.loads((DOCS / "data" / "secondlook.json").read_text())
        t = pd.read_parquet(ROOT / "data" / "derived" / "spend_panel28.parquet")
        train = t[t.split == "train"].reset_index(drop=True)
        self.plans = menu_loader.load()
        self.sponsors = self.data["order"]
        ev = load_events(2024, set(train.DUPERSID))
        self.donor_cost = {s: cost_matrix(self.plans[s], ev, list(train.DUPERSID)) for s in self.sponsors}
        self.donors = donor_finder(train.spend_y2.to_numpy())
        k = len(self.sponsors)
        lazy = []
        for s in self.sponsors:
            c = self.donor_cost[s]
            lp = [p.plan_id for p in self.plans[s]].index(self.data["menus"][s]["lowest_premium"])
            lazy.append(c[:, lp] - c.min(1))
        y = (np.concatenate(lazy) > 250).astype(int)
        X5 = five_matrix(train)
        self.reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2, random_state=SEED, device="cpu")
        self.reg.fit(X5, np.log1p(train.spend_y2.to_numpy()))
        self.clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, random_state=SEED, device="cpu")
        self.clf.fit(np.vstack([np.hstack([X5, np.tile(np.eye(k)[m], (len(train), 1))]) for m in range(k)]), y)

    def answer(self, menu: str, answers: list[int]) -> dict:
        v = self.data["answer_values"]
        er, ip = v["urgent"][answers[4]]
        row = pd.DataFrame([{"age": v["age"][answers[0]], "self_rated_health": v["health"][answers[1]],
                             "n_conditions": v["conditions"][answers[2]], "rx_fills": v["prescriptions"][answers[3]],
                             "er_visits": er, "inpatient_stays": ip}])
        q = np.expm1(np.column_stack(self.reg.predict(five_matrix(row), output_type="quantiles", quantiles=QUANTILES)))[0]
        m = self.sponsors.index(menu)
        score = float(self.clf.predict_proba(np.hstack([five_matrix(row), np.eye(len(self.sponsors))[[m]]]))[0, 1])
        scen = lambda x: [round(float(c), 2) for c in self.donor_cost[menu][self.donors(x)].mean(0)]
        return {"typical": scen(q[2]), "bad": scen(q[4]), "flag": score > self.data["five_answer_threshold"],
                "typical_spend": round(float(q[2])), "bad_spend": round(float(q[4])), "computed": "live on this computer"}


app = FastAPI(title="Second Look (local)", docs_url=None, redoc_url=None)
_model: Model | None = None


def model() -> Model:
    global _model
    if _model is None:
        _model = Model()
    return _model


class Answers(BaseModel):
    menu: str
    answers: list[int]


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "local": True}


@app.post("/api/answers")
def answers(body: Answers) -> dict:
    return model().answer(body.menu, body.answers)


@app.post("/api/read")
async def read(pdf: UploadFile = File(...)) -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plan.pdf"
        path.write_bytes(await pdf.read())
        fields = reader.check_fields(path, reader.run_model(READER_MODEL, path))
    out = [{"key": k, "label": LABELS[k], "status": f["status"], "display": display(f["value"]), "quote": f["quote"],
            "page": f["found_page"], "reason": f["reason"]} for k, f in fields.items() if k in LABELS]
    return {"fields": out, "seconds": round(time.perf_counter() - started), "model": READER_MODEL}


app.mount("/", StaticFiles(directory=DOCS, html=True), name="site")
