"""Load the real 2027 menus (data/menus/menus_2027.json) as Plans, verifying every quote.

Deductible, out-of-pocket limit and five cost-sharing rows come from the committed
held-out gold. Every other value carries a quote: the quote must appear in that
PDF's extracted text, the value is parsed from it, and any extra number typed in
the menu file (a cap, a per-fill limit) must itself appear in the quote. A failed
check raises, so the benchmark cannot run on an unverified menu.
"""

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

from sameaslastyear.costs import CostShare, Plan
from sameaslastyear.reader import MONEY, load_pages, normalize, parse_cost_share, parse_money_role, parse_percent

ROOT = Path(__file__).resolve().parents[1]
MENUS = ROOT / "data" / "menus" / "menus_2027.json"
GOLD = ROOT / "eval" / "gold" / "heldout_gold.json"


@lru_cache(maxsize=None)
def _paths() -> dict:
    with (ROOT / "data" / "sbc" / "manifest.csv").open() as f:
        return {r["doc_id"]: ROOT / r["path"] for r in csv.DictReader(f)}


@lru_cache(maxsize=None)
def _text(doc: str) -> str:
    return normalize(" ".join(load_pages(_paths()[doc], max_pages=200)))


def verified(src: dict) -> str:
    q = normalize(src["quote"])
    if q not in _text(src["doc"]):
        raise ValueError(f"quote not found in {src['doc']}: {src['quote']!r}")
    for key in ("max_per_admission", "max_per_year", "max_per_fill"):
        if key in src and src[key] not in [float(m.replace(",", "")) for m in re.findall(MONEY, q)]:
            raise ValueError(f"{key}={src[key]} is not in the quote {src['quote']!r}")
    return q


def _money(q: str) -> list[float]:
    return [float(m.replace(",", "")) for m in re.findall(MONEY, q)]


def _from_gold(g: dict) -> CostShare:
    return CostShare(copay=g["copay"], coinsurance=g["coinsurance"], deductible_applies=g["deductible_applies"],
                     not_covered=g["not_covered"])


def _from_quote(src: dict) -> CostShare:
    v = parse_cost_share(verified(src))
    if v is None:
        raise ValueError(f"no cost sharing parsed from {src['quote']!r}")
    return CostShare(copay=v["copay"], coinsurance=v["coinsurance"], deductible_applies=v["deductible_applies"],
                     not_covered=v["not_covered"], max_per_fill=src.get("max_per_fill"))


def premium(src: dict) -> float:
    q = verified(src)
    if "column" in src:
        return _money(q)[src["column"]]
    share = parse_percent(verified(src["share"]))
    return round(_money(q)[0] * share, 2)


def load(path: Path = MENUS) -> dict[str, list[Plan]]:
    spec = json.loads(path.read_text())
    gold = json.loads(GOLD.read_text())["documents"]
    out = {}
    for sponsor, s in spec["sponsors"].items():
        plans = []
        for plan_id, p in s["plans"].items():
            g = gold[plan_id]["fields"]
            services = {k: _from_gold(g[k]) for k in ("primary_care", "specialist", "emergency_room",
                                                       "generic_drugs", "inpatient_facility")}
            if "generic_cap" in p:
                verified(p["generic_cap"])
                services["generic_drugs"].max_per_fill = p["generic_cap"]["max_per_fill"]
            if "inpatient_detail" in p:
                d = p["inpatient_detail"]
                if _money(verified(d))[0] != services["inpatient_facility"].copay:
                    raise ValueError(f"{plan_id}: inpatient detail does not match the gold copay")
                cs = services["inpatient_facility"]
                cs.per_day = bool(d.get("per_day"))
                cs.max_per_admission, cs.max_per_year = d.get("max_per_admission"), d.get("max_per_year")
            services["outpatient_facility"] = _from_quote(p["outpatient_facility"])
            services["preferred_brand_drugs"] = _from_quote(p["preferred_brand_drugs"])
            er = services["emergency_room"]
            # An ER visit that leads to admission is billed with the stay: the visit copay is waived.
            services["emergency_room_admitted"] = CostShare(coinsurance=er.coinsurance,
                                                            deductible_applies=er.deductible_applies)
            hsa = p.get("hsa")
            plans.append(Plan(
                plan_id=plan_id, name=p["name"], premium_monthly=premium(p["premium"]),
                deductible=float(g["deductible_individual"]), oop=float(g["oop_individual"]),
                deductible_family=float(g["deductible_family"]), oop_family=float(g["oop_family"]),
                oop_rx=parse_money_role(verified(p["oop_rx"]), "individual") if "oop_rx" in p else None,
                services=services, hsa_eligible=hsa is not None,
                employer_hsa=_money(verified(hsa))[0] if hsa else 0.0,
            ))
        out[sponsor] = plans
    return out
