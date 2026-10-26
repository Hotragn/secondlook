"""Cost rules used to check an SBC's own coverage examples (Peg, Joe, Mia).

CMS's service-level inputs for the examples were not found (DECISIONS 17), so the
examples are not recomputed claim by claim. Instead each example is reconciled
against the plan fields read elsewhere in the same SBC, with a $1 tolerance for
rounding:
  arithmetic  deductibles + copayments + coinsurance + limits = the printed total
  bounds      deductibles <= the plan's individual deductible;
              cost sharing (total minus limits) <= the individual out-of-pocket limit;
              the total <= the total example cost
  parameters  the example box's plan deductible, specialist cost and hospital
              cost sharing equal the deductible, specialist and inpatient facility
              fields read from the summary and the medical-events table
An example "matches" when every check whose inputs were verified passes and the
arithmetic check could run.
"""

TOLERANCE = 1.0
PEOPLE = ("peg", "joe", "mia")


def _v(fields: dict, key: str):
    f = fields.get(key, {})
    return f.get("value") if f.get("status") == "verified" else None


def _same_cost(example: dict | None, plan: dict | None) -> bool | None:
    if not example or not plan:
        return None
    if example.get("coinsurance") is not None:
        return plan.get("coinsurance") is not None and abs(plan["coinsurance"] - example["coinsurance"]) < 1e-9
    if example.get("copay") is not None:
        return plan.get("copay") is not None and abs(plan["copay"] - example["copay"]) < TOLERANCE
    return None


def parameter_checks(fields: dict) -> dict:
    ded, ex_ded = _v(fields, "deductible_individual"), _v(fields, "ex_plan_deductible")
    return {
        "deductible": None if ded is None or ex_ded is None else abs(ded - ex_ded) < TOLERANCE,
        "specialist": _same_cost(_v(fields, "ex_specialist"), _v(fields, "specialist")),
        "hospital": _same_cost(_v(fields, "ex_hospital"), _v(fields, "inpatient_facility")),
    }


def reconcile(fields: dict) -> dict:
    params = parameter_checks(fields)
    ded_ind, oop_ind = _v(fields, "deductible_individual"), _v(fields, "oop_individual")
    out = {"parameters": params}
    for p in PEOPLE:
        parts = {k: _v(fields, f"{p}_{k}") for k in
                 ("total_cost", "deductibles", "copayments", "coinsurance", "limits", "total")}
        checks = {}
        if all(parts[k] is not None for k in ("deductibles", "copayments", "coinsurance", "limits", "total")):
            checks["arithmetic"] = abs(parts["deductibles"] + parts["copayments"] + parts["coinsurance"]
                                       + parts["limits"] - parts["total"]) <= TOLERANCE
        if parts["deductibles"] is not None and ded_ind is not None:
            checks["deductible_bound"] = parts["deductibles"] <= ded_ind + TOLERANCE
        if parts["total"] is not None and parts["limits"] is not None and oop_ind is not None:
            checks["oop_bound"] = parts["total"] - parts["limits"] <= oop_ind + TOLERANCE
        if parts["total"] is not None and parts["total_cost"] is not None:
            checks["cost_bound"] = parts["total"] <= parts["total_cost"] + TOLERANCE
        for name, ok in params.items():
            if ok is not None:
                checks[f"param_{name}"] = ok
        if "arithmetic" not in checks:
            status = "incomplete"
        else:
            status = "match" if all(checks.values()) else "mismatch"
        out[p] = {"status": status, "checks": checks, "printed": parts}
    return out
