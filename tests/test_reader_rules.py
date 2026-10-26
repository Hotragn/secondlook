"""Unit tests for the reader's parsers and the coverage-example rules.

Cases come from dev-set SBC wording only (held-out documents are not used here).
"""

import pytest

from sameaslastyear import reader, rules


@pytest.mark.parametrize("quote,role,before,after,expected", [
    ("$3,500", "individual", "", " individual / $7,000 family", 3500.0),
    ("$7,000", "family", "$3,500 individual / ", " family. If you", 7000.0),
    ("$7,000", "individual", "$3,500 individual / ", " family", None),
    ("Individual: $1,500 Family: $3,000", "family", "", "", 3000.0),
    ("Individual: $1,500 Family: $3,000", "individual", "", "", 1500.0),
    ("For in-network providers: $1,250/individual or $2,500/family For out-of-network providers: $2,500/individual",
     "family", "", "", 2500.0),
    ("Self-only coverage: $1,750 Other than self-only coverage: $3,500", "family", "", "", 3500.0),
    ("$0", "individual", "", " See the Common", 0.0),
    ("$0", "family", "", "", 0.0),
    ("$500", "family", "", "", None),
])
def test_money_roles(quote, role, before, after, expected):
    assert reader.parse_money_role(quote, role, before, after) == expected


@pytest.mark.parametrize("quote,copay,coins,ded", [
    ("30% coinsurance After Deductible", None, 0.3, True),
    ("$10 copay/visit, deductible doesn't apply", 10.0, None, False),
    ("$20 / visit, deductible does not apply. 20% coinsurance for other covered services received during a visit.",
     20.0, None, False),
    ("No Charge", 0.0, None, None),
    ("$40 copay per visit, deductible does not apply 40% coinsurance", 40.0, None, False),
])
def test_cost_share(quote, copay, coins, ded):
    v = reader.parse_cost_share(quote)
    assert (v["copay"], v["coinsurance"], v["deductible_applies"]) == (copay, coins, ded)


def test_not_covered_and_unparseable():
    assert reader.parse_cost_share("Not covered")["not_covered"] is True
    assert reader.parse_cost_share("See separate Prescription Drug Plan SBC") is None


def test_plan_name_rejects_document_title():
    assert reader.PARSERS["plan_name"]("Summary of Benefits and Coverage: What this Plan Covers") is None


def test_verify_exact_page_fixed_and_missing():
    pages = ["Primary care visit\n30% coinsurance After \nDeductible", "Specialist visit $40 copay"]
    assert reader.verify(pages, "30% coinsurance After Deductible", 1) == ("exact", 1)
    assert reader.verify(pages, "$40 copay", 1) == ("page_fixed", 2)
    assert reader.verify(pages, "$45 copay", 2) == ("not_found", None)


def _fields(peg_total=4712.0):
    v = lambda x: {"status": "verified", "value": x}
    f = {"deductible_individual": v(3500.0), "oop_individual": v(6450.0), "ex_plan_deductible": v(3500.0),
         "ex_specialist": v({"copay": None, "coinsurance": 0.3}), "specialist": v({"copay": None, "coinsurance": 0.3}),
         "ex_hospital": v({"copay": None, "coinsurance": 0.3}), "inpatient_facility": v({"copay": None, "coinsurance": 0.3})}
    for p, parts in {"peg": (7540, 3500, 0, 1212, 0, peg_total), "joe": (5400, 3500, 0, 570, 0, 4070),
                     "mia": (1900, 1900, 0, 0, 0, 1900)}.items():
        for k, x in zip(("total_cost", "deductibles", "copayments", "coinsurance", "limits", "total"), parts):
            f[f"{p}_{k}"] = v(float(x))
    return f


def test_examples_reconcile():
    r = rules.reconcile(_fields())
    assert [r[p]["status"] for p in rules.PEOPLE] == ["match", "match", "match"]


def test_example_arithmetic_mismatch():
    r = rules.reconcile(_fields(peg_total=4800.0))
    assert r["peg"]["status"] == "mismatch" and r["peg"]["checks"]["arithmetic"] is False


def test_example_cost_parameters():
    assert reader.parse_example_cost("Hospital (facility) copayment $0") == {"copay": 0.0, "coinsurance": None}
    assert reader.parse_example_cost("$100 per day, Maximum $200 per admission") == {"copay": 100.0, "coinsurance": None}
    assert reader.parse_example_cost("Specialist coinsurance 30%") == {"copay": None, "coinsurance": 0.3}
    assert reader.parse_example_cost("Other coinsurance") is None
