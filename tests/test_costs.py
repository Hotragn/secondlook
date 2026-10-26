"""Unit tests for the rules engine (BRIEF D), with hand-worked numbers."""

import pytest

from sameaslastyear import costs
from sameaslastyear.costs import CostShare, Event, Plan


def plan(**kw):
    base = dict(plan_id="p", name="p", premium_monthly=0.0, deductible=1000.0, oop=5000.0,
                services={"specialist": CostShare(coinsurance=0.2), "primary_care": CostShare(copay=25.0)})
    base.update(kw)
    return Plan(**base)


def test_deductible_then_coinsurance():
    # $1,000 deductible, then 20% of the remaining $2,000.
    assert costs.member_cost(plan(), [Event("specialist", 3000.0)]) == pytest.approx(1400.0)


def test_copay_without_deductible_leaves_deductible_untouched():
    p = plan()
    assert costs.member_cost(p, [Event("primary_care", 200.0)]) == 25.0
    # The copay did not count toward the deductible: a later $500 specialist bill is all deductible.
    assert costs.member_cost(p, [Event("primary_care", 200.0), Event("specialist", 500.0, month=2)]) == 525.0


def test_hsa_plan_applies_deductible_before_copays():
    p = plan(hsa_eligible=True, deductible=1750.0, services={"primary_care": CostShare(copay=20.0)})
    assert costs.member_cost(p, [Event("primary_care", 150.0)]) == 150.0
    events = [Event("primary_care", 1750.0, month=1), Event("primary_care", 150.0, month=2)]
    assert costs.member_cost(p, events) == 1770.0


def test_explicit_deductible_wording_wins():
    p = plan(hsa_eligible=True, services={"primary_care": CostShare(copay=20.0, deductible_applies=False)})
    assert costs.member_cost(p, [Event("primary_care", 150.0)]) == 20.0


def test_out_of_pocket_cap():
    p = plan(oop=2000.0)
    assert costs.member_cost(p, [Event("specialist", 20000.0)]) == 2000.0


def test_preventive_is_free_before_deductible():
    assert costs.member_cost(plan(), [Event("preventive", 400.0), Event("specialist", 100.0)]) == 100.0


def test_per_day_copay_with_admission_cap():
    p = plan(services={"inpatient_facility": CostShare(copay=150.0, per_day=True, max_per_admission=750.0)})
    assert costs.member_cost(p, [Event("inpatient_facility", 20000.0, nights=3)]) == 450.0
    assert costs.member_cost(p, [Event("inpatient_facility", 20000.0, nights=7)]) == 750.0


def test_per_day_copay_with_yearly_cap():
    p = plan(services={"inpatient_facility": CostShare(copay=200.0, per_day=True, max_per_year=600.0)})
    stays = [Event("inpatient_facility", 9000.0, month=2, nights=2), Event("inpatient_facility", 9000.0, month=6, nights=3)]
    assert costs.member_cost(p, stays) == 600.0


def test_coinsurance_capped_per_fill():
    # "30% coinsurance or $75 copay, whichever is less", deductible does not apply.
    p = plan(services={"preferred_brand_drugs": CostShare(coinsurance=0.3, max_per_fill=75.0, deductible_applies=False)})
    assert costs.member_cost(p, [Event("preferred_brand_drugs", 500.0)]) == 75.0
    assert costs.member_cost(p, [Event("preferred_brand_drugs", 100.0)]) == pytest.approx(30.0)


def test_separate_prescription_limit():
    p = plan(oop=2000.0, oop_rx=1000.0, services={"generic_drugs": CostShare(coinsurance=0.5, deductible_applies=False),
                                                  "specialist": CostShare(coinsurance=0.2)})
    events = [Event("generic_drugs", 5000.0), Event("specialist", 20000.0, month=2)]
    assert costs.member_cost(p, events) == 1000.0 + 2000.0


def test_not_covered_paid_in_full_outside_limit():
    p = plan(oop=100.0, services={"specialist": CostShare(not_covered=True)})
    assert costs.member_cost(p, [Event("specialist", 300.0)]) == 300.0


def test_family_embedded_deductible():
    p = plan(deductible=500.0, deductible_family=1000.0, oop=5000.0, oop_family=10000.0)
    members = [[Event("specialist", 1000.0, month=1)], [Event("specialist", 1000.0, month=2)],
               [Event("specialist", 1000.0, month=3)]]
    # 500 + 20% of 500; the same again; the family deductible is now met, so 20% of 1,000.
    assert costs.family_cost(p, members) == pytest.approx(600.0 + 600.0 + 200.0)


def test_family_aggregate_deductible():
    p = plan(deductible=500.0, deductible_family=1000.0, embedded=False, oop=5000.0, oop_family=10000.0)
    members = [[Event("specialist", 800.0, month=1)], [Event("specialist", 800.0, month=2)],
               [Event("specialist", 800.0, month=3)]]
    # 800 to the deductible; 200 more meets it, then 20% of 600; then 20% of 800.
    assert costs.family_cost(p, members) == pytest.approx(800.0 + 320.0 + 160.0)


def test_family_out_of_pocket_individual_and_family_caps():
    p = plan(deductible=0.0, deductible_family=0.0, oop=2000.0, oop_family=3000.0,
             services={"specialist": CostShare(coinsurance=0.5)})
    members = [[Event("specialist", 10000.0, month=1)], [Event("specialist", 10000.0, month=2)]]
    assert costs.family_cost(p, members) == 3000.0  # 2,000 for the first person, family cap leaves 1,000


def test_hsa_tax_savings_payroll_and_limit():
    assert costs.hsa_tax_savings(2000.0, 0.22, 0.05) == pytest.approx(2000.0 * (0.22 + 0.05 + 0.0765))
    # The employer seed uses part of the $4,500 self-only limit.
    assert costs.hsa_tax_savings(5000.0, 0.22, employer_seed=700.0, through_payroll=False) == pytest.approx(3800.0 * 0.22)
    assert costs.hsa_tax_savings(6000.0, 0.10, self_only=True, age_55_plus=True,
                                 through_payroll=False) == pytest.approx(5500.0 * 0.10)


def test_annual_cost_includes_premium_and_employer_hsa():
    p = plan(premium_monthly=50.0, hsa_eligible=True, employer_hsa=700.0)
    assert costs.annual_cost(p, [Event("specialist", 500.0)]) == pytest.approx(600.0 + 500.0 - 700.0)


def test_2027_limits_match_sources():
    assert (costs.HSA_LIMIT_SELF, costs.HSA_LIMIT_FAMILY) == (4500.0, 9000.0)
    assert costs.HDHP_MIN_DEDUCTIBLE == (1750.0, 3500.0) and costs.HDHP_MAX_OOP == (8700.0, 17400.0)
    assert costs.ACA_MAX_OOP == (12000.0, 24000.0) and costs.HSA_CATCH_UP == 1000.0


@pytest.mark.parametrize("person,total_cost,printed_total", [("peg", 7540.0, 4712.0), ("joe", 5400.0, 4070.0),
                                                             ("mia", 1900.0, 1900.0)])
def test_sbc_coverage_examples_georgia_uhc_hdhp(person, total_cost, printed_total):
    """Dev SBC ga-shbp-uhc-hdhp-2026: $3,500 deductible, then 30% coinsurance on every service, so the
    engine must reproduce the printed totals exactly (Peg $4,712, Joe $4,070, Mia $1,900)."""
    p = plan(deductible=3500.0, oop=6450.0, hsa_eligible=True, services={"specialist": CostShare(coinsurance=0.3)})
    assert costs.member_cost(p, [Event("specialist", total_cost)]) == pytest.approx(printed_total)
