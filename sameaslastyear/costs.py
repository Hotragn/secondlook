"""Rules engine (BRIEF D): what a member pays under a plan for a year of care.

Applies, in date order: preventive care at $0; the deductible (embedded or
aggregate for families); copays (per visit, per day with per-admission or
per-year caps, per fill) and coinsurance (with an optional per-fill cap),
each with or without the deductible; and the out-of-pocket limit, with an
optional separate prescription-drug limit. HSA helpers add the employer seed
and the tax saved on payroll contributions.
"""

from dataclasses import dataclass

# 2027 limits.
# IRS Rev. Proc. 2026-24, sec. 3 (https://www.irs.gov/pub/irs-drop/rp-26-24.pdf):
HSA_LIMIT_SELF = 4500.0  # § 223(b)(2)(A) self-only HSA contribution limit
HSA_LIMIT_FAMILY = 9000.0  # § 223(b)(2)(B) family HSA contribution limit
HDHP_MIN_DEDUCTIBLE = (1750.0, 3500.0)  # § 223(c)(2)(A) self-only, family
HDHP_MAX_OOP = (8700.0, 17400.0)  # § 223(c)(2)(A) self-only, family
# 26 U.S.C. § 223(b)(3)(B): catch-up for age 55+, fixed by statute (not indexed).
HSA_CATCH_UP = 1000.0
# CMS, "Premium Adjustment Percentage, Maximum Annual Limitation on Cost Sharing ...
# for the 2027 Benefit Year" (2026-01-29), https://www.cms.gov/files/document/2027-papi-parameters-guidance-2026-01-29.pdf
ACA_MAX_OOP = (12000.0, 24000.0)  # self-only, other than self-only
# Under the ACA embedded self-only rule, no one person in family coverage pays more than
# the self-only maximum.
ACA_EMBEDDED_INDIVIDUAL_MAX = ACA_MAX_OOP[0]
PAYROLL_TAX = 0.0765  # employee FICA: 6.2% Social Security + 1.45% Medicare

RX = "rx"  # service categories that count toward a separate prescription limit
RX_CATEGORIES = {"generic_drugs", "preferred_brand_drugs"}


@dataclass
class CostShare:
    copay: float | None = None
    coinsurance: float | None = None
    deductible_applies: bool | None = None  # None: use the plan default
    not_covered: bool = False
    per_day: bool = False  # copay is per day (inpatient)
    max_per_admission: float | None = None
    max_per_year: float | None = None  # cap on this row's copays in a year
    max_per_fill: float | None = None  # cap on coinsurance per prescription


@dataclass
class Plan:
    plan_id: str
    name: str
    premium_monthly: float
    deductible: float
    oop: float
    services: dict[str, CostShare]
    deductible_family: float | None = None
    oop_family: float | None = None
    embedded: bool = True
    oop_rx: float | None = None  # separate prescription out-of-pocket limit, if any
    hsa_eligible: bool = False
    employer_hsa: float = 0.0  # employer HSA deposit for self-only coverage

    def deductible_applies(self, cs: CostShare) -> bool:
        """Explicit wording wins. HSA plans apply the deductible to everything but preventive
        care; otherwise coinsurance waits for the deductible and copays do not."""
        if cs.deductible_applies is not None:
            return cs.deductible_applies
        if self.hsa_eligible:
            return True
        return cs.coinsurance is not None and cs.copay is None


@dataclass
class Event:
    category: str  # a key of Plan.services, or "preventive"
    allowed: float  # total paid for the event, from all sources
    month: int = 1
    nights: int = 0


def event_cost(plan: Plan, ev: Event, deductible_left: float, oop_left: float,
               copays_used: dict) -> tuple[float, float]:
    """(member cost, part of it that went to the deductible) for one event.

    `deductible_left` and `oop_left` come from the caller's ledgers, so self-only and family
    (embedded or aggregate) coverage share these rules. Not-covered care is paid in full and
    does not count toward the out-of-pocket limit."""
    if ev.category == "preventive" or ev.allowed <= 0:
        return 0.0, 0.0
    cs = plan.services[ev.category]
    if cs.not_covered:
        return ev.allowed, 0.0
    remaining, cost, ded = ev.allowed, 0.0, 0.0
    if plan.deductible_applies(cs):
        ded = max(0.0, min(deductible_left, remaining))
        cost += ded
        remaining -= ded
    if cs.copay is not None and remaining > 0:
        c = cs.copay * (max(ev.nights, 1) if cs.per_day else 1)
        if cs.max_per_admission is not None:
            c = min(c, cs.max_per_admission)
        if cs.max_per_year is not None:
            c = min(c, max(0.0, cs.max_per_year - copays_used.get(ev.category, 0.0)))
        c = min(c, remaining)
        copays_used[ev.category] = copays_used.get(ev.category, 0.0) + c
        cost += c
        remaining -= c
    if cs.coinsurance is not None and remaining > 0:
        x = cs.coinsurance * remaining
        if cs.max_per_fill is not None:
            x = min(x, cs.max_per_fill)
        cost += x
    capped = min(cost, max(0.0, oop_left))
    return capped, min(ded, capped)


def _bucket(plan: Plan, ev: Event) -> str:
    return RX if plan.oop_rx is not None and ev.category in RX_CATEGORIES else "medical"


def member_cost(plan: Plan, events: list[Event]) -> float:
    """Self-only coverage: what one person pays in cost sharing for a year of events."""
    ded_used, oop_used, copays, total = 0.0, {"medical": 0.0, RX: 0.0}, {}, 0.0
    limit = {"medical": plan.oop, RX: plan.oop_rx}
    for ev in sorted(events, key=lambda e: e.month):
        b = _bucket(plan, ev)
        c, d = event_cost(plan, ev, plan.deductible - ded_used, limit[b] - oop_used[b], copays)
        ded_used += d
        if ev.category != "preventive" and not plan.services[ev.category].not_covered:
            oop_used[b] += c
        total += c
    return total


def family_cost(plan: Plan, members: list[list[Event]]) -> float:
    """Family coverage, medical out-of-pocket only (a separate prescription limit is applied
    per person).

    Embedded: each person stops paying toward the deductible at the individual deductible,
    and everyone stops once the family total reaches the family deductible.
    Aggregate: no one's care is paid until the family deductible is met.
    Out-of-pocket: each person is capped at the individual limit (never above the ACA
    self-only maximum) and the family at the family limit."""
    fam_ded = plan.deductible_family if plan.deductible_family is not None else 2 * plan.deductible
    fam_oop = plan.oop_family if plan.oop_family is not None else 2 * plan.oop
    ind_oop = min(plan.oop, ACA_EMBEDDED_INDIVIDUAL_MAX)
    n = len(members)
    ded = [0.0] * n
    oop = [{"medical": 0.0, RX: 0.0} for _ in range(n)]
    copays = [{} for _ in range(n)]
    fam_oop_used, total = 0.0, 0.0
    timeline = sorted(((ev.month, i, ev) for i, evs in enumerate(members) for ev in evs), key=lambda t: (t[0], t[1]))
    for _, i, ev in timeline:
        fam_ded_left = fam_ded - sum(ded)
        ded_left = min(plan.deductible - ded[i], fam_ded_left) if plan.embedded else fam_ded_left
        b = _bucket(plan, ev)
        if b == RX:
            oop_left = plan.oop_rx - oop[i][RX]
        else:
            oop_left = min(ind_oop - oop[i]["medical"], fam_oop - fam_oop_used)
        c, d = event_cost(plan, ev, ded_left, oop_left, copays[i])
        ded[i] += d
        if ev.category != "preventive" and not plan.services[ev.category].not_covered:
            oop[i][b] += c
            if b != RX:
                fam_oop_used += c
        total += c
    return total


def hsa_tax_savings(employee_contribution: float, federal_rate: float, state_rate: float = 0.0,
                    through_payroll: bool = True, self_only: bool = True, age_55_plus: bool = False,
                    employer_seed: float = 0.0) -> float:
    """Tax saved on the employee's HSA contribution, capped at the IRS limit net of the
    employer seed. Payroll contributions also skip the 7.65% employee FICA."""
    limit = (HSA_LIMIT_SELF if self_only else HSA_LIMIT_FAMILY) + (HSA_CATCH_UP if age_55_plus else 0.0)
    allowed = max(0.0, min(employee_contribution, limit - employer_seed))
    rate = federal_rate + state_rate + (PAYROLL_TAX if through_payroll else 0.0)
    return allowed * rate


def annual_cost(plan: Plan, events: list[Event]) -> float:
    """Premiums plus cost sharing, less the employer's HSA deposit (self-only)."""
    return 12 * plan.premium_monthly + member_cost(plan, events) - (plan.employer_hsa if plan.hsa_eligible else 0.0)
