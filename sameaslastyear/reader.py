"""Plan reader: Gemma 4 locates and quotes, code verifies and parses (BRIEF A).

Page text comes from pypdfium2 in content-stream order, which keeps each SBC
table cell's text together. The model sees the SBC in three sections (important
questions, common medical events, coverage examples) and returns every field as
{quote, page} under a JSON schema. Code then
  1. checks the quote is in the extracted text of that page after whitespace and
     punctuation normalisation (coverage-example quotes must also sit inside that
     person's example box), and
  2. parses the value from the quote with regular expressions.
The model never supplies a number. A field without a verified, parseable quote is
"needs_check".
"""

import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

import pypdfium2 as pdfium

OLLAMA = "http://127.0.0.1:11434"
NUM_CTX = 16384

# ---------------------------------------------------------------- fields

SUMMARY_FIELDS = {
    "plan_name": "The name of this plan as printed near the top of page 1 (for example 'UMP Classic'), not the document title 'Summary of Benefits and Coverage'.",
    "network_type": "The plan type, e.g. 'Plan Type: PPO' or 'HMO', as printed near the top of page 1.",
    "deductible_individual": "The in-network overall deductible for one person (the answer to 'What is the overall deductible?').",
    "deductible_family": "The in-network overall deductible for a family.",
    "deductible_family_rule": "The sentence that explains how the family deductible works for family members.",
    "services_before_deductible": "The answer to 'Are there services covered before you meet your deductible?'.",
    "oop_individual": "The in-network out-of-pocket limit for one person.",
    "oop_family": "The in-network out-of-pocket limit for a family.",
    "hsa": "Text saying the plan is eligible for, or paired with, a Health Savings Account (HSA). Empty if the document does not say.",
}

SERVICE_ROWS = {
    "primary_care": "Primary care visit to treat an injury or illness",
    "specialist": "Specialist visit",
    "preventive": "Preventive care/screening/immunization",
    "diagnostic_test": "Diagnostic test (x-ray, blood work)",
    "imaging": "Imaging (CT/PET scans, MRIs)",
    "generic_drugs": "Generic drugs",
    "preferred_brand_drugs": "Preferred brand drugs",
    "non_preferred_brand_drugs": "Non-preferred brand drugs",
    "specialty_drugs": "Specialty drugs",
    "outpatient_facility": "Outpatient surgery facility fee (e.g., ambulatory surgery center)",
    "outpatient_physician": "Outpatient surgery physician/surgeon fees",
    "emergency_room": "Emergency room care",
    "emergency_transport": "Emergency medical transportation",
    "urgent_care": "Urgent care",
    "inpatient_facility": "Hospital stay facility fee (e.g., hospital room)",
    "inpatient_physician": "Hospital stay physician/surgeon fees",
    "mental_health_outpatient": "Mental/behavioral health or substance use disorder outpatient services",
    "mental_health_inpatient": "Mental/behavioral health or substance use disorder inpatient services",
    "pregnancy_office_visits": "Pregnancy office visits",
    "childbirth_professional": "Childbirth/delivery professional services",
    "childbirth_facility": "Childbirth/delivery facility services",
    "home_health": "Home health care",
    "rehabilitation": "Rehabilitation services",
    "habilitation": "Habilitation services",
    "skilled_nursing": "Skilled nursing care",
    "durable_medical_equipment": "Durable medical equipment",
    "hospice": "Hospice services",
    "children_eye_exam": "Children's eye exam",
    "children_glasses": "Children's glasses",
    "children_dental": "Children's dental check-up",
}

PEOPLE = {"peg": "Peg is Having a Baby", "joe": "Managing Joe", "mia": "Mia's Simple Fracture"}
EXAMPLE_PARTS = {
    "total_cost": "the 'Total Example Cost' line",
    "deductibles": "the 'Deductibles' line under cost sharing",
    "copayments": "the 'Copayments' line",
    "coinsurance": "the 'Coinsurance' line",
    "limits": "the 'Limits or exclusions' line",
    "total": "the 'The total ... would pay is' line",
}
EXAMPLE_PARAMS = {
    "ex_plan_deductible": "In Peg's example box: the 'The plan's overall deductible' line.",
    "ex_specialist": "In Peg's example box: the 'Specialist copayment' or 'Specialist coinsurance' line.",
    "ex_hospital": "In Peg's example box: the 'Hospital (facility) coinsurance' line.",
    "ex_other": "In Peg's example box: the 'Other coinsurance' line.",
}
EXAMPLE_FIELDS = {
    **EXAMPLE_PARAMS,
    **{f"{p}_{k}": f"In {name.split(' is')[0].split('Managing ')[-1].split(chr(39))[0]}'s example: {d}."
       for p, name in PEOPLE.items() for k, d in EXAMPLE_PARTS.items()},
}

SECTIONS = {
    "summary": SUMMARY_FIELDS,
    "services": {k: f"In-network cost for the row '{v}'." for k, v in SERVICE_ROWS.items()},
    "examples": EXAMPLE_FIELDS,
}

SYSTEM = (
    "You read a health plan's Summary of Benefits and Coverage (SBC) and copy text from it.\n"
    "For every field, return the shortest exact text from the document that states the answer, "
    "and the page number shown in the '=== Page N ===' marker above it.\n"
    "Rules:\n"
    "- Copy characters exactly as printed. Never paraphrase, summarise, calculate or add words.\n"
    "- For a cost in the medical events table, copy only the in-network cell (the first 'What You Will Pay' "
    "column), not the out-of-network cell or the limitations column.\n"
    "- For deductibles and out-of-pocket limits, copy the in-network amount with the words that say who it "
    "applies to (for example '$1,500 individual / $3,000 family').\n"
    "- If the document does not state a field, return an empty quote and page 0."
)


def schema(fields: dict) -> dict:
    item = {"type": "object", "properties": {"quote": {"type": "string"}, "page": {"type": "integer"}},
            "required": ["quote", "page"]}
    return {"type": "object", "properties": {k: item for k in fields}, "required": list(fields)}


# ---------------------------------------------------------------- text

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[‘’‛′`]", "'", s)
    s = re.sub(r"[“”‟″]", '"', s)
    s = re.sub(r"[‐-―−￾]", "-", s)
    s = re.sub(r"[-•▪■]", " ", s)  # bullets and private-use glyphs
    return " ".join(s.split())


def load_pages(path: Path, max_pages: int = 16) -> list[str]:
    pdf = pdfium.PdfDocument(path)
    try:
        return [pdf[i].get_textpage().get_text_range().replace("\r\n", "\n")
                for i in range(min(len(pdf), max_pages))]
    finally:
        pdf.close()


def sections(pages: list[str]) -> dict[str, list[int]]:
    """Page indices (0-based) for each section, found from the SBC's fixed headings."""
    norm = [normalize(p).lower() for p in pages]
    find = lambda pat, default: next((i for i, p in enumerate(norm) if re.search(pat, p)), default)
    end = find(r"total mia would pay", len(pages) - 1)
    examples = find(r"peg is having a baby", end)
    events = find(r"common medical event", 0)
    return {
        "summary": list(range(0, events + 1)),
        "services": list(range(events, examples)) or list(range(0, examples + 1)),
        "examples": list(range(examples, end + 1)),
    }


def prompt_text(pages: list[str], idx: list[int]) -> str:
    return "\n\n".join(f"=== Page {i + 1} ===\n{pages[i]}" for i in idx)


# ---------------------------------------------------------------- model

def chat(model: str, fields: dict, document: str, timeout: int = 900) -> tuple[dict, dict]:
    listing = "\n".join(f"- {k}: {d}" for k, d in fields.items())
    body = {
        "model": model, "stream": False, "think": False, "keep_alive": "10m",
        "format": schema(fields),
        "options": {"temperature": 0, "num_ctx": NUM_CTX, "seed": 0},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Fields:\n{listing}\n\nDocument:\n{document}"},
        ],
    }
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        r = json.load(resp)
    stats = {"seconds": round(time.perf_counter() - t0, 2),
             "prompt_tokens": r.get("prompt_eval_count"), "output_tokens": r.get("eval_count")}
    try:
        out = json.loads(r["message"]["content"])
    except (KeyError, json.JSONDecodeError):
        out = {}
    return out, stats


def run_model(model: str, path: Path) -> dict:
    """Raw model output for one SBC: {section: {field: {quote, page}}} plus timings."""
    pages = load_pages(path)
    secs = sections(pages)
    raw, timing = {}, {}
    for name, fields in SECTIONS.items():
        raw[name], timing[name] = chat(model, fields, prompt_text(pages, secs[name]))
    return {"sections": secs, "raw": raw, "timing": timing}


# ---------------------------------------------------------------- parsing

MONEY = r"\$\s?(\d[\d,]*(?:\.\d+)?)"
OUT_OF_NETWORK = r"(out[- ]of[- ]\s?network|non[- ]?network|non[- ]?participating|non[- ]?preferred provider)"


def in_network(q: str) -> str:
    m = re.search(OUT_OF_NETWORK, q, re.I)
    return q[: m.start()] if m and m.start() > 0 else q


def amounts(q: str) -> list[tuple[float, int, int]]:
    return [(float(m.group(1).replace(",", "")), m.start(), m.end()) for m in re.finditer(MONEY, q)]


ROLE = {
    "individual": r"individuals?|persons?|members?|self[- ]only|single|employee[- ]only|each|you(?!\s*\+)",
    "family": r"family|families|other than self[- ]only|aggregate|two or more|2 or more|household|you \+ family",
}
LABEL_TAIL = r"(?:\s+(?:coverage|tier|only))?"
LABEL_END = LABEL_TAIL + r"\s*:"


def amount_roles(text: str, span: tuple[int, int]) -> list[tuple[float, str | None]]:
    """Each amount inside text[span] with the role named next to it.

    Suffix style: '$500 individual', '$1,250/individual', '$7,000 You + Family'.
    Prefix style: 'Individual $8,480', 'Family: $3,000'. A role word straight after the
    previous amount is that amount's suffix, not a prefix for the next one.
    """
    every = amounts(text)
    out = []
    for value, s, e in every:
        if not (span[0] <= s and e <= span[1]):
            continue
        nxt = min([s2 for _, s2, _ in every if s2 > s] + [len(text)])
        prv = max([e2 for _, _, e2 in every if e2 <= s] + [0])
        role = None
        for name, pat in ROLE.items():  # family last, so 'other than self-only' wins over 'self-only'
            m = re.match(rf"\s*(?:/(?=\S)|per\s+|for (?:an?|each|the)\s+)?(?:[A-Za-z]+\s*\+\s*)?(?:{pat})\b",
                         text[e:nxt], re.I)
            if m and not re.match(LABEL_END, text[e + m.end():nxt], re.I):
                role = name
        if role is None:
            before = text[prv:s]
            for name, pat in ROLE.items():
                m = re.search(rf"(?:{pat})\b{LABEL_TAIL}\s*:?\s*$", before, re.I)
                if not m:
                    continue
                labelled = before[m.start():].rstrip().endswith(":")
                separated = prv == 0 or re.search(r"[/;,|.]|\b(?:or|and)\b", before[: m.start()])
                if labelled or separated:
                    role = name
        out.append((value, role))
    return out


def parse_money_role(q: str, role: str, before: str = "", after: str = "") -> float | None:
    """Amount for `role` from the quote; role words may come from the document text around it."""
    qn = normalize(q)
    cut = in_network(qn)
    text = before + cut + (after if cut == qn else "")
    span = (len(before), len(before) + len(cut))
    roles = amount_roles(text, span)
    if not roles:
        if re.search(r"\bno (overall )?(deductible|out-of-pocket)|does not have (an overall|a) deductible", cut, re.I):
            return 0.0
        return None
    for value, r in roles:
        if r == role:
            return value
    if len(roles) == 1 and roles[0][1] is None and (role == "individual" or roles[0][0] == 0):
        return roles[0][0]  # a bare amount is the per-person figure; a bare $0 means no deductible for anyone
    return None


def parse_money(q: str) -> float | None:
    found = amounts(normalize(q))
    return found[-1][0] if found else None


def parse_percent(q: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s?%", normalize(q))
    return float(m.group(1)) / 100 if m else None


def parse_cost_share(q: str) -> dict | None:
    """In-network cost sharing from a table cell quote."""
    q = in_network(normalize(q))
    low = q.lower()
    tokens = sorted(
        [("copay", v, s) for v, s, _ in amounts(q)]
        + [("coinsurance", float(m.group(1)) / 100, m.start()) for m in re.finditer(r"(\d+(?:\.\d+)?)\s?%", q)],
        key=lambda t: t[2],
    )
    value = {"copay": None, "coinsurance": None, "not_covered": False, "deductible_applies": None}
    if re.search(r"deductible (does not|doesn't|does n't) apply|no deductible|not subject to (the )?deductible|deductible (is )?waived", low):
        value["deductible_applies"] = False
    elif re.search(r"after (the )?(in-network )?deductible|deductible applies|subject to (the )?deductible|after ded\b", low):
        value["deductible_applies"] = True
    if tokens:
        kind, v, pos = tokens[0]
        value[kind] = v
        if len(tokens) > 1:
            kind2, v2, pos2 = tokens[1]
            between = low[pos: pos2]
            if kind2 != kind and re.search(r"\b(then|plus|and|after which|followed by)\b", between) and pos2 - pos < 40:
                value[kind2] = v2
        return value
    if re.search(r"\bno charge\b|\bno cost\b|covered (at|in) full|\bno copay", low):
        value["copay"] = 0.0
        return value
    if re.search(r"\bnot covered\b", low):
        value["not_covered"] = True
        return value
    return None


def parse_example_cost(q: str) -> dict | None:
    """An example box's plan parameter: a copay ('copayment $0', '$100 per day') or coinsurance ('30%')."""
    pct, money = parse_percent(q), amounts(normalize(q))
    if pct is not None:
        return {"copay": None, "coinsurance": pct}
    if money:
        return {"copay": money[0][0], "coinsurance": None}
    if re.search(r"\bno charge\b", q, re.I):
        return {"copay": 0.0, "coinsurance": None}
    return None


def parse_network_type(q: str) -> str | None:
    m = re.search(r"\b(HMO|PPO|EPO|POS|HDHP|CDHP|HSA|DHMO|Indemnity)\b", normalize(q))
    return m.group(1) if m else None


def parse_family_rule(q: str) -> bool | None:
    low = normalize(q).lower()
    if re.search(r"each (family )?member must meet (their|his or her) own|embedded|individual deductible", low) and \
            not re.search(r"before the plan begins to pay", low):
        return True
    if re.search(r"overall family deductible must be met|family deductible must be met|aggregate", low):
        return False
    return None


def parse_yes_no(q: str) -> bool | None:
    low = normalize(q).lower()
    if re.match(r"\s*yes\b", low):
        return True
    if re.match(r"\s*no\b", low):
        return False
    return None


def parse_hsa(q: str) -> bool | None:
    low = normalize(q).lower()
    if not re.search(r"\bhsa\b|health savings account", low):
        return None
    return not re.search(r"not (eligible|compatible|qualified)|ineligible", low)


PARSERS = {
    "plan_name": lambda q: None if re.search(r"summary of benefits|what this plan covers", q, re.I) else (normalize(q) or None),
    "network_type": parse_network_type,
    "deductible_individual": lambda q: parse_money_role(q, "individual"),
    "deductible_family": lambda q: parse_money_role(q, "family"),
    "deductible_family_rule": parse_family_rule,
    "services_before_deductible": parse_yes_no,
    "oop_individual": lambda q: parse_money_role(q, "individual"),
    "oop_family": lambda q: parse_money_role(q, "family"),
    "hsa": parse_hsa,
    **{k: parse_cost_share for k in SERVICE_ROWS},
    "ex_plan_deductible": parse_money,
    "ex_specialist": lambda q: parse_example_cost(q),
    "ex_hospital": lambda q: parse_example_cost(q),
    "ex_other": lambda q: parse_example_cost(q),
    **{f"{p}_{k}": parse_money for p in PEOPLE for k in EXAMPLE_PARTS},
}


# ---------------------------------------------------------------- verify

def example_blocks(pages: list[str], idx: list[int]) -> dict[str, str]:
    """Each person's coverage-example box, cut at the end of its 'The total X would pay is $N' line.

    Some SBCs print the three headings together before the boxes, so headings cannot be
    used as boundaries; each box's own total line can.
    """
    text = normalize(" ".join(pages[i] for i in idx))
    ends = {}
    for p in PEOPLE:
        m = re.search(rf"the total {p} would pay is\s*\$\s?[\d,]+(?:\.\d+)?", text, re.I)
        if m:
            ends[p] = m.end()
    blocks, start = {}, 0
    for p, e in sorted(ends.items(), key=lambda kv: kv[1]):
        blocks[p] = text[start:e]
        start = e
    return blocks


def verify(pages: list[str], quote: str, page: int, block: str | None = None) -> tuple[str, int | None]:
    """'exact' on the cited page, 'page_fixed' if only on another page, else 'not_found'."""
    q = normalize(quote)
    if not q:
        return "empty", None
    norm = [normalize(p) for p in pages]
    if block is not None and q not in block:
        return "not_in_example_box", None
    if 1 <= page <= len(pages) and q in norm[page - 1]:
        return "exact", page
    hits = [i + 1 for i, p in enumerate(norm) if q in p]
    return ("page_fixed", hits[0]) if hits else ("not_found", None)


MONEY_ROLE_FIELDS = {"deductible_individual": "individual", "deductible_family": "family",
                     "oop_individual": "individual", "oop_family": "family"}


def check_fields(path: Path, model_out: dict) -> dict:
    pages = load_pages(path)
    blocks = example_blocks(pages, model_out["sections"]["examples"])
    fields = {}
    for section, spec in SECTIONS.items():
        got = model_out["raw"].get(section, {})
        for key in spec:
            item = got.get(key) or {}
            quote, page = str(item.get("quote", "")), int(item.get("page", 0) or 0)
            person = key.split("_")[0] if key.split("_")[0] in PEOPLE else ("peg" if key.startswith("ex_") else None)
            match, found_page = verify(pages, quote, page, blocks.get(person) if person else None)
            value = None
            if match in ("exact", "page_fixed") and key in MONEY_ROLE_FIELDS:
                # The quote may occur more than once (e.g. '$7,000' for two tiers); use the first
                # occurrence whose neighbouring words name the field's role.
                src, nq = normalize(pages[found_page - 1]), normalize(quote)
                at = src.find(nq)
                while at >= 0 and value is None:
                    value = parse_money_role(quote, MONEY_ROLE_FIELDS[key], src[max(0, at - 40): at],
                                             src[at + len(nq): at + len(nq) + 40])
                    at = src.find(nq, at + 1)
            elif match in ("exact", "page_fixed"):
                value = PARSERS[key](quote)
            status = "verified" if value is not None else "needs_check"
            reason = None
            if status == "needs_check":
                reason = {"empty": "no quote", "not_found": "quote not in the document",
                          "not_in_example_box": "quote not in this person's example box"}.get(
                    match, "value not parseable from the quote")
            fields[key] = {"quote": quote, "page": page, "match": match, "found_page": found_page,
                           "value": value, "status": status, "reason": reason}
    return fields
