"""Transcription aid for the held-out gold labels (eval/README.md).

Prints, for each held-out SBC, the source text around the fields being
transcribed: the deductible and out-of-pocket answers, six medical-event rows,
and the three coverage-example totals. It only searches the extracted text; no
model is involved. Values were typed into eval/gold/heldout_gold.json by hand
from this output, with page images checked where the text was unclear.

Run: uv run python -I scripts/gold_snippets.py [doc_id ...]
"""

import csv
import json
import re
import sys
from pathlib import Path

from sameaslastyear.reader import load_pages, normalize

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = {
    "deductible": r"what is the overall deductible\?",
    "oop": r"what is the out-of-pocket limit for this plan\?",
    "primary_care": r"primary care visit to treat an injury or illness",
    "specialist": r"specialist visit",
    "urgent_care": r"urgent care",
    "emergency_room": r"emergency room (care|services)",
    "generic_drugs": r"generic drugs",
    "inpatient_facility": r"facility fee \(e\.g\., hospital room\)",
}


def main() -> None:
    split = json.loads((ROOT / "eval" / "split.json").read_text())
    rows = {r["doc_id"]: r for r in csv.DictReader((ROOT / "data" / "sbc" / "manifest.csv").open())}
    ids = sys.argv[1:] or [d["doc_id"] for d in split["held_out"]]
    for doc_id in ids:
        pages = load_pages(ROOT / rows[doc_id]["path"])
        text = normalize(" ".join(pages))
        print(f"##### {doc_id}")
        for name, pat in ANCHORS.items():
            m = re.search(pat, text, re.I)
            width = 230 if name in ("deductible", "oop") else 150
            print(f"  {name}: {text[m.end(): m.end() + width] if m else '<not found>'}")
        totals = re.findall(r"The total (Peg|Joe|Mia) would pay is \$\s?([\d,]+)", text)
        print(f"  totals: {totals}")


if __name__ == "__main__":
    main()
