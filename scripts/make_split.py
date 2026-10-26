"""Freeze the SBC dev / held-out split for the plan-reader eval (eval A).

Rule (no randomness): an SBC is held out when its printed coverage period starts
on 1 January 2027. Those 20 documents come from four sponsors (Washington PEBB,
Oregon PEBB, State of Tennessee, CalPERS) that have no document in the dev set,
so the held-out score measures unseen sponsors on next year's documents. Each
entry carries the file's SHA-256 so the split is tied to exact bytes.

Run: uv run python -I scripts/make_split.py
Writes eval/split.json and prints its SHA-256.
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sbc" / "manifest.csv"
OUT = ROOT / "eval" / "split.json"
HELD_OUT_SIZE = 20


def is_2027(period: str) -> bool:
    start = period.split("-")[0]
    month, day, year = start.split("/")
    return (int(month), int(day), int(year) % 100) == (1, 1, 27)


def main() -> None:
    with MANIFEST.open() as f:
        sbcs = [r for r in csv.DictReader(f) if r["doc_type"] == "sbc"]

    held, dev = [], []
    for r in sorted(sbcs, key=lambda r: r["doc_id"]):
        entry = {"doc_id": r["doc_id"], "sponsor": r["sponsor"], "sha256": r["sha256"]}
        (held if is_2027(r["coverage_period"]) else dev).append(entry)

    assert len(held) == HELD_OUT_SIZE, f"held-out has {len(held)} SBCs, expected {HELD_OUT_SIZE}"
    overlap = {e["sponsor"] for e in held} & {e["sponsor"] for e in dev}
    assert not overlap, f"sponsors on both sides: {overlap}"

    split = {
        "rule": "held_out = SBCs whose printed coverage period starts 1/1/2027; dev = all other SBCs",
        "frozen_before_scoring": True,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "counts": {"held_out": len(held), "dev": len(dev)},
        "held_out_sponsors": sorted({e["sponsor"] for e in held}),
        "dev_sponsors": sorted({e["sponsor"] for e in dev}),
        "held_out": held,
        "dev": dev,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(split, indent=2) + "\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT.relative_to(ROOT)}: {len(held)} held out, {len(dev)} dev")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
