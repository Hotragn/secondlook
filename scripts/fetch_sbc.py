"""Download the public SBC corpus and premium rate sheets listed in data/sbc/sources.csv.

Each file is saved as data/sbc/<pdf|rates>/<doc_id>.pdf. For every file the
manifest records the source URL, retrieval time, SHA-256, size and page count.
For SBCs it also checks the text for the federal SBC template markers, parses
the printed coverage period, and notes whether the Peg / Joe / Mia coverage
examples are present. A file that is not a PDF stops the run.

Run: uv run python -I scripts/fetch_sbc.py
Writes data/sbc/manifest.csv
"""

import csv
import hashlib
import re
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
SBC_DIR = ROOT / "data" / "sbc"
SOURCES = SBC_DIR / "sources.csv"
MANIFEST = SBC_DIR / "manifest.csv"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

FIELDS = [
    "doc_id", "doc_type", "sponsor", "sponsor_type", "carrier", "plan", "year",
    "url", "retrieved_utc", "sha256", "bytes", "pages", "path",
    "sbc_template", "coverage_period", "coverage_examples", "notes",
]

DATE = r"\d{1,2}/\d{1,2}/\d{2,4}"
PERIOD = re.compile(rf"coverage\s+period:?\s*({DATE})\s*[-–—]\s*({DATE})", re.I)
# The 2020+ SBC template's three coverage examples (Mia's apostrophe is often curly).
EXAMPLES = ("peg is having a baby", "managing joe", "simple fracture")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    with tmp.open("rb") as f:
        if f.read(5) != b"%PDF-":
            tmp.unlink()
            raise RuntimeError(f"not a PDF: {url}")
    tmp.rename(dest)


def pdf_text(path: Path) -> tuple[int, str]:
    pdf = pdfium.PdfDocument(path)
    try:
        pages = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
    finally:
        pdf.close()
    return len(pages), " ".join(" ".join(pages).split())


def main() -> None:
    previous = {}
    if MANIFEST.exists():
        with MANIFEST.open() as f:
            previous = {r["doc_id"]: r for r in csv.DictReader(f)}

    with SOURCES.open() as f:
        sources = list(csv.DictReader(f))

    rows = []
    for src in sources:
        sub = "pdf" if src["doc_type"] == "sbc" else "rates"
        dest = SBC_DIR / sub / f"{src['doc_id']}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            download(src["url"], dest)
        digest = sha256(dest)
        prev = previous.get(src["doc_id"], {})
        if prev.get("sha256") == digest:
            retrieved = prev["retrieved_utc"]
        else:
            retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        pages, text = pdf_text(dest)
        low = text.lower()
        period = PERIOD.search(text)
        row = {
            "doc_id": src["doc_id"],
            "doc_type": src["doc_type"],
            "sponsor": src["sponsor"],
            "sponsor_type": src["sponsor_type"],
            "carrier": src["carrier"],
            "plan": src["plan"],
            "year": src["plan_year"],
            "url": src["url"],
            "retrieved_utc": retrieved,
            "sha256": digest,
            "bytes": dest.stat().st_size,
            "pages": pages,
            "path": str(dest.relative_to(ROOT)),
            "sbc_template": "",
            "coverage_period": "",
            "coverage_examples": "",
            "notes": src["notes"],
        }
        if src["doc_type"] == "sbc":
            row["sbc_template"] = "summary of benefits and coverage" in low
            row["coverage_period"] = f"{period.group(1)}-{period.group(2)}" if period else ""
            row["coverage_examples"] = sum(e in low for e in EXAMPLES)
        rows.append(row)
        print(
            f"{row['doc_id']:42} {pages:>3}p {row['bytes']:>9,}B "
            f"sbc={row['sbc_template']!s:5} period={row['coverage_period']:23} "
            f"examples={row['coverage_examples']}",
            flush=True,
        )

    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    sbcs = [r for r in rows if r["doc_type"] == "sbc"]
    print(
        f"wrote {MANIFEST.relative_to(ROOT)}: {len(sbcs)} SBCs from "
        f"{len({r['sponsor'] for r in sbcs})} sponsors, "
        f"{len(rows) - len(sbcs)} rate sheets"
    )


if __name__ == "__main__":
    sys.exit(main())
