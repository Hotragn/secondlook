"""Download the MEPS public use files this project needs and save each as parquet.

File numbers were looked up on meps.ahrq.gov (download_data_files_detail.jsp) on
2026-10-25. Each Stata zip is downloaded into its own directory, its SHA-256 is
recorded, the .dta inside is read with pandas (numeric codes kept, value labels
not applied) and written to data/raw/meps/<file>.parquet together with a
variable-label table. data/meps_manifest.csv records URL, retrieval time,
hashes, rows and columns.

Run: uv run python -I scripts/fetch_meps.py
"""

import csv
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "meps"
MANIFEST = ROOT / "data" / "meps_manifest.csv"
BASE = "https://meps.ahrq.gov/mepsweb/data_files/pufs"

FILES = [
    ("HC-257", "Panel 28 Longitudinal Data File (2023-2024)", "h257"),
    ("HC-256", "2024 Full Year Consolidated Data File", "h256"),
    ("HC-254A", "2024 Prescribed Medicines File", "h254a"),
    ("HC-254D", "2024 Hospital Inpatient Stays File", "h254d"),
    ("HC-254E", "2024 Emergency Room Visits File", "h254e"),
    ("HC-254F", "2024 Outpatient Visits File", "h254f"),
    ("HC-254G", "2024 Office-Based Medical Provider Visits File", "h254g"),
]

FIELDS = [
    "file_id", "title", "url", "retrieved_utc", "zip_bytes", "zip_sha256",
    "dta_member", "rows", "cols", "parquet", "parquet_bytes",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def extract_dta(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        # Some zips also carry macOS resource forks (__MACOSX/._name.dta); skip them.
        members = [
            m for m in zf.namelist()
            if m.lower().endswith(".dta") and not Path(m).name.startswith("._")
        ]
        if len(members) != 1:
            raise RuntimeError(f"{zip_path.name}: expected one .dta, found {members}")
        # Write under a fixed local name so a crafted member path cannot escape out_dir.
        target = out_dir / Path(members[0]).name
        with zf.open(members[0]) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for file_id, title, stem in FILES:
        url = f"{BASE}/{stem}/{stem}dta.zip"
        dl_dir = RAW / "download" / stem
        dl_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dl_dir / f"{stem}dta.zip"
        stamp = dl_dir / "retrieved_utc.txt"
        if not zip_path.exists():
            download(url, zip_path)
            stamp.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        retrieved = stamp.read_text().strip()
        dta = extract_dta(zip_path, dl_dir)

        with pd.read_stata(dta, convert_categoricals=False, iterator=True) as reader:
            labels = reader.variable_labels()
            df = reader.read()

        parquet = RAW / f"{stem}.parquet"
        df.to_parquet(parquet, index=False)
        pd.DataFrame(
            {"variable": list(labels), "label": list(labels.values())}
        ).to_csv(RAW / f"{stem}_variables.csv", index=False)
        dta.unlink()

        rows.append({
            "file_id": file_id,
            "title": title,
            "url": url,
            "retrieved_utc": retrieved,
            "zip_bytes": zip_path.stat().st_size,
            "zip_sha256": sha256(zip_path),
            "dta_member": dta.name,
            "rows": len(df),
            "cols": df.shape[1],
            "parquet": str(parquet.relative_to(ROOT)),
            "parquet_bytes": parquet.stat().st_size,
        })
        print(f"{file_id:8} {len(df):>8,} rows x {df.shape[1]:>5} cols  {title}", flush=True)

    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
