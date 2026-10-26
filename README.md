# Same As Last Year (working name)

Choose next year's employer health plan from the plan PDFs and five answers. You get what each plan will likely cost you in a typical year and in a bad year, and what would change the answer. Every number traces back to a quoted line in your own plan document, and everything runs on your machine.

Status: Step 1 of 9 (setup and data) is done. See `RESUME.md` for current numbers and `DECISIONS.md` for why things are the way they are. The spec is `BRIEF.md`.

## Reproduce the setup

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) 0.40 or later.

```bash
make setup   # Python 3.12 env from uv.lock, plus the Gemma 4 QAT builds (about 25 GB)
make data    # MEPS files to parquet, SBC corpus and rate sheets, with manifests
make check   # TabPFN 10-row smoke test and the Gemma model check
```

## Data

- `data/meps_manifest.csv`: MEPS HC-257, HC-256 and HC-254A/D/E/F/G. Each entry has the source URL, retrieval time, SHA-256 and row count. The parquet files stay local in `data/raw/` and are not committed.
- `data/sbc/manifest.csv`: 45 public Summary of Benefits and Coverage PDFs from 11 sponsors, plus 15 premium rate sheets.
- `eval/split.json`: the frozen held-out set of 20 SBCs (all calendar-2027 documents).

## What it can't do

To be written as the product takes shape.
