# Lekka

*Do the math before you click "same as last year".*

Lekka tells you whether the cheapest 2027 health plan is safe for you.
- **When it is,** Lekka says so, and shows a typical year and a bad year (1 in 10) for every plan.
- **When your costs are hard to predict,** it tells you to look closer and does not pretend to know the winner.

Every plan number links to the line in the plan document it came from, highlighted, with the page number.

Status: Steps 1–5 of 9 are done. See `RESUME.md` for current numbers and `DECISIONS.md` for every choice. The spec is `BRIEF.md`.

## Try it
- **Static site (`docs/`):** no backend. Pick one of three real 2027 menus (Washington PEBB, Oregon PEBB, Tennessee state employees). Then either answer five questions or pick one of three real, anonymous people from the US Medical Expenditure Panel Survey. Nothing you choose leaves your browser. To view it locally:

  ```bash
  python3 -m http.server 8766 --bind 127.0.0.1 --directory docs
  ```

- **Local mode:**

  ```bash
  make app
  ```

  Then open http://127.0.0.1:8765. It is the same UI, but your five answers are worked out live on this computer's CPU. You can also upload a plan's Summary of Benefits and Coverage PDF, and it is read on this computer. Nothing is sent anywhere: the server listens on 127.0.0.1 only, the fonts are self-hosted, and the PDF reader talks to a local Ollama server.

## How it works
- **Reading plans.** `gemma4:e4b-it-qat` runs locally through Ollama.
  - It only quotes. It returns each field as `{quote, page}`, and code checks that the quote is in the PDF's text before parsing the number from it.
  - A field without a verified quote is shown as "check this", never guessed.
  - On 20 held-out 2027 plan documents, 91.9% of 260 key values were right, 3.5% wrong and 4.6% flagged for checking (`eval/results/step3_reader.json`).
- **Predicting next year's care.** TabPFN v2 is trained on MEPS Panel 28: the same adults in 2023 and 2024.
- **Pricing plans.** Each plan is priced on the real 2024 care of the survey people whose actual spending was nearest your predicted median (typical year) or 90th percentile (bad year).
- **Deciding.** The rules engine (`sameaslastyear/costs.py`) handles deductibles, copays, coinsurance, out-of-pocket limits and HSA deposits. It reproduces a real plan's printed coverage examples to the dollar.
- **The "look closer" check.** A TabPFN classifier flags people for whom the lowest premium is likely to cost more than $250 extra. On held-out people it has an AUC of 0.76 (`eval/results/step4b_lazy_safe.json`).

On 758 held-out survey adults and the three real menus, simply taking the lowest premium gave the cheapest plan to 82% of people. "Same as last year" did so for 78%. Average overpayment was $106 against $149 (`eval/results/step4_decision.json`).

## Reproduce
Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) 0.40 or later.

```bash
make setup         # Python 3.12 env from uv.lock, plus the Gemma 4 QAT builds (about 25 GB)
make data          # MEPS files to parquet, SBC corpus and rate sheets, with manifests
make check         # TabPFN smoke test and the Gemma model check
make eval-spend    # Step 2: spend quantiles, metrics, learning curve, charts
make eval-reader   # Step 3: the plan reader on the 20 held-out 2027 SBCs (cached model output)
make eval-decision # Step 4: plan-choice strategies on the real 2027 menus
make eval-lazy     # Step 4b: the lazy-safe check
make app-data      # Step 5: docs/data/lekka.json for the site
make capture       # Step 5: screenshots and GIFs in docs/media/
make test          # unit tests
```

## Data
- `data/meps_manifest.csv`: the MEPS files with source URL, retrieval time, SHA-256 and row counts. The parquet files stay local in `data/raw/`.
- `data/sbc/manifest.csv`: 45 public Summary of Benefits and Coverage PDFs from 11 sponsors, plus rate sheets and three plan documents.
- `data/menus/menus_2027.json`: the three 2027 menus. Every value is quote-verified when loaded.
- `eval/split.json`: the frozen held-out set of 20 SBCs.
- `eval/spend_split.json`: the frozen person split (2,275 train, 758 test).
- `eval/gold/heldout_gold.json`: gold values for the held-out SBCs. Claude transcribed them, not a person; the method and caveats are in `eval/README.md`.

## What it can't do
- **Only three menus, and self-only coverage.** It prices the three 2027 menus in `docs/data`, and only self-only coverage. Family coverage is in the rules engine and its tests, but not in the benchmark or the app.
- **Plans you upload are read, not priced.** A Summary of Benefits and Coverage has no premium, so in local mode Lekka shows what it read, with quotes and "check this" flags.
- **"Look closer" names no plan.** In testing, picking a plan for flagged people did worse than taking the lowest premium. The check finds about half of the people for whom the lowest premium costs more than $250 extra.
- **The care model is approximate.** Each visit takes one cost-sharing row on its full payment. Prescriptions are split into generic and brand at $50 a fill, because the survey does not say which is which. Dental, vision, home health and other spending are left out; together these are about 12% of 2024 spending. The full list is in "How sure is this?" and in `eval/results/step4_decision.json`.
- **Typical and bad years rest on 5 survey people each.** Small changes in the prediction (for example, local mode's CPU arithmetic against the precomputed grid) can swap which people are used, and move a plan's estimate by a few hundred dollars.
- **Regional plans are treated as available to everyone.** Some plans (for example Kaiser Permanente NW) are regional.
- **The reader is weakest on unusual tables.** It struggles with Tennessee's tier tables, cells with two in-network rates, and typos in the source PDF (one plan prints its deductible as "$1.800"). Such values are flagged or shown with a "check this" note.
- **TabPFN v2, not v3.5.** The newer v3.5 model needs a Prior Labs license acceptance that has not been given.
- **Gold labels are unchecked.** The evaluation's gold values were transcribed by Claude and have not been checked by a person.
