# Product brief. Working name: "Same As Last Year" (the real name comes in Step 7)

## What it is
Every fall, employees choose next year's health plan from 2–4 options. Most re-pick last year's plan because comparing them is painful. This product takes the plan PDFs plus five quick answers and returns **one honest answer**:
- what each plan will likely cost *this person* next year, in a typical year and a bad (1-in-10) year
- what would change the answer

Every number traces back to a quoted line, with its page, in the person's own plan document. It runs **locally**: health and salary details never leave the machine.

## Who it's for
Any US employee choosing a 2027 employer plan during open enrollment, and anyone who has ever clicked "keep my current plan" without checking.

## Non-negotiables
1. **Secrets:** keys are in `.env` (already filled). Never print, log, echo or commit them. Read them through the environment only.
2. **No fake anything:**
   - No invented results, users, quotes or numbers.
   - Every number in the README, results files or UI copy must come from a script in this repo and be reproducible with one command.
   - If something is an approximation, label it as one where it appears.
3. **Privacy:**
   - Personal inputs (answers, uploaded PDFs) are processed locally. No network call ever carries them.
   - Network is used only to download public data and models during setup.
   - An optional hosted-model path may be used **only** on public demo documents.
4. **The model reads, the code decides:**
   - The language model may only locate and quote text.
   - Every dollar amount, percentage, date and decision is computed by deterministic code from quoted text.
   - A field without a verifiable quote is rejected and shown as "needs checking", never guessed.
5. **Quality bar:**
   - A restrained, premium UI: real typography, clear hierarchy, works on a phone.
   - No emoji soup, no generic gradient "AI app" look, no filler copy.
6. **Honest limits are written down:** README "What it can't do", plus a log of failures found.

## Architecture
```
Plan PDFs ──> text + tables (pdfplumber / pdftotext -layout)
          ──> Gemma 4 via Ollama (JSON schema): each field as {quote, page}
          ──> code: quote exists in source? value parsed from quote by regex
          ──> coverage-example check: recompute the SBC's own Peg / Joe / Mia
              examples from extracted fields and compare with the printed totals
          ──> Plan object
Five answers ──> TabPFN regressor (in-context on real MEPS people)
          ──> next-year spending quantiles
          ──> scenario engine: real MEPS people's actual next-year events (donors)
              run through each plan's rules
          ──> cost distribution per plan (premiums, HSA, taxes included)
          ──> result card + sources
```

## Data (all public; record each source URL, retrieval date and SHA-256 in a manifest)
- **MEPS (AHRQ):**
  - **HC-257** (Panel 28 longitudinal, 2023–2024) for the same-people, two-year history.
  - The **2024 event files** (prescribed medicines, office-based, outpatient, ER, inpatient) and the 2024 full-year consolidated file, for event-level costs.
  - Find the exact file numbers on meps.ahrq.gov and record them.
- **Summary of Benefits and Coverage (SBC) PDFs:**
  - 40–60 real, public SBCs from at least 10 different employers or plan sponsors (universities, states and cities publish them), across BCBS, Aetna, Cigna, UnitedHealthcare and Kaiser where possible.
  - Store them under `data/sbc/` with a `manifest.csv`.
  - **Freeze a held-out set of 20 before any scoring**, and commit the split file with its hash.
- **Premium rate sheets:** the same sponsors' public employee premium tables, so menus are real.
- **Real 2027 menus:** find 2–3 employers that have already published their **2027** open-enrollment plan documents and rates. These are the "real recent data" test cases.
- **2027 IRS limits:** HSA $4,500 self-only and $9,000 family; HDHP minimum deductible $1,750 / $3,500; out-of-pocket maximum $8,700 / $17,400. Verify against an official IRS source and cite it in the code.

## Depth requirements (each needs its test or eval, saved under `eval/results/` as JSON plus a chart)

### A. Plan reader (Gemma 4, local)
- **Schema fields:**
  - plan name and network type
  - overall deductible (individual and family; embedded or not)
  - out-of-pocket limit (individual and family)
  - services covered before the deductible
  - for every "Common Medical Event" row: in-network cost (copay or coinsurance, and whether the deductible applies)
  - HSA eligibility
  - the three coverage examples (Peg, Joe, Mia): total example cost, deductibles, copays, coinsurance, limits, and the total the member pays
- **Rules:** every field must come with a verbatim quote and page. Code verifies the quote is in the extracted text (whitespace-normalised) and parses the value from the quote.
- **Coverage-example reconciliation:**
  - Recompute what Peg, Joe and Mia pay from the extracted fields, and compare with the printed totals.
  - If CMS's service-level inputs for the examples can be found, implement them exactly. Otherwise use a documented tolerance check and say so.
- **Eval on the frozen held-out set:**
  - Gold values are transcribed from each PDF and committed. Note the method in `eval/README`.
  - Metrics: field accuracy, coverage-example reconciliation rate, seconds per PDF and peak memory.
  - Compare `gemma4:e2b`, `gemma4:e4b`, and a larger Gemma 4 if it fits in 24 GB.
  - Keep a failure log.

### B. Spend predictor (TabPFN)
- **Population:** MEPS Panel 28 adults aged 18–64 with private or employer coverage in both years. Document the exact filter and N.
- **Data:** year-1 features (age, sex, region, self-rated health, chronic-condition flags, utilisation counts, year-1 total spending), predicting year-2 total spending.
- **Model:** `TabPFNRegressor(...).predict(X, output_type="quantiles", quantiles=[...])`. If the training set exceeds TabPFN's context, use nearest-neighbour context selection, and document it.
- **Baselines:**
  - "same as last year"
  - age-and-sex-group median
  - HistGradientBoosting (quantile loss)
  - linear quantile regression
- **Metrics:** pinball loss at 0.1, 0.5 and 0.9, plus interval coverage, on a person-level held-out split (fixed seed, split file committed).
- Report where TabPFN loses, too.

### C. Decision benchmark (the headline)
- **Menus:** real sponsor menus (SBC-derived plans plus real premiums).
- **The truth:** for each held-out MEPS person, the cost of each plan in year 2 is computed from their **actual year-2 events** (event-level cost sharing). If event-level isn't feasible for some category, fall back to an approximation **and label it**.
- **Strategies:**

  | Strategy | How it picks |
  |---|---|
  | Lowest premium | The cheapest monthly price |
  | Lowest deductible | The smallest deductible |
  | Same as last year | The plan that would have been best given year-1 events |
  | TabPFN expected cost | Lowest cost averaged over predicted outcomes |
  | TabPFN risk-capped | Lowest cost subject to a P90 cap |
  | Oracle | The truly cheapest plan (the reference) |

- **Metrics:** share of people for whom each strategy picked the truly cheapest plan, mean regret in dollars and P90 regret in dollars.

### D. Rules engine
- Deductible, coinsurance and copays (deductible applies or not), out-of-pocket cap, embedded family deductible, preventive at $0.
- HSA: employee contribution, employer seed, and tax savings (federal, state and payroll when paid through payroll; the marginal rate is an input).
- 2027 limits as constants, each with a source comment.
- Unit tests for every rule, including hand-worked examples and the SBC coverage examples.

### E. App
- A local web app, with flows: upload PDFs → review extracted fields (quotes highlighted, "needs checking" states) → five questions → result card → sources.
- **The result card shows:**
  - the lowest expected cost and its difference from the current plan
  - the bad-year (P90) cost of each plan
  - what would flip the answer
  - links to the quoted lines
- **A static demo mode** (no backend) with the real 2027 menus precomputed, so anyone can open it.
- **Friend test on real recent data:** run 3 real anonymous MEPS people (a healthy single adult, a family with a child, someone with a chronic condition) against a real 2027 menu. Save their result cards and a write-up.

### F. Observability (optional, only if it does real work)
Sentry tracing with personal data off. Keep it only if traces reveal and fix a real bug or bottleneck, and log that story in `DECISIONS.md`.

## Repo hygiene
- Git repo here; a **private** GitHub repo under `Hotragn` (temporary name; renamed in Step 7).
- Small, clear commits.
- `make setup`, `make eval` and `make app` (or equivalents), plus `README.md` with reproduction steps.
- `DECISIONS.md` (numbered decisions with the reason) and `RESUME.md` (current status, numbers, risks, next step).
- Ship a complete, honest vertical slice first, then deepen.

## Steps (do only the step you are asked for; end each with a STATUS block in RESUME.md)
1. Setup and data: environment, models, MEPS files, SBC corpus and manifest, held-out split, 2027 menus, private repo.
2. Spend predictor and benchmark B.
3. Plan reader and eval A.
4. Rules engine, scenario engine and benchmark C (the headline numbers).
5. App, static demo and friend test.
6. Hardening: tests, `make eval` reproducibility, README, captioned screenshots, short GIFs of the flows.
7. Name, logo and brand system: unique and human, no AI slop. Apply them to the app and README.
8. Launch video, following the skill at `/Users/hotragn/Documents/Claude Code/tools/brag/skills/brag-slim/SKILL.md` (read it and follow it; do not install anything globally).
9. Post assets: a numbers sheet (every headline number with the command that produced it) and final captioned screenshots and GIFs.
