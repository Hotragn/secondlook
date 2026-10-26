# Decisions

Numbered, with the reason. Newest at the bottom.

## Step 1: setup and data (2026-10-25)

**1. Python environment: uv, Python 3.12.15, locked in `uv.lock`.**
uv and poppler were installed through the user's Homebrew (`~/.homebrew`); uv manages its own CPython 3.12.15. Runtime dependencies are in `pyproject.toml`; `pytest` is in the `dev` group. `uv.lock` pins all 96 resolved packages (tabpfn 9.2.0, torch 2.14.1, pandas 3.1.0, scikit-learn 1.9.1, pdfplumber 0.11.10, pyarrow 26.0.0, fastapi 0.162.0).

**2. TabPFN: smoke test on v2 weights, because v2.5 and later need a Prior Labs license login.**
tabpfn 9.2.0 defaults to model v3.5. Versions 2.5, 2.6, 3 and 3.5 call `ensure_license_accepted()` before downloading: a one-time browser login at ux.priorlabs.ai to accept the license, with the token cached in `~/.cache/tabpfn/auth_token` or set as `TABPFN_TOKEN`. That is an account action for the owner, so it was not done here. v2 downloads without a login, so `scripts/smoke_tabpfn.py` uses `ModelVersion.V2`. It passes on Apple MPS: 10 rows, quantiles 0.1/0.5/0.9, monotone and finite. Step 2 should use v3.5 if the license is accepted, and v2 otherwise; record which one ran.

**3. The Ollama server was restarted.**
The running `ollama serve` (up 7.5 days, orphaned from an earlier shell) could not load any model. The llama.cpp runner failed with "failed to create library" and the MLX runner with "Unable to load kernel vn_copyint64int32". A fresh server on another port loaded the same model in 2.3 s, so the old process was stopped and a new one started (`nohup ollama serve`, log in `~/.ollama/logs/server.log`).

**4. The plan reader uses the GGUF QAT Gemma 4 builds, not the default `gemma4:e2b` / `gemma4:e4b` tags.**
Both default tags were pulled as the brief asks. On Apple Silicon they resolve to MLX nvfp4 builds (7.5 GB and 9.5 GB). They run, but reject a JSON-schema `format` with "structured output is unavailable", and the reader depends on schema-constrained output. The QAT builds (Google's quantisation-aware 4-bit weights) run on the llama.cpp runner and honour the schema: `gemma4:e2b-it-qat`, `gemma4:e4b-it-qat`, `gemma4:26b-a4b-it-qat`. The evidence is in `eval/results/step1_model_check.json` (`scripts/check_models.py`).

**5. The larger Gemma 4 that fits in 24 GB is `gemma4:26b-a4b-it-qat`.**
At a 16k context it is 15 GB resident, 100% on the GPU, at 37.5 tokens/s (mixture of experts, about 4B active parameters), with no new swap. The Metal budget on this M4 is 19.07 GB (`torch.mps.recommended_max_memory()`); Ollama reports 17.8 GiB. `gemma4:31b` (19–20 GB of weights on the Ollama library page) was not pulled: the weights alone exceed that budget, so it would need CPU offload or a raised wired-memory limit, which is a system setting. `gemma4:12b` is unnecessary with the 26B fitting. Peak memory under real SBC prompts is measured in Step 3.

**6. MEPS: the exact files and how they are stored.**
Looked up on meps.ahrq.gov on 2026-10-25:
- HC-257: Panel 28 Longitudinal Data File. Year 1 = 2023, year 2 = 2024 (`TOTEXPY1` / `TOTEXPY2`).
- HC-256: 2024 Full Year Consolidated Data File.
- HC-254A: 2024 Prescribed Medicines.
- HC-254D: 2024 Hospital Inpatient Stays.
- HC-254E: 2024 Emergency Room Visits.
- HC-254F: 2024 Outpatient Visits.
- HC-254G: 2024 Office-Based Medical Provider Visits.

The Stata zips are read with `convert_categoricals=False`, so the MEPS numeric codes (including negative reserved codes) are kept as numbers. Each file becomes one parquet file plus a variable-label CSV under `data/raw/meps/` (gitignored). `data/meps_manifest.csv` (committed) holds the URL, retrieval time, zip SHA-256, rows and columns.

Not downloaded, but available if Step 4 needs them for event-level truth: HC-254B (dental), HC-254C (other medical expenses), HC-254H (home health), HC-254I (event appendix) and HC-255 (2024 conditions).

**7. SBC corpus: 45 SBCs from 11 public sponsors, limited to sources a script can fetch.**
The 11 sponsors are 8 public-employer programs, 1 city and 2 university systems. Every file is listed in `data/sbc/sources.csv` and re-downloadable with `scripts/fetch_sbc.py`, which writes `data/sbc/manifest.csv` with URL, retrieval time, SHA-256, pages, the printed coverage period, and checks for the federal SBC template. All 45 carry the template title and all three coverage examples (Peg, Joe, Mia).

Carriers: Kaiser 11, BCBS plans 20 (BCBST/Cigna 6, Horizon 5, Regence/UMP 3, Blue Shield of California 2, Highmark 2, Anthem 1, BCBSTX 1), Aetna 5, UnitedHealthcare 4, Cigna 3, Moda 2.

Skipped because they block scripted downloads (HTTP 403): Included Health (CalPERS PPOs), benefitoptions.az.gov, hr.umich.edu.

Oregon PEBB: only the full-time variants were kept. New Jersey: only the medical SBCs were kept; their drugs are in separate Rx SBCs.

**8. The held-out set is the 20 calendar-2027 SBCs, frozen before any scoring.**
Rule: an SBC is held out if its printed coverage period starts on 1/1/2027. That gives exactly 20 SBCs, from four sponsors (Washington PEBB 9, State of Tennessee 6, Oregon PEBB 4, CalPERS 1). None of these sponsors appears in the 25-document dev set.

Why not a random split: SBCs from one sponsor and carrier share a layout, so a random split would leak templates into the test. This rule scores the reader on unseen sponsors and on next year's documents, which is the real use. The split is deterministic (`scripts/make_split.py`), and each entry carries the PDF's SHA-256.

- `eval/split.json` SHA-256: `4724245afe2d78030e8a22dfaa2845129c3bd378322d380d6ca544b5d29823c6`

**9. The real 2027 menus are Washington PEBB, Oregon PEBB and Tennessee; CalPERS is partial.**

| Sponsor | 2027 SBCs | 2027 premiums |
|---|---|---|
| Washington PEBB | 9 (UMP Classic, Select, CDHP; Kaiser WA Classic, SoundChoice, Value, CDHP; Kaiser NW Classic, CDHP) | `50-0626` employee monthly premiums |
| Oregon PEBB | 4 (Moda PPO, Moda Coordinated Care, Kaiser Traditional, Kaiser Deductible; full-time) | 2027 rates, all tables |
| Tennessee | 6 (Premier, Standard, Limited, CDHP, Local CDHP, Copay) | State and higher-education active premiums |
| CalPERS | 1 (Blue Shield Trio, watermarked "pending regulatory approval") | in-state, out-of-state and region 1–3 tables |

These menus are all in the held-out set. Using them for benchmark C and the friend test is fine, because nothing is tuned on them.

**10. Rate sheets are kept for every sponsor; one year mismatch is recorded.**
There are 15 rate files. Delaware's posted rate sheet is FY2027 (from July 2026), but its posted SBCs are FY2026, so Delaware's menu years do not match. New Jersey's premiums come from an Aon presentation to its benefits commission, not a member-facing sheet. Colorado's and Georgia's premiums sit inside their benefits guides. Each caveat is in the `notes` column of the manifest.

## Step 2: spend predictor (2026-10-26)

**11. Population: 3,033 privately covered adults from MEPS Panel 28.**
Filter applied to HC-257 (`scripts/spend_data.py`):

| Step | People |
|---|---|
| HC-257 Panel 28 persons | 8,872 |
| In both 2023 and 2024 (`YEARIND = 1`) | 8,709 |
| Positive longitudinal weight (`LONGWT > 0`) | 8,709 |
| Aged 18–64 at the end of 2023 and of 2024 (`AGEY1X`, `AGEY2X`) | 4,673 |
| Any private coverage in 2023 and in 2024 (`INSCOVY1 = INSCOVY2 = 1`) | **3,033** |

"Any private" is MEPS's code 1 for `INSCOV`; it includes employer and non-group private plans held at any time in the year. A stricter subgroup, employer or union group coverage in all 24 months (`PEG{month}Y1/Y2 = 1`), has 2,173 people. It is reported as a breakdown, not used as the population.

The five year-2 components kept for Step 4 (office, outpatient, ER, inpatient, Rx) sum to 88.95% of year-2 total spending in this population. The rest is dental, home health, vision and other.

**12. Spend split: 25% test by household, fixed seed, frozen before scoring.**
`StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20261026)`, fold 0 is the test set. Grouping by household (`DUID`) keeps family members, whose spending is correlated, on one side. Stratifying by year-1 spending tier balances the tiers.

| | People | Households | Zero | Low | Mid | High |
|---|---|---|---|---|---|---|
| Train | 2,275 | 1,466 | 223 | 684 | 684 | 684 |
| Test | 758 | 499 | 74 | 228 | 228 | 228 |

Tiers use year-1 total spending: zero = $0; low = $1–$1,408; mid = $1,409–$5,184; high = over $5,184. The cut-offs are terciles of positive year-1 spending in the population.

- `eval/spend_split.json` SHA-256: `d9577f32d9f4a365f1e4f5cd2f88f0ad796dfe87423a4d0862d1076edc4039aa`

**13. Modelling choices, fixed before scoring.**
- **Features** (26, all from 2023):
  - age, female, region (three dummies)
  - self-rated physical and mental health, from round 2 or else round 1. Round 3 is skipped because it is interviewed in 2024.
  - 12 chronic-condition flags and their count
  - office, outpatient, ER, inpatient and Rx counts
  - total spending
- **Target:** 2024 total spending (`TOTEXPY2`). The learned models fit `log1p(spend)` and predictions are mapped back with `expm1`. Quantiles survive any increasing transform, so this changes nothing about what is predicted. Spend and count features enter as `log1p`.
- **Quantiles:** 0.1, 0.25, 0.5, 0.75, 0.9. Where a baseline's quantiles cross, each row is sorted.
- **No hyperparameter tuning** for any model; library defaults with fixed seeds.
- **TabPFN:** v3.5 is the main model if the Prior Labs token in `.env` passes the license check, with v2 as an extra row; otherwise v2 alone. The training set (2,275 rows) fits in context, so no nearest-neighbour context selection is needed.
- **Baselines:**
  - same as last year (every quantile = 2023 spending)
  - age-band × sex training quantiles (18–34, 35–49, 50–64; the 0.5 row is the group median)
  - `HistGradientBoostingRegressor(loss="quantile")`, one model per quantile
  - linear `QuantileRegressor(alpha=0, solver="highs")` on standardised features with median imputation
- **Metrics on the test set:**
  - pinball loss at 0.1, 0.5 and 0.9, in dollars
  - P10–P90 coverage (nominal 80%) and interval width
  - all of the above overall, by year-1 tier, and for the 24-month employer-coverage subgroup
- **Uncertainty:** 95% CIs from 1,000 bootstrap resamples of test households, plus paired differences against the main TabPFN model.
- **Weighted versions** use `LONGWT`. The bootstrap ignores MEPS strata and PSUs, so its CIs are an approximation.

**14. TabPFN v3.5 did not run; v2 is the main TabPFN for these results.**
The `TABPFN_TOKEN` in `.env` is valid: Prior Labs' `verify_token` returns true. But the account that owns it has not accepted the TabPFN 3.5 license (`tabpfn-3-5-license-v1.0`), so `check_license_accepted` returns false and tabpfn refuses to download the v3.5 weights. Accepting the license is the account owner's action, on the Licenses tab at ux.priorlabs.ai.

`scripts/spend_eval.py` tries v3.5 first on every run. Once the license is accepted, `make eval-spend` makes v3.5 the main model and keeps v2 as an extra row, with no code change.

How the token is handled:
- Only `TABPFN_TOKEN` is read from `.env` (`dotenv_values`), never printed.
- `TABPFN_NO_BROWSER=1` stops tabpfn from opening a login page.
- tabpfn's `save_token` is disabled, so the token is not copied in plain text to `~/.cache/tabpfn/auth_token`.
- The token goes only to api.priorlabs.ai for the license check. MEPS data never leaves the machine.

**15. Learning curve (Step 2b): design fixed before running.**
- **Models:** TabPFN v2, gradient boosting and linear quantile regression.
- **Training sizes:** 100, 200, 400, 800 and the full 2,275 people.
- **Draws:** 5 random draws without replacement per size; seeds are `20261026 + 1000 × size index + draw`, all recorded in the JSON. At the full size every draw holds all people, so only the model seeds differ.
- **Test set:** the frozen Step 2 test set at every size.
- **CIs:** household bootstrap of the per-person losses averaged over the 5 draws. The spread across draws is reported separately.
- **The question's rule:** a model "reaches full-data accuracy" at the smallest size whose mean loss is within 5% of its own full-data loss. This was chosen before seeing results.

The shared Step 2 code moved into the `sameaslastyear` package (`sameaslastyear/spend.py`), installed in editable mode by `uv sync`. `make eval-spend` gave identical metrics before and after the move.

## Step 3: plan reader (2026-10-26)

**16. Reader design: the model quotes, code verifies and parses.**
- **Page text:** pypdfium2 text in content-stream order. On the dev SBCs this keeps each table cell's text contiguous ("30% coinsurance After Deductible"). pdfplumber's table extraction split or merged cells differently from issuer to issuer, and line-ordered text interleaves wrapped cells.
- **Model calls:** the SBC is sent in three calls, each with only its pages, found from fixed SBC headings:
  - important questions: 9 fields
  - common medical events: all 30 standard rows
  - coverage examples: 18 example lines plus the 4 plan parameters in Peg's box
- **Model settings:** Ollama JSON schema `{quote, page}` per field, temperature 0, seed 0, no thinking, 16k context. The longest section is about 4k tokens.
- **Verification:** the quote must appear in the cited page's text after normalising whitespace, quotes and dashes. If it appears only on another page, it is accepted as `page_fixed` and counted separately. Coverage-example quotes must also sit inside that person's example box. This matters because "Deductibles $1,500" can appear in all three boxes.
- **Parsing:** values come from the quote with regular expressions. The model never types a number. For deductible and out-of-pocket fields, code may read the role word (individual or family) from the document text next to the verified quote; the amount itself always comes from the quote.
- **needs_check:** anything without a verified, parseable quote, including a plan name that is just the SBC title. Nothing is guessed.

**17. Coverage examples are reconciled, not recomputed.**
One search did not find CMS's service-level inputs for the Peg, Joe and Mia examples, so each example is checked for consistency within the SBC instead (`sameaslastyear/rules.py`, $1 tolerance):
- **Arithmetic:** deductibles, copayments, coinsurance and limits add up to the printed total.
- **Bounds:** the deductible, out-of-pocket and total-cost limits hold.
- **Parameters:** the example box's deductible, specialist cost and hospital cost sharing agree with the fields read from the summary and the table.

This is weaker than a claim-level recompute, and it is labelled as such.

**18. Gold labels were committed before any held-out reader run; the one leak is disclosed.**
- **Who made them:** Claude transcribed the 260 held-out gold values (`eval/gold/heldout_gold.json`) from extracted text. No person has checked them. 25 values are flagged ambiguous, and accuracy is reported with and without them.
- **When:** they were committed in `45f9098`, before any held-out reader output existed.
- **The leak:** transcribing them exposed held-out cell patterns. After that commit, the reader was changed only for failures seen on dev documents.

**19. Parser changes made during development, each traced to a dev-set failure.**
Re-scoring the same cached `gemma4:e4b-it-qat` output on the first 5 dev SBCs, verified fields went from 239 of 305 to 289 of 305 after these fixes. With all fixes, e4b verifies 406 of 427 fields (95.1%) on the 7 dev SBCs, and 15 of 21 coverage examples reconcile. On 2 further dev SBCs, e2b verified 86.9% and the 26B 95.1%. Each change, and the dev failure behind it:
- **Role words before the amount.** NJ Aetna prints "Individual $8,480 / Family $16,960".
- **"You" and "You + Family" tiers.** Georgia SHBP prints "$3,500 You $7,000 You + Family".
- **Repeated quotes.** "$7,000" appears for two Georgia tiers, so every occurrence of a quote is checked.
- **Bare "$0" deductible.** It means $0 for families too (NJ Horizon HMO).
- **"Out-of- Network" with a space.** Delaware Highmark spells it that way.
- **Forward labels.** A role label followed by a colon ("Family:", "self-only coverage:") belongs to the next amount; a colon elsewhere does not (Albuquerque).
- **Example-box boundaries.** Boxes are cut at each "The total X would pay is $N" line, because Albuquerque, Delaware Highmark, NJ Aetna and UC Care print all three headings before the boxes.
- **Example-box parameters.** These can be copays ("Hospital (facility) copayment $0", "$100 per day"), not only coinsurance.

**What dev showed the reconciliation catching:**
- UC Care's coverage examples assume its Tier 2 network ($500 deductible, 30% specialist), while the summary fields hold Tier 1 ($0, $30). All three examples are therefore flagged as mismatches.
- Albuquerque prints the three "Total Example Cost" amounts in one row, so they cannot be tied to a person's box and stay needs_check.

No prompt wording changed after the first dev run.

**20. The reader is frozen before the held-out run.**
- **What is frozen:** the reader, parsers, rules and prompt fingerprint, committed just before `make eval-reader` first ran.
- **The run:** the held-out run is the only one, with all three models in sequence so timings do not overlap. Its raw output is committed, so re-running the target only re-scores.

## Step 4: rules engine and benchmark C (2026-10-26)

**21. The 2027 limits are checked against primary sources and cited in `sameaslastyear/costs.py`.**
- **IRS Rev. Proc. 2026-24 (rp-26-24.pdf):**
  - HSA limit $4,500 self-only and $9,000 family
  - HDHP minimum deductible $1,750 / $3,500
  - HDHP out-of-pocket maximum $8,700 / $17,400
- **Catch-up:** $1,000, from 26 U.S.C. § 223(b)(3)(B), not from the Rev. Proc.
- **CMS 2027 PAPI guidance (2026-01-29):** ACA maximum out-of-pocket $12,000 self-only and $24,000 other.
- **Embedded rule:** under the ACA's embedded self-only rule, no person in family coverage pays more than $12,000.

**22. Rules engine (`sameaslastyear/costs.py`).**
- **Preventive care:** $0.
- **Deductible:** embedded or aggregate for families. Explicit SBC wording decides whether the deductible applies. Otherwise HSA plans apply it to everything but preventive care, coinsurance waits for the deductible, and copays do not.
- **Copays:** per visit, or per day with per-admission or per-year caps.
- **Coinsurance:** with an optional per-fill cap.
- **Out-of-pocket:** medical limit, plus an optional separate prescription limit. Not-covered care does not count toward it.
- **HSA:** the employer seed counts as cash; employee tax savings are federal + state + 7.65% FICA when paid through payroll, capped at the IRS limit net of the seed.
- **Tests:** 20 unit tests in `tests/test_costs.py`, including three SBC coverage examples (Georgia UHC HDHP, a dev SBC) that the engine reproduces to the dollar: Peg $4,712, Joe $4,070, Mia $1,900.

**23. Real 2027 menus (`data/menus/menus_2027.json`, loaded and verified by `sameaslastyear/menus.py`).**
16 plans in three menus:
- Washington PEBB: 9 plans
- Oregon PEBB: 4 plans
- Tennessee state: Premier, Standard and CDHP. The other three TN SBCs are not on the state active-employee rate sheet.

Where the values come from:
- **Gold labels:** deductible, out-of-pocket limit, and the primary care, specialist, ER, generic and inpatient rows.
- **Quoted values:** everything else carries a quote that is re-verified against the PDF text on every load, with each number checked to appear in its quote. That covers premiums, per-day hospital copays and caps, per-fill caps, separate prescription limits, outpatient surgery, brand drugs and HSA deposits.

Three documents were added to the corpus for this:
- the UMP CDHP 2027 certificate (WA state HSA deposit, $700.08)
- the TN 2027 benefits guide (state HSA deposit, $500)
- the OR 2027 enrollment guide (employee share: 5% for Moda PPO and Kaiser Traditional, 1% otherwise)

Washington's $700.08 deposit is applied to all three PEBB CDHPs; that is an approximation, because the KP documents don't state it.

**24. Benchmark C design, fixed before the first run.**
- **People:** the 758 Step 2 test people, self-only coverage. Every plan is assumed available to everyone, although Kaiser NW, for example, is regional.
- **Truth:** premiums plus each person's actual 2024 MEPS events priced under the plan, less any employer HSA deposit.
- **Same as last year:** prices their 2023 events. The 2023 event files were added to `scripts/fetch_meps.py`: HC-248A, D, E, F and G.
- **TabPFN strategies:** use the Step 2 v2 quantiles, reused rather than retrained; the GPU was busy with the reader run. Each predicted percentile maps to the 5 training people whose actual 2024 spending is nearest it, and their real events are priced.
  - Expected cost is the mean over the five percentile scenarios.
  - Risk-capped means lowest expected cost among plans whose 90th-percentile cost is within $1,000 of the lowest.
- **Uncertainty:** 1,000 household-bootstrap resamples, 200 for the 90th-percentile regret.
- **Approximations:** the event-to-row mapping approximations are listed in the script docstring and the JSON. Dental is excluded as not a medical-plan cost. The modelled events cover 88.4% of the test people's 2024 spending.
- **No retuning:** after seeing results, the strategies were not tuned on this test set.
