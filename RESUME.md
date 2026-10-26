# Resume

Where the build stands, for whoever picks it up next. The spec is `BRIEF.md`; the reasons behind choices are in `DECISIONS.md`. Each step ends with a STATUS block below.

## Next step

Step 4: rules engine, scenario engine and benchmark C, the headline numbers.

Open items:
- **TabPFN 3.5:** once the Prior Labs account accepts the TabPFN 3.5 license, re-run `make eval-spend` to get v3.5 results (DECISIONS 14).
- **2023 event files:** Step 4's "same as last year" strategy needs them; they are not downloaded yet.

---

## STATUS: Step 1, setup and data (2026-10-26, 00:57 UTC)

### Done

**Environment**
- uv 0.13.5 and poppler 26.10.0 (`pdftotext`) installed via Homebrew.
- CPython 3.12.15 in `.venv` (1.2 GB). `uv.lock` pins 96 packages.

**TabPFN 9.2.0**
- The smoke test passes on v2 weights on Apple MPS (`scripts/smoke_tabpfn.py`): 10 rows, quantiles 0.1/0.5/0.9, monotone and finite.
- First run took 8.9 s, including the 44 MB weight download. A warm run takes 0.6 s.

**Gemma 4 on Ollama 0.40.2.** Measured at a 16k context by `scripts/check_models.py`; results in `eval/results/step1_model_check.json`.

| Model | Download | Resident | Speed | Schema output |
|---|---|---|---|---|
| `gemma4:e2b` (MLX) | 7.5 GB | – | – | rejected |
| `gemma4:e4b` (MLX) | 9.5 GB | – | – | rejected |
| `gemma4:e2b-it-qat` | 4.3 GB | 3.8 GB | 58.1 tok/s | valid, verbatim quote |
| `gemma4:e4b-it-qat` | 6.1 GB | 5.6 GB | 32.7 tok/s | valid, verbatim quote |
| `gemma4:26b-a4b-it-qat` | 15 GB | 15 GB | 37.5 tok/s | valid, verbatim quote |

All three QAT builds run 100% on the GPU.

**Larger Gemma 4 in 24 GB: yes, `gemma4:26b-a4b-it-qat`.** The Metal budget is 19.07 GB, and the model added no new swap. `gemma4:31b` was not attempted: its 19–20 GB of weights exceed the budget on their own.

**MEPS** (`data/meps_manifest.csv`; parquet in `data/raw/meps/`, not committed)

| File | Contents | Rows × columns |
|---|---|---|
| HC-257 | Panel 28 longitudinal, 2023–2024 | 8,872 × 2,837 |
| HC-256 | 2024 full-year consolidated | 19,140 × 1,615 |
| HC-254A | 2024 prescribed medicines | 204,550 × 64 |
| HC-254D | 2024 inpatient stays | 1,912 × 50 |
| HC-254E | 2024 ER visits | 4,351 × 53 |
| HC-254F | 2024 outpatient visits | 22,150 × 57 |
| HC-254G | 2024 office-based visits | 145,818 × 45 |

- The zips total 20,549,714 bytes; the parquet files total 30 MB.
- 8,769 of the 8,872 HC-257 people are in HC-256.
- 2024 events for Panel 28 people: Rx 95,153, inpatient 863, ER 1,836, outpatient 10,894, office-based 69,258.

**SBC corpus** (`data/sbc/manifest.csv`)
- 45 SBCs (37.1 MB) from 11 sponsors: 8 public-employer programs, 1 city, 2 university systems. Plus 15 rate sheets (18.4 MB).
- All 45 carry the federal SBC template title, the Peg / Joe / Mia coverage examples, and a parseable coverage period.

| Coverage period | SBCs |
|---|---|
| Calendar 2027 | 20 |
| NJ Jul–Dec 2026 | 8 |
| Calendar 2026 (Georgia, UC) | 4 |
| FY Jul 2025–Jun 2026 (Delaware, Colorado) | 10 |
| FY Jul 2026–Jun 2027 (Albuquerque) | 2 |
| Sep 2026–Aug 2027 (UT System) | 1 |

**Real 2027 menus (SBCs plus 2027 premiums)**
- Washington PEBB: 9 SBCs.
- Oregon PEBB: 4 SBCs.
- Tennessee: 6 SBCs.
- CalPERS (partial): 1 SBC and 5 rate tables.

**Held-out split**
- 20 held out (all calendar-2027 SBCs) and 25 dev. No sponsor appears on both sides.
- Frozen before any scoring.
- `eval/split.json` SHA-256: `4724245afe2d78030e8a22dfaa2845129c3bd378322d380d6ca544b5d29823c6`

**Repo**
- Git on `main`.
- Private GitHub repo: https://github.com/Hotragn/sameaslastyear-wip (visibility checked: PRIVATE).
- A scan found no `.env` value in any tracked file.

### Failed or substituted
- **Ollama:** the default `gemma4:e2b` / `gemma4:e4b` tags are MLX builds with no JSON-schema output, so the reader uses the QAT GGUF builds instead (DECISIONS 4).
- **Ollama server restart:** the existing 7.5-day-old server could not load any model, so it was restarted (DECISIONS 3).
- **TabPFN:** the default v3.5 model needs a license login, so the smoke test ran on v2 (DECISIONS 2).
- **CalPERS:** only the Blue Shield Trio 2027 SBC was collected, and it is watermarked "pending regulatory approval". The PERS PPO SBCs (Included Health) return HTTP 403 to scripts. The Anthem and Kaiser CalPERS 2027 SBCs were not found on pages a script can read.
- **Blocked sponsors:** Arizona (benefitoptions.az.gov) and University of Michigan (hr.umich.edu) block scripted downloads, so they are not in the corpus.
- **Delaware:** the rate sheet is FY2027 but the SBCs are FY2026.
- **New Jersey:** the separate Rx SBCs were not collected; its medical SBCs point to a separate prescription drug plan SBC.
- **Albuquerque:** each SBC is the first 7–8 pages of a 150–160-page plan booklet.
- **Housekeeping:** I removed my own test pull `gemma4:e2b-it-q4_K_M` and stopped a `gemma4:12b` (MLX) pull mid-download; no partial blobs remain.

### Risks
1. **TabPFN license.** v3.5 needs the owner's one-time Prior Labs license acceptance (a browser login). Without it, Step 2 runs on the older v2 weights.
2. **Schedule.** Week 3 closes 2026-10-26 06:59 UTC, about 6 hours after this STATUS. Steps 2–9 are not done.
3. **Memory.** The 26B reader takes 15 GB of the 19 GB GPU budget on a 24 GB machine, so other heavy apps can push it into swap. Step 3 measures peak memory on real SBCs.
4. **No 2027 documents in dev.** Every 2027 document is held out, so prompts are developed only on 2025–2026 layouts.
5. **Redistribution.** 60 sponsor and carrier PDFs are committed to the private repo. Before the repo goes public, decide whether to keep them or publish only the manifest and the fetch script.
6. **Source drift.** Sponsors can replace a PDF at the same URL. The manifest SHA-256 makes that visible, and the committed copies are the reference.
7. **Tennessee carriers.** BCBST and Cigna administer the Tennessee plans jointly, so carrier-level results need care.
8. **IRS limits.** The 2027 IRS limits in the brief are not yet checked against an IRS source; that happens in Step 4.
9. **MEPS coverage.** Full event-level truth in Step 4 may need HC-254C (other medical) and HC-254H (home health), which were not downloaded.
10. **Disk.** Free space fell from 110 GiB to 63 GiB, mostly the five Gemma builds (42 GB). The two MLX builds (17 GB) are unused by the reader and can be removed with `ollama rm` if space is needed.

---

## STATUS: Step 2, spend predictor and benchmark B (2026-10-26, 01:11 UTC)

### Done
- **Population (DECISIONS 11):** 3,033 MEPS Panel 28 adults aged 18–64 with any private coverage in both 2023 and 2024. 2,173 of them had employer coverage in all 24 months.
- **Features and target:** 26 features from 2023; the target is 2024 total spending. The year-2 component totals (office, outpatient, ER, inpatient, Rx), year-2 counts and out-of-pocket spending are saved in `data/derived/spend_panel28.parquet` for Step 4. That file is local only and `make eval-spend` rebuilds it.
- **Split (DECISIONS 12):** 2,275 train and 758 test people, by household, stratified by year-1 tier, seed 20261026. `eval/spend_split.json` SHA-256 is `d9577f32d9f4a365f1e4f5cd2f88f0ad796dfe87423a4d0862d1076edc4039aa`. It was committed and pushed (`876c809`) before any model was scored.
- **Models:** TabPFN v2 (v3.5 refused; DECISIONS 14) and four baselines, with quantiles 0.1–0.9. No nearest-neighbour context selection was needed: 2,275 training rows fit in context.
- **Outputs:** `eval/results/step2_spend.json`, `eval/results/step2_spend.png` and `eval/results/step2_spend.svg`, all from `make eval-spend`. Test-set predictions are in `data/derived/step2_test_predictions.parquet` (local).
- **Runtime:** `make eval-spend` takes 18.3 s wall on an Apple M4 with 24 GB, macOS 26.6.2. TabPFN v2 fit and predict takes 10.8 s on MPS. A second run gave identical metrics.

### Results: 758 test people, unweighted, 95% household-bootstrap CIs

| Model | P10 loss | P50 loss | P90 loss | P10–P90 coverage | Band width |
|---|---|---|---|---|---|
| **TabPFN v2** | **$834** [691, 1,009] | **$3,496** [2,824, 4,310] | **$3,772** [2,784, 4,978] | 77.2% [74.3, 80.3] | $16,534 |
| Same as last year | $4,395 | $4,682 | $4,968 | 5.7% | $0 |
| Age-sex group quantiles | $869 | $3,957 | $4,544 | 86.3% | $19,195 |
| Gradient boosting (quantile) | $875 | $3,508 | $3,893 | 73.2% | $12,244 |
| Linear quantile regression | $880 | $3,564 | $3,823 | 77.2% | $18,832 |

Paired difference, model minus TabPFN v2 (positive means TabPFN has lower loss):

| Model | P10 | P50 | P90 |
|---|---|---|---|
| Same as last year | +$3,562 [2,628, 4,513] | +$1,185 [734, 1,642] | +$1,195 [880, 1,531] |
| Age-sex group quantiles | +$35 [10, 56] | +$461 [278, 664] | +$771 [391, 1,160] |
| Gradient boosting | +$41 [9, 84] | +$11 [−76, 93] | +$120 [−63, 329] |
| Linear quantile regression | +$47 [−28, 176] | +$68 [−36, 199] | +$51 [−241, 333] |

`LONGWT`-weighted results keep the same order:

| Model | P50 loss | P10–P90 coverage |
|---|---|---|
| TabPFN v2 | $3,325 | 75.8% [72.1, 79.6] |
| Gradient boosting | $3,339 | 73.3% |
| Linear quantile regression | $3,358 | 76.1% |
| Age-sex group quantiles | $3,726 | 86.5% |
| Same as last year | $4,363 | 6.2% |

### Where TabPFN loses or does not win
- **No clear win over the strong baselines.** TabPFN is statistically tied with linear quantile regression at all three quantiles, and with gradient boosting at P50 and P90. Its only clear edge over gradient boosting is $41 at P10.
- **Its bands are slightly too narrow.**
  - Unweighted, the P10–P90 band holds 77.2% of people against the 80% target; the CI just reaches 80%.
  - Weighted to the population, it holds 75.8%, and the CI excludes 80%.
  - Age-sex group quantiles reach 86% coverage, but only with bands about $2,700 wider (age-sex $19,195 vs TabPFN $16,534).
- **Point losses by tier, none significant (every CI includes $0):**
  - People with $0 in 2023 (74 people):
    - P50: gradient boosting is $8 better.
    - P90: gradient boosting is $90 better and linear QR $56 better.
    - Band coverage: 85.1% for both of those, against TabPFN's 75.7%.
  - Low tier, P90: linear QR is $33 better.
  - High tier, P50: gradient boosting is $24 better.
  - Employer-coverage subgroup, P50: gradient boosting is $30 better.
- **The win that holds:** TabPFN beats "same as last year" by $1,185 per person in median pinball loss, and the age-sex medians by $461. Both CIs exclude $0.

### Failed or substituted
- **TabPFN v3.5:** refused because the license is not accepted on the token's account. v2 ran instead (DECISIONS 14).
- **Bug caught before commit:** in the first run, the paired-difference block was shifted by one model. The main model's own zero diff was counted, so every "vs TabPFN" row carried the next model's name and the last baseline's diff was dropped. It was fixed and re-run; the committed JSON has correctly aligned diffs (checked against the per-model losses).
- **Value labels:** `scripts/fetch_meps.py` now also saves MEPS value labels per variable. pandas keeps the variable-to-label-set link only in a private attribute (`_varlist` / `_lbllist`).

### Risks
1. **Weaker TabPFN.** Until the license is accepted, the TabPFN in every result is v2, not the current v3.5.
2. **Narrow bands.** P90 risk caps in Step 4 that come from TabPFN quantiles sit a little low: 77% coverage where 80% is the target. Step 4 should report this, or calibrate the bands on train only.
3. **Approximate CIs.** The household bootstrap ignores MEPS strata and PSUs, so the CIs are approximate (labelled in the JSON).
4. **No 2023 events yet.** Step 4's "same as last year" strategy needs year-1 events, but 2023 event files are not downloaded. Without them it must approximate from year-1 component totals and say so.
5. **Spending outside the five components.** About 11% of year-2 spending falls outside the five saved components (dental, home health, vision, other). Step 4 must exclude it or label how it is covered.

---

## STATUS: Step 2b, learning curve (2026-10-26, 01:25 UTC)

`make eval-spend` now also writes `eval/results/step2b_learning_curve.json`, `eval/results/step2b_learning_curve.png` and `eval/results/step2b_learning_curve.svg`. The full target takes about 3 minutes; the learning curve alone takes 141 s, of which TabPFN is 102 s. Two runs gave identical results.

**Smallest training size within 5% of the model's own full-data loss:**

| Model | P50 | P90 |
|---|---|---|
| TabPFN v2 | 200 | 200 |
| Gradient boosting | 200 | 400 |
| Linear quantile regression | 800 | 2,275 |

**Paired difference, model minus TabPFN** (95% household-bootstrap CIs; negative means the model beats TabPFN):

| Size | Gradient boosting, P50 | Gradient boosting, P90 | Linear QR, P50 |
|---|---|---|---|
| 100 | **−$326** [−507, −146] | −$282 [−711, +276] | +$82,620 |
| 200 | −$30 [−130, +67] | +$198 [−18, +467] | +$1,426 [883, 2,079] |
| 400 | +$44 [−11, +103] | **+$215** [20, 445] | +$511 [268, 811] |
| 800 | +$40 [−25, +99] | **+$235** [50, 437] | +$209 [73, 390] |
| 2,275 | +$14 [−76, +97] | +$115 [−60, +310] | +$70 [−34, +203] |

**Answer: partly.**
- **Linear QR needs far more people.** TabPFN reaches full-data accuracy with far fewer people than linear quantile regression, which is unusable below 400 people. Unpenalised, it gives a P50 loss of $86,640 at 100 people.
- **Gradient boosting matches TabPFN at the median.** Both are within 5% at 200 people.
- **TabPFN's edge over gradient boosting is the bad-year quantile and calibration:**
  - It is better at P90 with 400–800 people (CIs exclude $0).
  - Its P10–P90 coverage stays at 74–80% at every size, while gradient boosting's falls to 63–73%.
- **Where TabPFN loses:** at 100 people, gradient boosting beats it at the median by $326 (CI $146–$507). TabPFN is also much less stable there: the median-loss standard deviation across draws is $548, against $55.

---

## STATUS: Step 4, rules engine and benchmark C (2026-10-26, 02:15 UTC)

### Done
- **Rules engine** (`sameaslastyear/costs.py`):
  - deductible: embedded or aggregate
  - copay vs coinsurance, with or without the deductible
  - per-day copays with caps; per-fill coinsurance caps
  - out-of-pocket limit, with separate prescription limits where a plan has them
  - preventive care at $0
  - HSA employer seed and payroll tax savings
  - 2027 IRS and CMS limits cited from primary sources (DECISIONS 21)
  - `make test` passes all 41 tests. The 20 engine tests include three Georgia SBC coverage examples reproduced to the dollar.
- **Real 2027 menus:** WA PEBB 9 plans, OR PEBB 4, TN state 3. Every non-gold value is quote-verified on load (DECISIONS 23).
- **2023 event files** HC-248A/D/E/F/G were added, for "same as last year".
- **Benchmark C:** `make eval-decision` writes `eval/results/step4_decision.json`, `eval/results/step4_decision.png` and `eval/results/step4_decision.svg` in 6 s. The modelled events cover 88.4% of the test people's 2024 spending.

### Results: 758 people × 3 menus, pooled, 95% household-bootstrap CIs

| Strategy | Got the cheapest plan | Mean regret | 90th-percentile regret |
|---|---|---|---|
| Lowest premium | **82.1%** [80.4, 83.9] | **$106** [91, 120] | **$310** [229, 423] |
| TabPFN expected cost | 79.5% [77.5, 81.4] | $121 [106, 138] | $456 [347, 535] |
| TabPFN risk-capped | 79.1% | $124 | $459 |
| Same as last year | 78.1% [76.1, 80.1] | $149 [129, 172] | $608 [504, 689] |
| Lowest deductible | 0.6% | $1,078 | $1,912 |

**Paired mean-regret differences against TabPFN expected cost:**
- Lowest premium: −$15 [−24, −7]. Lowest premium is better.
- Same as last year: +$28 [12, 45]. TabPFN is better.
- Risk-capped: +$3 [−1, 6], a tie.

**By menu** (got the cheapest plan / mean regret):

| Menu | Lowest premium | TabPFN | Same as last year |
|---|---|---|---|
| WA PEBB | 76% / $138 | 73% / $148 | 73% / $185 |
| OR PEBB | 94% / $35 | 91% / $39 | 90% / $40 |
| TN state | 76% / $144 | 74% / $177 | 71% / $223 |

**By 2023 spending tier:** prediction does not beat the lowest premium in any tier.
- **Zero and low 2023 spenders:** all three strategies are within a point of each other at the cheapest plan (91–96%).
- **High 2023 spenders:**
  - Lowest premium: 71% got the cheapest plan, $166 mean regret.
  - TabPFN expected cost: 64%, $197.
  - Same as last year: 63%, $236.

**Headline, verbatim from the JSON:** "Across 758 held-out MEPS adults and 3 real 2027 employer menus, choosing the plan with the lowest TabPFN-expected cost picked the truly cheapest plan 79% of the time, against 78% for picking what would have been best last year and 82% for the lowest premium; mean regret was $121 versus $149 and $106."

**What this means:**
- On these menus, the low-premium plans, most with an employer HSA deposit, are truly cheapest for most people. For example, KP NW CDHP is cheapest for 576 of 758 people in WA, and the TN CDHP for 578 of 758.
- A "lowest premium" rule is hard to beat on average.
- TabPFN-based choice beats "same as last year", the rule the product is named after, but not the lowest premium.

### Failed or substituted
- **TabPFN:** the Step 2 v2 predictions were reused rather than re-run on CPU. The GPU was busy with the reader run, and Step 2's run is the same model and split.
- **WA HSA:** the $700.08 deposit for the KP CDHPs is assumed from the UMP CDHP certificate.
- **TN menu:** only the 3 plans priced on the state rate sheet.
- **Availability:** regional plan availability is ignored.
- **Approximations:** event-to-row mapping, a $50 generic/brand threshold, no separate imaging or physician lines, and self-only coverage only. All are labelled in the JSON.

### Risks
1. **Not a product claim yet.** The result is approximate in the ways listed above. It is reproducible, but not a claim about real enrollees until the mappings are validated.
2. **A design question for the product.** If the plain lowest-premium rule usually wins, the product's value lies in finding the people for whom it fails, and the high-spender tier shows prediction does not yet find them. This needs a product decision, not a re-tune on this test set.
3. **Narrow TabPFN bands.** Their 77% coverage (Step 2) may push risk-capped choices the wrong way.
