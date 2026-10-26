# Resume

Where the build stands, for whoever picks it up next. The spec is `BRIEF.md`; the reasons behind choices are in `DECISIONS.md`. Each step ends with a STATUS block below.

## Next step

Step 2: spend predictor and benchmark B (TabPFN on MEPS Panel 28).

Before starting, check whether the Prior Labs license for TabPFN v3.5 has been accepted (DECISIONS 2). If it has not, Step 2 runs on v2 and says so.

---

## STATUS: Step 1, setup and data (2026-10-26, about 01:00 UTC)

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
