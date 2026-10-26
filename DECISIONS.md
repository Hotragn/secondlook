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
