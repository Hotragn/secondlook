.PHONY: setup data check split-hash eval-spend eval-reader eval-decision eval-lazy test

setup:
	uv sync
	ollama pull gemma4:e2b-it-qat
	ollama pull gemma4:e4b-it-qat
	ollama pull gemma4:26b-a4b-it-qat

data:
	uv run python -I scripts/fetch_meps.py
	uv run python -I scripts/fetch_sbc.py

check:
	uv run python scripts/smoke_tabpfn.py
	uv run python -I scripts/check_models.py gemma4:e2b-it-qat gemma4:e4b-it-qat gemma4:26b-a4b-it-qat

# The splits are frozen; compare these with the hashes recorded in DECISIONS.md.
split-hash:
	shasum -a 256 eval/split.json eval/spend_split.json

# Step 2 / benchmark B: spend table, models and metrics, learning curve (2b), charts.
eval-spend:
	uv run python -I scripts/spend_data.py
	uv run python -I scripts/spend_eval.py
	uv run python -I scripts/spend_learning_curve.py
	uv run python -I scripts/spend_chart.py

# Step 3 / eval A: three Gemma 4 builds on the 20 held-out SBCs. Raw model output is
# cached in eval/results/step3_raw, so a re-run only re-scores.
eval-reader:
	uv run python -I scripts/reader_eval.py heldout
	uv run python -I scripts/reader_chart.py

# Step 4 / benchmark C: strategies on the real 2027 menus. Uses eval-spend's spend table and
# TabPFN test predictions (data/derived/), so run make eval-spend first.
eval-decision:
	uv run python -I scripts/decision_eval.py
	uv run python -I scripts/decision_chart.py

# Step 4b: the lazy-safe check (pre-registered in DECISIONS 25). Needs make eval-spend first.
eval-lazy:
	uv run python -I scripts/lazy_safe_eval.py
	uv run python -I scripts/lazy_safe_chart.py

test:
	uv run pytest -q tests
