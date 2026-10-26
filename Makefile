.PHONY: setup data check split-hash eval-spend

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
