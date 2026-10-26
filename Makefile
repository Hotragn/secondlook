.PHONY: setup data check split-hash

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

# The split is frozen; compare this with the hash recorded in DECISIONS.md.
split-hash:
	shasum -a 256 eval/split.json
