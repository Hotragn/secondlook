"""Check which local Gemma 4 builds load, fit on the GPU and honour a JSON schema.

For each model: one schema-constrained chat call against the local Ollama server,
then `ollama ps` for resident size and CPU/GPU split. Models run one at a time
(each is unloaded before the next) so the sizes do not overlap.

Run: uv run python -I scripts/check_models.py [model ...]
With no arguments it checks DEFAULT_MODELS and writes eval/results/step1_model_check.json;
with explicit models it only prints.
"""

import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "step1_model_check.json"
OLLAMA = "http://127.0.0.1:11434"
NUM_CTX = 16384

DEFAULT_MODELS = [
    "gemma4:e2b",  # tag from the brief; resolves to an MLX nvfp4 build on Apple Silicon
    "gemma4:e4b",
    "gemma4:e2b-it-qat",  # GGUF builds, run by the llama.cpp runner
    "gemma4:e4b-it-qat",
    "gemma4:26b-a4b-it-qat",
]

SOURCE = "Page 2 text: 'What is the overall deductible? $1,500 individual / $3,000 family'."
SCHEMA = {
    "type": "object",
    "properties": {"quote": {"type": "string"}, "page": {"type": "integer"}},
    "required": ["quote", "page"],
}


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read() or b"{}")


def ps_row(model: str) -> dict:
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines()[1:]:
        if line.split()[0] == model:
            cols = [c.strip() for c in line.split("  ") if c.strip()]
            return {"size": cols[2], "processor": cols[3], "context": cols[4], "runner": cols[5]}
    return {}


def check(model: str) -> dict:
    r = post("/api/chat", {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "60s",
        "format": SCHEMA,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
        "messages": [{
            "role": "user",
            "content": SOURCE + " Return the verbatim quote that states the deductible, and its page.",
        }],
    })
    result = {"model": model, "error": r.get("error")}
    if not result["error"]:
        content = r["message"]["content"]
        parsed = json.loads(content)
        quote_ok = " ".join(parsed["quote"].split()) in " ".join(SOURCE.split())
        result.update({
            "schema_valid": set(parsed) >= {"quote", "page"},
            "quote_is_verbatim": quote_ok,
            "page": parsed["page"],
            "load_s": round(r["load_duration"] / 1e9, 1),
            "tokens_per_s": round(r["eval_count"] / (r["eval_duration"] / 1e9), 1),
            **ps_row(model),
        })
    subprocess.run(["ollama", "stop", model], capture_output=True)
    return result


def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    results = [check(m) for m in models]
    for r in results:
        print(json.dumps(r))
    if sys.argv[1:]:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_ctx": NUM_CTX,
        "results": results,
    }, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
