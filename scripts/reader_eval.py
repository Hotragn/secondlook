"""Step 3, eval A: run the plan reader on SBCs and score it.

  uv run python -I scripts/reader_eval.py dev [--models a,b] [--docs x,y]   prompt development (dev SBCs only)
  uv run python -I scripts/reader_eval.py heldout                             the scored run (make eval-reader)

Raw model output is cached in eval/results/step3_raw/<set>/<model>/<doc_id>.json
with the prompt fingerprint, so scoring re-runs without calling the models; a
changed prompt invalidates the cache. While a model runs, its resident size
(Ollama /api/ps) and the RSS of Ollama processes are sampled every 0.5 s.
The held-out set is scored against eval/gold/heldout_gold.json.

Writes eval/results/step3_reader.json (heldout) or eval/results/step3_dev.json (dev).
"""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sameaslastyear import reader, rules

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sbc" / "manifest.csv"
SPLIT = ROOT / "eval" / "split.json"
SPLIT_SHA256 = "4724245afe2d78030e8a22dfaa2845129c3bd378322d380d6ca544b5d29823c6"
GOLD = ROOT / "eval" / "gold" / "heldout_gold.json"
RAW = ROOT / "eval" / "results" / "step3_raw"
MODELS = ["gemma4:e2b-it-qat", "gemma4:e4b-it-qat", "gemma4:26b-a4b-it-qat"]
GOLD_FIELDS = ["deductible_individual", "deductible_family", "oop_individual", "oop_family",
               "primary_care", "specialist", "urgent_care", "emergency_room", "generic_drugs",
               "inpatient_facility", "peg_total", "joe_total", "mia_total"]
FINGERPRINT = hashlib.sha256((reader.SYSTEM + json.dumps(reader.SECTIONS, sort_keys=True)
                              + str(reader.NUM_CTX)).encode()).hexdigest()[:16]


class MemorySampler(threading.Thread):
    def __init__(self, model: str):
        super().__init__(daemon=True)
        self.model, self.stop, self.peak_size, self.peak_vram, self.peak_rss = model, False, 0, 0, 0

    def run(self):
        while not self.stop:
            try:
                with urllib.request.urlopen(f"{reader.OLLAMA}/api/ps", timeout=5) as r:
                    for m in json.load(r).get("models", []):
                        if m.get("name") == self.model:
                            self.peak_size = max(self.peak_size, m.get("size", 0))
                            self.peak_vram = max(self.peak_vram, m.get("size_vram", 0))
                ps = subprocess.run(["ps", "-axo", "rss=,command="], capture_output=True, text=True).stdout
                rss = sum(int(line.split()[0]) * 1024 for line in ps.splitlines()
                          if "ollama" in line or "llama-server" in line)
                self.peak_rss = max(self.peak_rss, rss)
            except Exception:
                pass
            time.sleep(0.5)


def documents(which: str) -> list[dict]:
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == SPLIT_SHA256, "eval/split.json changed"
    split = json.loads(SPLIT.read_text())
    ids = [d["doc_id"] for d in split["held_out" if which == "heldout" else "dev"]]
    rows = {r["doc_id"]: r for r in csv.DictReader(MANIFEST.open())}
    return [rows[i] for i in ids]


def run(which: str, models: list[str], docs: list[dict]) -> dict:
    perf = {}
    for model in models:
        out_dir = RAW / which / model.replace(":", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [d for d in docs if not cached(out_dir / f"{d['doc_id']}.json")]
        sampler = MemorySampler(model)
        sampler.start()
        if todo:
            reader.chat(model, {"ok": "Return the word OK."}, "=== Page 1 ===\nOK")  # load before timing
        for d in todo:
            result = reader.run_model(model, ROOT / d["path"])
            result.update({"doc_id": d["doc_id"], "sha256": d["sha256"], "model": model,
                           "fingerprint": FINGERPRINT})
            (out_dir / f"{d['doc_id']}.json").write_text(json.dumps(result, indent=1) + "\n")
            secs = sum(t["seconds"] for t in result["timing"].values())
            print(f"{model} {d['doc_id']}: {secs:.1f}s", flush=True)
        sampler.stop = True
        sampler.join()
        subprocess.run(["ollama", "stop", model], capture_output=True)
        if todo:
            perf[model] = {"peak_ollama_size_bytes": sampler.peak_size, "peak_vram_bytes": sampler.peak_vram,
                           "peak_ollama_process_rss_bytes": sampler.peak_rss}
    return perf


def cached(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text()).get("fingerprint") == FINGERPRINT


def same(field: str, pred, gold) -> bool:
    if pred is None:
        return False
    if isinstance(gold, dict):
        for k in ("copay", "coinsurance"):
            a, b = pred.get(k), gold.get(k)
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 0.005):
                return False
        if bool(pred.get("not_covered")) != bool(gold.get("not_covered")):
            return False
        if gold.get("deductible_applies") is not None and pred.get("deductible_applies") is not None:
            return pred["deductible_applies"] == gold["deductible_applies"]
        return True
    return abs(float(pred) - float(gold)) < 0.5


def doc_bootstrap(per_doc: list[float], resamples: int = 2000) -> list[float]:
    rng = np.random.default_rng(20261026)
    x = np.array(per_doc)
    boot = x[rng.integers(0, len(x), size=(resamples, len(x)))].mean(1)
    return [float(v) for v in np.percentile(boot, [2.5, 97.5])]


def score(which: str, models: list[str], docs: list[dict], perf: dict) -> dict:
    gold = json.loads(GOLD.read_text())["documents"] if which == "heldout" else {}
    results, failures = {}, []
    for model in models:
        out_dir = RAW / which / model.replace(":", "_")
        status, matches, ex_status, per_doc_acc, seconds, verdicts = Counter(), Counter(), Counter(), [], [], Counter()
        checks_failed, by_field, per_doc_clear = Counter(), {k: Counter() for k in GOLD_FIELDS}, []
        wrong_docs = flagged_wrong_docs = 0
        for d in docs:
            raw = json.loads((out_dir / f"{d['doc_id']}.json").read_text())
            seconds.append(sum(t["seconds"] for t in raw["timing"].values()))
            fields = reader.check_fields(ROOT / d["path"], raw)
            status.update(f["status"] for f in fields.values())
            matches.update(f["match"] for f in fields.values())
            rec = rules.reconcile(fields)
            for p in rules.PEOPLE:
                ex_status[rec[p]["status"]] += 1
                checks_failed.update(k for k, ok in rec[p]["checks"].items() if not ok)
            if d["doc_id"] in gold:
                ambiguous = set(gold[d["doc_id"]]["ambiguous"])
                correct = clear = n_clear = 0
                for key in GOLD_FIELDS:
                    g = gold[d["doc_id"]]["fields"][key]
                    f = fields[key]
                    ok = f["status"] == "verified" and same(key, f["value"], g)
                    verdict = "correct" if ok else ("needs_check" if f["status"] == "needs_check" else "wrong")
                    verdicts[verdict] += 1
                    by_field[key][verdict] += 1
                    correct += ok
                    if key not in ambiguous:
                        clear += ok
                        n_clear += 1
                    if not ok:
                        failures.append({"model": model, "doc_id": d["doc_id"], "field": key, "verdict": verdict,
                                         "ambiguous": key in ambiguous, "quote": f["quote"], "page": f["page"],
                                         "match": f["match"], "parsed": f["value"], "gold": g, "reason": f["reason"]})
                per_doc_acc.append(correct / len(GOLD_FIELDS))
                per_doc_clear.append((clear, n_clear))
                if any(f["doc_id"] == d["doc_id"] and f["model"] == model and f["verdict"] == "wrong" for f in failures):
                    wrong_docs += 1
                    flagged_wrong_docs += any(rec[p]["status"] == "mismatch" for p in rules.PEOPLE)
        n_fields = sum(status.values())
        n_ex = sum(ex_status.values())
        res = {
            "documents": len(docs),
            "fields_per_document": n_fields // max(len(docs), 1),
            "verified_rate": status["verified"] / n_fields,
            "needs_check_rate": status["needs_check"] / n_fields,
            "quote_match": dict(matches),
            "coverage_examples": {"match_rate": ex_status["match"] / n_ex, "counts": dict(ex_status),
                                  "failed_checks": dict(checks_failed)},
            "seconds_per_pdf": {"mean": statistics.mean(seconds), "median": statistics.median(seconds),
                                "max": max(seconds)},
            **({"memory": perf[model]} if model in perf else {}),
        }
        if per_doc_acc:
            res["gold_fields"] = {
                "fields": GOLD_FIELDS, "n": sum(verdicts.values()),
                "accuracy": sum(per_doc_acc) / len(per_doc_acc), "accuracy_ci95": doc_bootstrap(per_doc_acc),
                "wrong_rate": verdicts["wrong"] / sum(verdicts.values()),
                "needs_check_rate": verdicts["needs_check"] / sum(verdicts.values()),
                "counts": dict(verdicts),
                "accuracy_excluding_ambiguous": sum(c for c, _ in per_doc_clear) / sum(n for _, n in per_doc_clear),
                "n_excluding_ambiguous": sum(n for _, n in per_doc_clear),
                "by_field": {k: {"correct": v["correct"], "wrong": v["wrong"], "needs_check": v["needs_check"]}
                             for k, v in by_field.items()},
                "documents_with_a_wrong_value": wrong_docs,
                "of_which_flagged_by_example_reconciliation": flagged_wrong_docs,
            }
        results[model] = res
    return {"results": results, "failure_log": failures}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["dev", "heldout"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--docs", default="")
    args = ap.parse_args()
    models = args.models.split(",")
    docs = documents(args.which)
    if args.docs:
        docs = [d for d in docs if d["doc_id"] in args.docs.split(",")]
    if args.which == "heldout":
        assert GOLD.exists(), "commit the held-out gold before scoring"
    started = time.perf_counter()
    perf = run(args.which, models, docs)
    scored = score(args.which, models, docs, perf)
    out = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "set": args.which, "prompt_fingerprint": FINGERPRINT, "num_ctx": reader.NUM_CTX,
        "documents": [d["doc_id"] for d in docs], **scored,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    path = ROOT / "eval" / "results" / ("step3_reader.json" if args.which == "heldout" else "step3_dev.json")
    prev = json.loads(path.read_text()) if path.exists() else {}
    for m, r in prev.get("results", {}).items():  # keep memory figures measured on an earlier, uncached run
        if "memory" in r and "memory" not in out["results"].get(m, {"memory": 1}):
            out["results"][m]["memory"] = r["memory"]
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    for m, r in out["results"].items():
        g = r.get("gold_fields", {})
        print(f"{m}: verified {r['verified_rate']:.1%}, examples matched {r['coverage_examples']['match_rate']:.1%}, "
              f"{r['seconds_per_pdf']['mean']:.0f}s/pdf" + (f", gold accuracy {g['accuracy']:.1%}" if g else ""))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
