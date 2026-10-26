"""TabPFN smoke test: fit on 10 synthetic rows, predict quantiles, check the output.

Uses TabPFN v2 weights, the only version that downloads without a Prior Labs
license login (v2.5 and later are gated; see DECISIONS.md). The data is random
and exists only to prove the model loads and runs on this machine.

Run: uv run python scripts/smoke_tabpfn.py
"""

import time
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

QUANTILES = [0.1, 0.5, 0.9]


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=10)

    reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2, random_state=0)
    first_run = not Path(reg.model_path).exists()
    start = time.perf_counter()
    reg.fit(X, y)
    preds = reg.predict(X[:3], output_type="quantiles", quantiles=QUANTILES)
    elapsed = time.perf_counter() - start

    q = np.column_stack(preds)
    assert q.shape == (3, len(QUANTILES)), q.shape
    assert np.isfinite(q).all()
    assert (np.diff(q, axis=1) >= 0).all(), "quantiles are not monotone"

    print(f"torch {torch.__version__}, mps available: {torch.backends.mps.is_available()}")
    print(f"model: {reg.model_path}")
    note = " (includes weight download)" if first_run else ""
    print(f"fit+predict on 10 rows: {elapsed:.1f}s{note}")
    for row, target in zip(q, y[:3]):
        print(f"  y={target:+.2f}  q10={row[0]:+.2f}  q50={row[1]:+.2f}  q90={row[2]:+.2f}")
    print("OK")


if __name__ == "__main__":
    main()
