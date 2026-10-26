"""Spend predictor models and metrics shared by the Step 2 scripts (DECISIONS 13).

Every learned model fits log1p(2024 spending) and maps predicted quantiles back
with expm1; quantiles survive any increasing transform.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261026
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
FEATURES = [
    "age", "female", "region_midwest", "region_south", "region_west",
    "self_rated_health", "self_rated_mental_health",
    "high_bp", "coronary_hd", "angina", "heart_attack", "other_heart", "stroke",
    "emphysema", "high_chol", "cancer", "diabetes", "arthritis", "asthma", "n_conditions",
    "office_visits", "outpatient_visits", "er_visits", "inpatient_stays", "rx_fills", "spend_y1",
]
LOG_FEATURES = ["office_visits", "outpatient_visits", "er_visits", "inpatient_stays", "rx_fills", "spend_y1"]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].astype(float).copy()
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    return X.to_numpy()


def prepare_tabpfn_auth() -> None:
    """Expose only TABPFN_TOKEN from .env, never open a browser, never cache the token."""
    token = dotenv_values(ROOT / ".env").get("TABPFN_TOKEN")
    if token:
        os.environ["TABPFN_TOKEN"] = token.strip()
    os.environ["TABPFN_NO_BROWSER"] = "1"
    import tabpfn.browser_auth as auth

    auth.save_token = lambda token: None  # the library would write it to ~/.cache in plain text


def tabpfn_model(version: str, seed: int = SEED):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    v = {"v2": ModelVersion.V2, "v3.5": ModelVersion.V3_5}[version]

    def fit_predict(Xtr, ytr, Xte, _train, _test):
        reg = TabPFNRegressor.create_default_for_version(v, random_state=seed)
        reg.fit(Xtr, ytr)
        q = reg.predict(Xte, output_type="quantiles", quantiles=QUANTILES)
        return np.expm1(np.column_stack(q)), str(getattr(reg, "devices_", ""))

    return fit_predict


def same_as_last_year(_Xtr, _ytr, _Xte, _train, test):
    return np.repeat(test["spend_y1"].to_numpy()[:, None], len(QUANTILES), axis=1), "cpu"


def age_sex_quantiles(_Xtr, _ytr, _Xte, train, test):
    groups = train.groupby(["age_band", "female"])["spend_y2"]
    table = {k: np.quantile(v.to_numpy(), QUANTILES) for k, v in groups}
    return np.vstack([table[k] for k in zip(test["age_band"], test["female"])]), "cpu"


def per_quantile(make):
    def fit_predict(Xtr, ytr, Xte, _train, _test):
        cols = []
        for q in QUANTILES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cols.append(make(q).fit(Xtr, ytr).predict(Xte))
        return np.expm1(np.sort(np.column_stack(cols), axis=1)), "cpu"

    return fit_predict


def hist_gb(seed: int = SEED):
    return per_quantile(
        lambda q: HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=seed)
    )


linear_qr = per_quantile(
    lambda q: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        QuantileRegressor(quantile=q, alpha=0.0, solver="highs"),
    )
)


def person_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for j, q in enumerate(QUANTILES):
        diff = y - pred[:, j]
        out[f"pinball_q{round(q * 100)}"] = np.maximum(q * diff, (q - 1) * diff)
    lo, hi = pred[:, QUANTILES.index(0.1)], pred[:, QUANTILES.index(0.9)]
    out["coverage_p10_p90"] = ((y >= lo) & (y <= hi)).astype(float)
    out["width_p10_p90"] = hi - lo
    return out


def bootstrap_counts(n_households: int, resamples: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_households, size=(resamples, n_households))
    return np.stack([np.bincount(row, minlength=n_households) for row in draws])


def summarise(values: np.ndarray, mask: np.ndarray, weights: np.ndarray,
              household: np.ndarray, counts: np.ndarray) -> list[dict]:
    """Mean of each column over masked people, with a household-bootstrap 95% CI."""
    w = weights * mask
    sums = np.zeros((counts.shape[1], values.shape[1]))
    np.add.at(sums, household, values * w[:, None])
    totals = np.bincount(household, weights=w, minlength=counts.shape[1])
    point = sums.sum(0) / totals.sum()
    boot = (counts @ sums) / (counts @ totals)[:, None]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return [{"value": float(p), "ci95": [float(a), float(b)]} for p, a, b in zip(point, lo, hi)]


def rounded(d):
    if isinstance(d, dict):
        return {k: rounded(v) for k, v in d.items()}
    if isinstance(d, list):
        return [rounded(v) for v in d]
    if isinstance(d, float):
        return round(d, 4) if abs(d) < 10 else round(d, 1)
    return d
