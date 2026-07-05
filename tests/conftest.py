"""Shared pytest fixtures.

The session-scoped ``trained_model`` fixture trains a *tiny* XGBoost model on
synthetic data matching the FEATURE CONTRACT and writes ``model_xgb.json`` +
``metrics.json`` into the real ``MODELS_DIR``. This lets the serving/scoring tests
run WITHOUT a Kaggle GPU run. Any pre-existing artifacts are backed up and restored
on teardown so a developer's real model is never clobbered.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from fraud_scoring.config import MODELS_DIR, ensure_dirs
from fraud_scoring.features.build_features import FEATURE_COLUMNS, build_features

SEED = 42
_MODEL_PATH = MODELS_DIR / "model_xgb.json"
_METRICS_PATH = MODELS_DIR / "metrics.json"


def _synthetic_raw(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a raw transaction frame (Time, V1..V28, Amount) + a fraud label."""
    data: dict[str, np.ndarray] = {}
    data["Time"] = rng.uniform(0, 172_800, size=n)  # up to two days of seconds
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0.0, 1.0, size=n)
    data["Amount"] = np.abs(rng.gamma(2.0, 60.0, size=n))
    df = pd.DataFrame(data)

    # Label is a deterministic-ish function of a few features so the model learns.
    logit = 1.4 * df["V1"] - 1.1 * df["V2"] + 0.002 * df["Amount"] - 1.0
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)
    return df, y


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Write a tiny xgboost booster + metrics.json into MODELS_DIR for the session."""
    ensure_dirs()

    # Back up any real artifacts so we can restore them afterwards.
    backup: dict = {}
    for path in (_MODEL_PATH, _METRICS_PATH):
        backup[path] = path.read_bytes() if path.exists() else None

    rng = np.random.default_rng(SEED)
    raw, y = _synthetic_raw(600, rng)
    X, stats = build_features(raw)  # fits amount_mean / amount_std

    dtrain = xgb.DMatrix(X.to_numpy(dtype="float64"), label=y)
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.3,
        "eval_metric": "logloss",
        "seed": SEED,
    }
    booster = xgb.train(params, dtrain, num_boost_round=25)
    booster.save_model(str(_MODEL_PATH))

    metrics = {
        "models": {"xgboost": {"pr_auc": 0.5}},
        "feature_columns": list(FEATURE_COLUMNS),
        "amount_stats": {"mean": stats["amount_mean"], "std": stats["amount_std"]},
        "best_model": "xgboost",
        "seed": SEED,
    }
    _METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    yield {"model_path": _MODEL_PATH, "metrics_path": _METRICS_PATH}

    # Restore original state.
    for path, content in backup.items():
        if content is None:
            if path.exists():
                path.unlink()
        else:
            path.write_bytes(content)


@pytest.fixture()
def sample_txn() -> dict:
    """A single raw transaction dict (Time, V1..V28, Amount, account_id)."""
    rng = np.random.default_rng(7)
    txn: dict = {
        "transaction_id": "txn-0001",
        "Time": 3600.0 * 13,  # 13:00
        "Amount": 125.50,
        "account_id": "acct-123",
        "merchant_id": "merch-999",
    }
    for i in range(1, 29):
        txn[f"V{i}"] = float(rng.normal(0.0, 1.0))
    return txn
