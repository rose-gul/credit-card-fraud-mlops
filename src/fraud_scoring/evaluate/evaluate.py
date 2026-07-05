"""Offline evaluation of the XGBoost champion on the held-out test split.

This is the DVC ``evaluate`` stage. It loads the saved booster
(``models/model_xgb.json``), applies the persisted Amount statistics from
``models/metrics.json`` (so features match training exactly), scores the test
parquet, and writes ``metrics/eval.json`` with:

* headline metrics (roc_auc, pr_auc, precision/recall/f1 @0.5),
* a threshold sweep table,
* a confusion matrix at the ``params.thresholds.block_above`` cutoff,
* a simple business-cost metric (cost_fn=100 per missed fraud, cost_fp=5 per
  false alarm).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_scoring.config import (
    METRICS_DIR,
    MODELS_DIR,
    PARAMS,
    PROCESSED_DIR,
    ensure_dirs,
)
from fraud_scoring.features.build_features import TARGET, build_features

MODEL_PATH = MODELS_DIR / "model_xgb.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
EVAL_PATH = METRICS_DIR / "eval.json"

COST_FN = 100.0  # cost of a missed fraud (false negative)
COST_FP = 5.0  # cost of a false alarm (false positive)

_SWEEP_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def _load_amount_stats() -> dict[str, float]:
    """Read persisted Amount stats from metrics.json (fallback: None -> refit)."""
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            stats = metrics.get("amount_stats")
            if stats and "mean" in stats and "std" in stats:
                return {
                    "amount_mean": float(stats["mean"]),
                    "amount_std": float(stats["std"]),
                }
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return {}


def _confusion_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict[str, Any]:
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = COST_FN * int(fn) + COST_FP * int(fp)
    return {
        "threshold": float(thr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cost": float(cost),
    }


def evaluate() -> dict[str, Any]:
    """Score the test split with the saved booster and write ``metrics/eval.json``."""
    ensure_dirs()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at '{MODEL_PATH}'. Train first "
            "(`python -m fraud_scoring.models.train_xgb`) or fetch the Kaggle "
            "GPU run output before evaluating."
        )

    test_path = PROCESSED_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found at '{test_path}'. Run the preprocess stage "
            "first (`python -m fraud_scoring.data.preprocess`)."
        )

    test_df = pd.read_parquet(test_path)

    # Apply persisted train Amount stats so features match training (no refit).
    stats = _load_amount_stats()
    X_test, _ = build_features(test_df, stats=stats or None)
    y_true = test_df[TARGET].to_numpy(dtype=int)

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    dmatrix = xgb.DMatrix(X_test.to_numpy(dtype=np.float64))
    y_score = booster.predict(dmatrix)

    # Headline metrics @0.5.
    y_pred_05 = (y_score >= 0.5).astype(int)
    headline = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred_05, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_05, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_05, zero_division=0)),
    }

    # Threshold sweep.
    sweep = [_confusion_at(y_true, y_score, thr) for thr in _SWEEP_THRESHOLDS]

    # Confusion matrix + cost at the production block cutoff.
    block_above = float(PARAMS["thresholds"]["block_above"])
    block_cm = _confusion_at(y_true, y_score, block_above)

    report: dict[str, Any] = {
        "model": "xgboost",
        "n_test": int(len(test_df)),
        "n_fraud": int(y_true.sum()),
        **headline,
        "block_threshold": block_above,
        "confusion_at_block": block_cm,
        "cost_at_block": block_cm["cost"],
        "cost_params": {"cost_fn": COST_FN, "cost_fp": COST_FP},
        "amount_stats": stats or "refit_on_test",
        "threshold_sweep": sweep,
    }

    EVAL_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[evaluate] n_test={report['n_test']}  n_fraud={report['n_fraud']}")
    print(
        f"[evaluate] roc_auc={headline['roc_auc']:.4f}  "
        f"pr_auc={headline['pr_auc']:.4f}  f1@0.5={headline['f1']:.4f}"
    )
    print(
        f"[evaluate] @block={block_above:.2f}  "
        f"tp={block_cm['tp']} fp={block_cm['fp']} fn={block_cm['fn']} "
        f"recall={block_cm['recall']:.4f}  cost={block_cm['cost']:.0f}"
    )
    print(f"[evaluate] wrote {EVAL_PATH}")
    return report


if __name__ == "__main__":
    evaluate()
