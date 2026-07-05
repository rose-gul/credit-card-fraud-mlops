"""Scoring agent: loads the trained XGBoost booster and turns a raw transaction
into a calibrated fraud probability.

The agent is intentionally defensive: the real model is produced by a Kaggle GPU
run and may be *absent* at dev time. In that case :attr:`is_ready` is ``False`` and
:meth:`score` raises a clear ``RuntimeError`` so the orchestrator can fall back to a
DEGRADED response instead of crashing the API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from fraud_scoring.config import MODELS_DIR, get_settings
from fraud_scoring.features.build_features import (
    FEATURE_COLUMNS,
    RAW_FEATURE_COLS,
    build_features,
)

DEFAULT_MODEL_PATH = MODELS_DIR / "model_xgb.json"
DEFAULT_METRICS_PATH = MODELS_DIR / "metrics.json"


class ScoringAgent:
    """Load an XGBoost booster and score single transactions.

    Parameters
    ----------
    model_path:
        Path to the xgboost booster JSON. Defaults to ``get_settings().model_path``
        and finally ``MODELS_DIR/model_xgb.json``.
    metrics_path:
        Path to ``metrics.json`` carrying ``amount_stats`` / ``feature_columns`` /
        ``best_model``. Optional — sensible defaults are used when absent.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
    ) -> None:
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            self.model_path = Path(get_settings().model_path or DEFAULT_MODEL_PATH)
        self.metrics_path = Path(metrics_path) if metrics_path else DEFAULT_METRICS_PATH

        # Defaults; refined from metrics.json when available.
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)
        self.amount_stats: dict[str, float] | None = None
        self.model_version: str = "unknown"

        self.booster: xgb.Booster | None = None
        self.is_ready: bool = False

        self._load_metrics()
        self._load_booster()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_metrics(self) -> None:
        if not self.metrics_path.exists():
            return
        try:
            meta = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt metrics must not crash serving
            return

        cols = meta.get("feature_columns")
        if isinstance(cols, list) and cols:
            self.feature_columns = [str(c) for c in cols]

        stats = meta.get("amount_stats")
        if isinstance(stats, dict) and "mean" in stats and "std" in stats:
            # build_features expects the {amount_mean, amount_std} convention.
            self.amount_stats = {
                "amount_mean": float(stats["mean"]),
                "amount_std": float(stats["std"]),
            }

        best = meta.get("best_model")
        if best:
            self.model_version = str(best)

    def _load_booster(self) -> None:
        try:
            self.booster = self.load_model(self.model_path)
            self.is_ready = True
        except Exception:  # noqa: BLE001 - missing/corrupt model -> degraded mode
            self.booster = None
            self.is_ready = False

    @staticmethod
    def load_model(path: str | Path) -> xgb.Booster:
        """Load an xgboost booster from ``path`` (raises if the file is missing)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"xgboost model not found at {path}")
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def build_row(self, txn: dict) -> pd.DataFrame:
        """Turn one raw transaction dict into the aligned 1-row feature frame."""
        row = {col: float(txn.get(col, 0.0) or 0.0) for col in RAW_FEATURE_COLS}
        raw = pd.DataFrame([row])
        X, _ = build_features(raw, stats=self.amount_stats)
        # Align to the exact training column order (fill any gaps with 0.0).
        X = X.reindex(columns=self.feature_columns, fill_value=0.0)
        return X

    def score(self, txn: dict) -> float:
        """Return P(fraud) in [0, 1] for a single raw transaction dict."""
        if not self.is_ready or self.booster is None:
            raise RuntimeError(
                "ScoringAgent is not ready: xgboost model unavailable at "
                f"{self.model_path}. Train the model or provide a valid model_path."
            )
        X = self.build_row(txn)
        dmatrix = xgb.DMatrix(X.to_numpy(dtype="float64"))
        pred = self.booster.predict(dmatrix)
        prob = float(np.asarray(pred).ravel()[0])
        # Clip for numerical safety (margin/regression objectives can drift).
        return float(min(1.0, max(0.0, prob)))
