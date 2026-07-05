"""Explanation agent: produce regulator-style "adverse action" reason codes for a
scored transaction.

Strategy (fastest reliable path wins):

1. ``shap.TreeExplainer`` when the ``shap`` package is importable.
2. XGBoost's built-in exact TreeSHAP via ``booster.predict(..., pred_contribs=True)``
   — always available with xgboost and needs no extra dependency.
3. A cheap ``|feature_zscore * importance|`` heuristic as a last resort.

Each reason code is ``{"feature", "contribution", "direction"}`` where ``direction``
is ``"increases_risk"`` / ``"decreases_risk"`` and the list is sorted by
``|contribution|`` descending.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from fraud_scoring.config import MODELS_DIR
from fraud_scoring.features.build_features import FEATURE_COLUMNS

DEFAULT_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"


class ExplanationAgent:
    """Generate reason codes for a transaction using the loaded booster."""

    def __init__(
        self,
        scoring_agent=None,
        booster: xgb.Booster | None = None,
        feature_columns: list[str] | None = None,
        importance_path: str | Path | None = None,
    ) -> None:
        self.scoring_agent = scoring_agent
        self.booster = booster if booster is not None else getattr(scoring_agent, "booster", None)
        if feature_columns is not None:
            self.feature_columns = list(feature_columns)
        else:
            self.feature_columns = list(
                getattr(scoring_agent, "feature_columns", FEATURE_COLUMNS)
            )
        self.importance_path = (
            Path(importance_path) if importance_path else DEFAULT_IMPORTANCE_PATH
        )
        self.importances = self._load_importances()
        self._tree_explainer = None  # lazily built shap.TreeExplainer

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _load_importances(self) -> dict[str, float]:
        """Feature-importance (gain) map: prefer the saved json, else the booster."""
        imp: dict[str, float] = {name: 0.0 for name in self.feature_columns}

        if self.importance_path.exists():
            try:
                raw = json.loads(self.importance_path.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    if k in imp:
                        imp[k] = float(v)
                return imp
            except Exception:  # noqa: BLE001
                pass

        if self.booster is not None:
            try:
                gain = self.booster.get_score(importance_type="gain")
                for key, val in gain.items():
                    if key in imp:
                        imp[key] = float(val)
                    elif key.startswith("f") and key[1:].isdigit():
                        idx = int(key[1:])
                        if 0 <= idx < len(self.feature_columns):
                            imp[self.feature_columns[idx]] = float(val)
            except Exception:  # noqa: BLE001
                pass
        return imp

    def _row(self, txn: dict) -> pd.DataFrame:
        """Aligned 1-row feature frame; reuse the scoring agent when available."""
        if self.scoring_agent is not None and hasattr(self.scoring_agent, "build_row"):
            return self.scoring_agent.build_row(txn)
        # Standalone fallback (should rarely happen).
        from fraud_scoring.features.build_features import RAW_FEATURE_COLS, build_features

        row = {col: float(txn.get(col, 0.0) or 0.0) for col in RAW_FEATURE_COLS}
        X, _ = build_features(pd.DataFrame([row]))
        return X.reindex(columns=self.feature_columns, fill_value=0.0)

    # ------------------------------------------------------------------ #
    # Contribution estimators
    # ------------------------------------------------------------------ #
    def _shap_contributions(self, X: pd.DataFrame) -> np.ndarray | None:
        """TreeSHAP via the shap package; ``None`` if unavailable/failed."""
        try:
            import shap  # noqa: PLC0415 - optional dependency, guarded on purpose
        except Exception:  # noqa: BLE001
            return None
        try:
            if self._tree_explainer is None:
                self._tree_explainer = shap.TreeExplainer(self.booster)
            values = self._tree_explainer.shap_values(X.to_numpy(dtype="float64"))
            arr = np.asarray(values)
            if arr.ndim == 3:  # (classes, rows, features) for some versions
                arr = arr[-1]
            return arr[0]
        except Exception:  # noqa: BLE001
            return None

    def _xgb_contributions(self, X: pd.DataFrame) -> np.ndarray | None:
        """Exact TreeSHAP baked into xgboost; last column is the bias term."""
        if self.booster is None:
            return None
        try:
            dmatrix = xgb.DMatrix(X.to_numpy(dtype="float64"))
            contribs = self.booster.predict(dmatrix, pred_contribs=True)
            arr = np.asarray(contribs)
            if arr.ndim == 3:  # multiclass shape (rows, classes, features+1)
                arr = arr[:, -1, :]
            # Drop the trailing bias/expected-value column.
            return arr[0, :-1]
        except Exception:  # noqa: BLE001
            return None

    def _heuristic_contributions(self, X: pd.DataFrame) -> np.ndarray:
        """|zscore * importance| fallback that never fails."""
        values = X.to_numpy(dtype="float64").ravel()
        imp = np.array([self.importances.get(c, 0.0) for c in self.feature_columns])
        if not np.any(imp):
            imp = np.ones_like(values)
        # A rough standardization keeps large-magnitude raw features from dominating.
        scale = np.where(np.abs(values) > 0, np.abs(values), 1.0)
        z = values / np.maximum(scale.mean(), 1e-9)
        return z * imp

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def explain(self, txn: dict, top_n: int = 5) -> list[dict]:
        """Return the top-N adverse-action reason codes for ``txn``."""
        if self.booster is None:
            return []

        X = self._row(txn)
        contribs = self._shap_contributions(X)
        if contribs is None:
            contribs = self._xgb_contributions(X)
        if contribs is None:
            contribs = self._heuristic_contributions(X)

        contribs = np.asarray(contribs, dtype="float64").ravel()
        n = min(len(contribs), len(self.feature_columns))

        codes: list[dict] = []
        for i in range(n):
            c = float(contribs[i])
            codes.append(
                {
                    "feature": self.feature_columns[i],
                    "contribution": c,
                    "direction": "increases_risk" if c >= 0 else "decreases_risk",
                }
            )
        codes.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return codes[: max(1, top_n)]
