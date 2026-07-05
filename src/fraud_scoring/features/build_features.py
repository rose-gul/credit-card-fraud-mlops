"""Feature engineering for the fraud-scoring model.

Single source of truth for the model-input schema. Both the Kaggle training kernel
and the serving/inference path call :func:`build_features` so train and serve stay
consistent. Engineered features are toggled via ``params.features.*``.

Fit vs. apply
-------------
``build_features(df, stats=None)`` **fits** — it computes the Amount mean/std from
``df`` and returns them in ``stats``. ``build_features(df, stats=<dict>)`` **applies**
previously-fitted statistics, which is what inference/test must do to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_scoring.config import PARAMS

TARGET = "Class"

# Raw predictors present in the source csv (everything except the target).
RAW_FEATURE_COLS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Engineered-feature toggles (read once at import).
_FEAT = PARAMS["features"]
_LOG_AMOUNT: bool = bool(_FEAT.get("log_amount", False))
_HOUR_OF_DAY: bool = bool(_FEAT.get("hour_of_day", False))
_AMOUNT_ZSCORE: bool = bool(_FEAT.get("amount_zscore", False))


def _build_feature_columns() -> list[str]:
    """Final, ordered model-input columns. ``Time`` is dropped (only used for Hour)."""
    cols: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    if _LOG_AMOUNT:
        cols.append("Amount_log")
    if _HOUR_OF_DAY:
        cols.append("Hour")
    if _AMOUNT_ZSCORE:
        cols.append("Amount_z")
    return cols


FEATURE_COLUMNS: list[str] = _build_feature_columns()


def feature_names() -> list[str]:
    """Return the final ordered list of model-input feature columns."""
    return list(FEATURE_COLUMNS)


def build_features(
    df: pd.DataFrame, stats: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Construct the model-input matrix ``X`` with exactly :data:`FEATURE_COLUMNS`.

    Parameters
    ----------
    df:
        Input frame containing at least :data:`RAW_FEATURE_COLS`.
    stats:
        When ``None`` the Amount mean/std are fitted from ``df`` and returned.
        When provided they are applied as-is (inference/test path).

    Returns
    -------
    (X, stats):
        ``X`` has columns exactly equal to :data:`FEATURE_COLUMNS` (same order);
        ``stats`` carries the fitted statistics for reuse.
    """
    out = pd.DataFrame(index=df.index)

    # Passthrough raw predictors (Time excluded from the model input).
    for col in [f"V{i}" for i in range(1, 29)] + ["Amount"]:
        out[col] = df[col].to_numpy()

    amount = df["Amount"].to_numpy(dtype="float64")

    # Fit or reuse the Amount statistics used by the z-score feature.
    if stats is None:
        mean = float(np.mean(amount))
        std = float(np.std(amount))
        stats = {"amount_mean": mean, "amount_std": std}
    else:
        stats = dict(stats)
        mean = float(stats["amount_mean"])
        std = float(stats["amount_std"])

    if _LOG_AMOUNT:
        out["Amount_log"] = np.log1p(np.clip(amount, a_min=0.0, a_max=None))

    if _HOUR_OF_DAY:
        time_seconds = df["Time"].to_numpy(dtype="float64")
        out["Hour"] = (np.floor_divide(time_seconds, 3600) % 24).astype("float64")

    if _AMOUNT_ZSCORE:
        denom = std if std > 0.0 else 1.0
        out["Amount_z"] = (amount - mean) / denom

    # Guarantee exact column set and order the downstream model expects.
    X = out[FEATURE_COLUMNS]
    return X, stats


if __name__ == "__main__":
    from fraud_scoring.config import PROCESSED_DIR

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    X, stats = build_features(train)
    print("X shape:", X.shape)
    print("stats:", stats)
    print("FEATURE_COLUMNS:", FEATURE_COLUMNS)
