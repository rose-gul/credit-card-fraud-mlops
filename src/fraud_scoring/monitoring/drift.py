"""Population-Stability-Index (PSI) data-drift monitoring.

Compares a *reference* distribution (typically the training data) against a
*current* distribution (recent production traffic) feature-by-feature and flags
drift using the PSI bands from ``params.monitoring``.

PSI interpretation (industry convention, matches the configured bands):
    PSI < 0.10          -> no significant shift  (ok)
    0.10 <= PSI < 0.25  -> moderate shift        (warn)
    PSI >= 0.25         -> major shift            (alert)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from fraud_scoring.config import METRICS_DIR, PARAMS, ensure_dirs
from fraud_scoring.features.build_features import FEATURE_COLUMNS

DRIFT_PATH = METRICS_DIR / "drift_report.json"

_PSI_WARN = float(PARAMS["monitoring"]["psi_warn"])
_PSI_ALERT = float(PARAMS["monitoring"]["psi_alert"])

# Small epsilon so empty bins don't blow up the log ratio.
_EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between an ``expected`` and ``actual`` sample.

    Bin edges are quantiles of ``expected`` (equal-frequency binning), so the PSI
    is robust to skewed features. Returns a single non-negative float; larger
    means more distributional shift.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if expected.size == 0 or actual.size == 0:
        return 0.0

    # Quantile edges from the reference; collapse duplicates (constant regions).
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 2:
        # Degenerate (near-constant) reference feature -> no measurable drift.
        return 0.0
    # Open the outer edges so out-of-range current values are still counted.
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_frac = exp_counts / max(expected.size, 1)
    act_frac = act_counts / max(actual.size, 1)

    exp_frac = np.clip(exp_frac, _EPS, None)
    act_frac = np.clip(act_frac, _EPS, None)

    return float(np.sum((act_frac - exp_frac) * np.log(act_frac / exp_frac)))


def _classify(value: float) -> str:
    if value >= _PSI_ALERT:
        return "alert"
    if value >= _PSI_WARN:
        return "warn"
    return "ok"


def dataset_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] = FEATURE_COLUMNS,
    bins: int = 10,
) -> dict[str, Any]:
    """Per-feature PSI drift report with an overall status.

    Only features present in *both* frames are scored. Overall status is the most
    severe per-feature status (alert > warn > ok). The report is written to
    ``metrics/drift_report.json`` and also returned.
    """
    ensure_dirs()

    per_feature: dict[str, dict[str, Any]] = {}
    counts = {"ok": 0, "warn": 0, "alert": 0}

    for feat in features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue
        value = psi(
            reference_df[feat].to_numpy(),
            current_df[feat].to_numpy(),
            bins=bins,
        )
        status = _classify(value)
        counts[status] += 1
        per_feature[feat] = {"psi": value, "status": status}

    if counts["alert"] > 0:
        overall = "alert"
    elif counts["warn"] > 0:
        overall = "warn"
    else:
        overall = "ok"

    drifted = [f for f, r in per_feature.items() if r["status"] != "ok"]

    report: dict[str, Any] = {
        "overall_status": overall,
        "n_features": len(per_feature),
        "counts": counts,
        "drifted_features": drifted,
        "thresholds": {"psi_warn": _PSI_WARN, "psi_alert": _PSI_ALERT},
        "bins": bins,
        "n_reference": int(len(reference_df)),
        "n_current": int(len(current_df)),
        "features": per_feature,
    }

    DRIFT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    from fraud_scoring.config import PROCESSED_DIR

    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Train split not found at '{train_path}'. Run the preprocess stage "
            "first (`python -m fraud_scoring.data.preprocess`)."
        )

    df = pd.read_parquet(train_path)
    half = len(df) // 2
    reference = df.iloc[:half].reset_index(drop=True)
    current = df.iloc[half:].reset_index(drop=True).copy()

    # Inject synthetic drift: inflate Amount in the "current" half.
    if "Amount" in current.columns:
        current["Amount"] = current["Amount"] * 1.5 + 3.0

    report = dataset_drift(reference, current)
    print(json.dumps(
        {
            "overall_status": report["overall_status"],
            "counts": report["counts"],
            "drifted_features": report["drifted_features"],
        },
        indent=2,
    ))
    print(f"wrote {DRIFT_PATH}")
