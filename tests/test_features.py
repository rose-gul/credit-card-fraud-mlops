"""Tests for the feature-engineering contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_scoring.features.build_features import (
    FEATURE_COLUMNS,
    RAW_FEATURE_COLS,
    build_features,
    feature_names,
)


def _raw(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {"Time": rng.uniform(0, 100_000, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n)
    data["Amount"] = np.abs(rng.gamma(2.0, 50.0, n))
    return pd.DataFrame(data)


def test_feature_columns_count_and_order():
    assert FEATURE_COLUMNS == feature_names()
    assert len(FEATURE_COLUMNS) == 32
    expected = [f"V{i}" for i in range(1, 29)] + [
        "Amount",
        "Amount_log",
        "Hour",
        "Amount_z",
    ]
    assert FEATURE_COLUMNS == expected


def test_build_features_shape_and_columns():
    df = _raw(25)
    X, stats = build_features(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (25, len(FEATURE_COLUMNS))
    assert "amount_mean" in stats and "amount_std" in stats


def test_build_features_deterministic():
    df = _raw(30)
    X1, _ = build_features(df)
    X2, _ = build_features(df)
    pd.testing.assert_frame_equal(X1, X2)


def test_build_features_applies_supplied_stats():
    df = _raw(10)
    stats = {"amount_mean": 100.0, "amount_std": 25.0}
    X, out_stats = build_features(df, stats=stats)
    assert out_stats["amount_mean"] == 100.0
    assert out_stats["amount_std"] == 25.0
    expected_z = (df["Amount"].to_numpy() - 100.0) / 25.0
    np.testing.assert_allclose(X["Amount_z"].to_numpy(), expected_z)


def test_raw_feature_cols_are_inputs():
    assert RAW_FEATURE_COLS[0] == "Time"
    assert "Amount" in RAW_FEATURE_COLS
    assert len(RAW_FEATURE_COLS) == 30
