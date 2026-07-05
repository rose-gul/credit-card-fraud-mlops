"""Self-contained Kaggle GPU training kernel for fraud scoring.

This script runs INSIDE a Kaggle GPU kernel and must NOT import anything from
the ``fraud_scoring`` package -- Kaggle has no access to the repo. All logic
(feature engineering, hyperparameters, metrics) is inlined here and kept in sync
with ``params.yaml`` and the shared FEATURE CONTRACT.

Inputs  : /kaggle/input/creditcardfraud/creditcard.csv
Outputs : /kaggle/working/{model_xgb.json, model_torch.pt,
                            feature_importance.json, metrics.json}
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Constants (mirror params.yaml -- keep in sync with the repo).
# --------------------------------------------------------------------------- #
SEED = 42
TEST_SIZE = 0.2

INPUT_CSV = "/kaggle/input/creditcardfraud/creditcard.csv"
WORKING_DIR = "/kaggle/working"

# Ordered feature contract: V1..V28, Amount, Amount_log, Hour, Amount_z.
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = V_COLS + ["Amount", "Amount_log", "Hour", "Amount_z"]

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=1.0,
    eval_metric="aucpr",
)

TORCH_HIDDEN_DIMS = [128, 64, 32]
TORCH_DROPOUT = 0.3
TORCH_LR = 0.001
TORCH_WEIGHT_DECAY = 1e-5
TORCH_BATCH_SIZE = 2048
TORCH_EPOCHS = 25
TORCH_POS_WEIGHT_CAP = 50.0


# --------------------------------------------------------------------------- #
# Reproducibility.
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Feature engineering (FEATURE CONTRACT).
# --------------------------------------------------------------------------- #
def build_features(
    df: pd.DataFrame, amount_mean: float, amount_std: float
) -> pd.DataFrame:
    """Build the ordered model-input feature frame from a raw dataframe.

    ``amount_mean`` / ``amount_std`` are fit on TRAIN and reused for TEST so the
    z-score has no leakage. Raw ``Time`` is dropped from the model input.
    """
    out = pd.DataFrame(index=df.index)
    for col in V_COLS:
        out[col] = df[col].astype(np.float64)
    amount = df["Amount"].astype(np.float64)
    out["Amount"] = amount
    out["Amount_log"] = np.log1p(amount)
    out["Hour"] = (df["Time"].astype(np.float64) // 3600) % 24
    # Guard against a degenerate (zero) std.
    safe_std = amount_std if amount_std > 0 else 1.0
    out["Amount_z"] = (amount - amount_mean) / safe_std
    return out[FEATURE_COLUMNS]


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Recall (TPR) achievable at or below a target false-positive rate."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def precision_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float
) -> float:
    """Best precision achievable while holding recall >= target."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = recall >= target_recall
    if not mask.any():
        return 0.0
    return float(precision[mask].max())


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, device: str
) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, y_score, 0.01),
        "precision_at_recall_90": precision_at_recall(y_true, y_score, 0.90),
        "device": device,
    }


# --------------------------------------------------------------------------- #
# XGBoost (GPU with CPU fallback).
# --------------------------------------------------------------------------- #
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
) -> tuple[xgb.XGBClassifier, str]:
    """Train XGBoost on GPU, falling back to CPU if CUDA is unavailable."""

    def _make(device: str) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            tree_method="hist",
            device=device,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            n_jobs=-1,
            **XGB_PARAMS,
        )

    try:
        model = _make("cuda")
        model.fit(X_train, y_train)
        device = "cuda"
        print("[xgboost] trained on CUDA")
    except Exception as exc:  # noqa: BLE001 -- robust GPU->CPU fallback
        print(f"[xgboost] CUDA training failed ({exc!r}); falling back to CPU")
        model = _make("cpu")
        model.fit(X_train, y_train)
        device = "cpu"
    return model, device


# --------------------------------------------------------------------------- #
# PyTorch MLP (GPU with CPU fallback).
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    pos_weight_value: float,
) -> tuple[MLP, np.ndarray, str]:
    """Train the MLP; return (model, test_scores, device).

    Inputs are standardized with ``input_mean`` / ``input_std`` (fit on train).
    Returns predicted fraud probabilities for the test split.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[torch] using device: {device}")

    # Standardize (mean/std fit on train, applied to both splits).
    Xtr = (X_train - input_mean) / input_std
    Xte = (X_test - input_mean) / input_std

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=TORCH_BATCH_SIZE,
        shuffle=True,
    )

    model = MLP(X_train.shape[1], TORCH_HIDDEN_DIMS, TORCH_DROPOUT).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TORCH_LR, weight_decay=TORCH_WEIGHT_DECAY
    )
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(1, TORCH_EPOCHS + 1):
        running = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"[torch] epoch {epoch:02d}/{TORCH_EPOCHS}  loss={running / n:.6f}")

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(Xte_t)).cpu().numpy()
    return model, scores, device


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(WORKING_DIR, exist_ok=True)
    set_seed(SEED)

    print("=" * 70)
    print("Fraud-scoring GPU training kernel")
    print(f"torch version           : {torch.__version__}")
    print(f"torch.cuda.is_available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device             : {torch.cuda.get_device_name(0)}")
    print(f"xgboost version         : {xgb.__version__}")
    print("=" * 70)

    # ---- Load & split -----------------------------------------------------
    df = pd.read_csv(INPUT_CSV)
    print(f"loaded {len(df):,} rows from {INPUT_CSV}")

    y = df["Class"].astype(int).values
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"train rows: {len(train_df):,}  test rows: {len(test_df):,}")

    # ---- Amount stats fit on TRAIN ---------------------------------------
    amount_mean = float(train_df["Amount"].mean())
    amount_std = float(train_df["Amount"].std())

    X_train_df = build_features(train_df, amount_mean, amount_std)
    X_test_df = build_features(test_df, amount_mean, amount_std)
    X_train = X_train_df.values.astype(np.float64)
    X_test = X_test_df.values.astype(np.float64)

    # ---- Class imbalance --------------------------------------------------
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    torch_pos_weight = min(scale_pos_weight, TORCH_POS_WEIGHT_CAP)
    print(
        f"n_neg={n_neg:,}  n_pos={n_pos:,}  "
        f"scale_pos_weight={scale_pos_weight:.2f}  "
        f"torch_pos_weight={torch_pos_weight:.2f}"
    )

    # ---- XGBoost ----------------------------------------------------------
    xgb_model, xgb_device = train_xgboost(X_train, y_train, scale_pos_weight)
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_scores, xgb_device)

    # ---- PyTorch MLP ------------------------------------------------------
    input_mean = X_train.mean(axis=0)
    input_std = X_train.std(axis=0)
    input_std = np.where(input_std > 0, input_std, 1.0)  # guard zero-variance

    torch_model, torch_scores, torch_device = train_torch(
        X_train, y_train, X_test, input_mean, input_std, torch_pos_weight
    )
    torch_metrics = compute_metrics(y_test, torch_scores, torch_device)

    # ---- Best model by pr_auc --------------------------------------------
    best_model = (
        "xgboost" if xgb_metrics["pr_auc"] >= torch_metrics["pr_auc"] else "torch"
    )

    # ---- Persist artifacts ------------------------------------------------
    # XGBoost booster.
    xgb_model.get_booster().save_model(os.path.join(WORKING_DIR, "model_xgb.json"))

    # Torch state_dict + arch/meta (incl. input scaler + feature order).
    torch_meta = {
        "state_dict": torch_model.state_dict(),
        "arch": {
            "input_dim": X_train.shape[1],
            "hidden_dims": TORCH_HIDDEN_DIMS,
            "dropout": TORCH_DROPOUT,
        },
        "feature_columns": FEATURE_COLUMNS,
        "input_scaler": {
            "mean": input_mean.tolist(),
            "std": input_std.tolist(),
        },
        "amount_stats": {"mean": amount_mean, "std": amount_std},
        "seed": SEED,
    }
    torch.save(torch_meta, os.path.join(WORKING_DIR, "model_torch.pt"))

    # Feature importances (gain) keyed by feature name.
    booster = xgb_model.get_booster()
    gain_map = booster.get_score(importance_type="gain")
    # Booster keys are the training feature names (we passed a DataFrame? -> no,
    # numpy). Map fN -> feature name defensively for either convention.
    feature_importance: dict[str, float] = {}
    for name in FEATURE_COLUMNS:
        feature_importance[name] = 0.0
    for key, val in gain_map.items():
        if key in feature_importance:
            feature_importance[key] = float(val)
        elif key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if 0 <= idx < len(FEATURE_COLUMNS):
                feature_importance[FEATURE_COLUMNS[idx]] = float(val)
    with open(os.path.join(WORKING_DIR, "feature_importance.json"), "w") as fh:
        json.dump(feature_importance, fh, indent=2)

    # Metrics bundle (shared schema).
    metrics = {
        "models": {
            "xgboost": xgb_metrics,
            "torch": torch_metrics,
        },
        "feature_columns": FEATURE_COLUMNS,
        "amount_stats": {"mean": amount_mean, "std": amount_std},
        "torch_input_stats": {
            "mean": input_mean.tolist(),
            "std": input_std.tolist(),
        },
        "best_model": best_model,
        "dataset": "mlg-ulb/creditcardfraud",
        "seed": SEED,
    }
    with open(os.path.join(WORKING_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ---- Summary ----------------------------------------------------------
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("-" * 70)
    for name, m in (("xgboost", xgb_metrics), ("torch", torch_metrics)):
        print(
            f"{name:8s} [{m['device']:>4s}]  "
            f"roc_auc={m['roc_auc']:.4f}  pr_auc={m['pr_auc']:.4f}  "
            f"f1={m['f1']:.4f}  recall@1%fpr={m['recall_at_1pct_fpr']:.4f}  "
            f"prec@recall90={m['precision_at_recall_90']:.4f}"
        )
    print("-" * 70)
    print(f"best_model (by pr_auc): {best_model}")
    print("artifacts written to /kaggle/working:")
    for fname in (
        "model_xgb.json",
        "model_torch.pt",
        "feature_importance.json",
        "metrics.json",
    ):
        print(f"  - {fname}")
    print("=" * 70)


if __name__ == "__main__":
    main()
