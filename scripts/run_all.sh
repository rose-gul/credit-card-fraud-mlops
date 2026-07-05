#!/usr/bin/env sh
# ---------------------------------------------------------------------------
# run_all.sh — end-to-end convenience runner for the fraud-scoring pipeline.
#
# POSIX sh compatible. Run from the repo root:  sh scripts/run_all.sh
#
# Prerequisites:
#   * pip install -r requirements.txt
#   * Kaggle credentials at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY)
#     for the download and GPU-training steps.
#
# Note: the training step runs REMOTELY on a Kaggle GPU kernel; this script only
# pushes the kernel and pulls the resulting artifacts back into ./models.
# ---------------------------------------------------------------------------
set -eu

# Ensure the src/ layout is importable even without an editable install.
export PYTHONPATH="${PYTHONPATH:-src}"

echo "==> [1/7] Download raw dataset (Kaggle -> data/raw/creditcard.csv)"
python -m fraud_scoring.data.download

echo "==> [2/7] Preprocess (stratified split + scale -> data/processed/*.parquet)"
python -m fraud_scoring.data.preprocess

echo "==> [3/7] Train on Kaggle GPU (push kernel, poll, pull artifacts -> models/)"
# XGBoost vs PyTorch MLP; artifacts land in ./models. Remote GPU, not local.
python -m fraud_scoring.pipeline.run_kaggle_gpu

echo "==> [4/7] Benchmark models (XGBoost vs MLP by PR-AUC)"
python -m fraud_scoring.models.benchmark

echo "==> [5/7] Register the winning model in MLflow (fraud-scorer)"
python -m fraud_scoring.models.registry

echo "==> [6/7] Evaluate (governed metrics + decision bands -> metrics/eval.json)"
python -m fraud_scoring.evaluate.evaluate

echo "==> [7/7] Launch the serving API on http://0.0.0.0:8000"
# Foreground; Ctrl-C to stop. POST transactions to /score.
exec uvicorn fraud_scoring.serving.api:app --host 0.0.0.0 --port 8000
