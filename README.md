# fraud-scoring

**An end-to-end MLOps platform for credit-card fraud scoring.** Trains on Kaggle GPU,
tracks and registers models with MLflow, versions data and models with DVC, and serves
real-time decisions through a multi-agent FastAPI service with SHAP reason codes,
threshold/rule review, and a sanctions-style watchlist hook.

![python](https://img.shields.io/badge/python-3.11-blue)
![style](https://img.shields.io/badge/lint-ruff-black)
![tests](https://img.shields.io/badge/tests-pytest-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

> Domain: finance / regulated. Dataset: [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
> — 284,807 transactions, fraud rate ~0.17% (severe class imbalance).

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Example Use Cases](#example-use-cases)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Learning Handbook](#learning-handbook)
- [Security Notes](#security-notes)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Why This Project Exists

Credit-card fraud detection is a textbook example of a hard, high-stakes machine-learning
problem: fewer than 2 transactions in 1,000 are fraudulent, the cost of a false negative
(missed fraud) is very different from a false positive (a blocked good customer), and the
whole system operates in a **regulated** environment where every automated decline may need
an explainable, auditable justification (adverse-action reason codes).

Most tutorials stop at "train a classifier and print the ROC-AUC." This project is the
opposite: it is a complete, reproducible **MLOps** platform that treats the model as one
component in a governed lifecycle:

- **Reproducible data + training** — every artifact (raw data, processed splits, models,
  metrics) is versioned with DVC and tracked in MLflow, so any result can be regenerated
  and audited.
- **GPU training without owning a GPU** — training runs remotely on **Kaggle GPU kernels**
  via the Kaggle Kernels API, benchmarking **XGBoost** against a **PyTorch MLP** and
  registering the winner by **PR-AUC** (the right metric for extreme imbalance).
- **Explainable, rule-aware serving** — a multi-agent runtime turns a raw score into a
  governed decision (`APPROVE` / `REVIEW` / `BLOCK`) with SHAP reason codes and a
  sanctions-style watchlist flag, the way a real authorization system would.
- **Operable in production** — Docker, GitHub Actions CI, PSI-based drift monitoring, and a
  Prometheus `/metrics` endpoint are first-class, not afterthoughts.

It also demonstrates a newer idea: the repository is **built and maintained by Claude Code
subagents** (see [`.claude/agents/`](.claude/agents)). Each subagent owns a slice of the
pipeline — data, features, training, evaluation, serving — mirroring how a small platform
team would divide the work. See [How It Works](#how-it-works).

## Features

- **Imbalance-aware modeling** — PR-AUC as the primary metric, `scale_pos_weight` for
  XGBoost and a capped `pos_weight` for the MLP's `BCEWithLogitsLoss`.
- **Two-model benchmark** — GPU-accelerated **XGBoost** vs a **PyTorch MLP**; the better
  model (by PR-AUC) is promoted to the MLflow registry.
- **Remote GPU training** — a self-contained Kaggle kernel (`kaggle/train_kernel.py`) is
  pushed, polled, and harvested by a local orchestrator — no local GPU required.
- **Single feature contract** — 32 model features (`V1..V28, Amount, Amount_log, Hour,
  Amount_z`) defined once in `features/build_features.py` and mirrored in the kernel, so
  **train/serve parity** is guaranteed.
- **Experiment tracking + registry** — MLflow logs params, metrics, and artifacts and hosts
  the registered `fraud-scorer` model.
- **Reproducible pipeline** — DVC stages `download → preprocess → train_gpu → evaluate`
  wired through `dvc.yaml`; `dvc repro` rebuilds the world.
- **Multi-agent serving** — orchestrator → scoring → explanation → review agents produce a
  structured decision with reason codes, flags, and model version.
- **Explainability** — SHAP-based per-transaction reason codes suitable for adverse-action
  notices.
- **Governance & rules** — configurable decision bands, high-amount review rule, and a
  sanctions-style watchlist hook in the review agent.
- **Monitoring** — PSI drift detection with warn/alert bands and a Prometheus `/metrics`
  endpoint.
- **Ops-ready** — Dockerfile, docker-compose (serving + optional MLflow UI), GitHub Actions
  CI (ruff + pytest + docker build).
- **Built by subagents** — six Claude Code subagents in `.claude/agents/` build and maintain
  the repo.

## How It Works

The platform has a **build-time** plane (how the repo and pipeline are created and
maintained) and a **run-time** plane (how a transaction is scored). Build-time work is
divided across Claude Code subagents; run-time serving is a chain of runtime agents.

```
                        BUILD-TIME  (Claude Code subagents in .claude/agents/)
        ┌───────────────────────────────────────────────────────────────────────────┐
        │                          mlops-orchestrator                                 │
        │      (owns params.yaml + config.py, integration & verification)             │
        │   ┌────────────┬───────────────┬───────────────┬────────────┬───────────┐  │
        │   │data-       │ feature-      │ model-        │ evaluator  │ deployer  │  │
        │   │engineer    │ engineer      │ trainer       │            │           │  │
        │   └────────────┴───────────────┴───────────────┴────────────┴───────────┘  │
        └───────────────────────────────────────────────────────────────────────────┘

                                     RUN-TIME  DATA + MODEL FLOW

  ┌──────────────┐    ┌───────────────┐    ┌───────────────────────┐    ┌──────────────┐
  │ Kaggle       │    │  DVC stages   │    │  Kaggle GPU kernel     │    │   MLflow     │
  │ dataset  ───►│───►│ download      │───►│  (remote GPU)          │───►│  tracking +  │
  │ creditcard   │    │ preprocess    │    │  XGBoost vs PyTorch    │    │  registry    │
  │ .csv         │    │ (split+scale) │    │  MLP  → best by PR-AUC │    │ fraud-scorer │
  └──────────────┘    └───────────────┘    └───────────────────────┘    └──────┬───────┘
        data/raw        data/processed        models/*.json,*.pt               │
                                                                               ▼
                              MULTI-AGENT FastAPI SERVING  (POST /score)
        ┌──────────────────────────────────────────────────────────────────────────┐
        │  orchestrator ─► scoring-agent ─► explanation-agent ─► review-agent        │
        │   (routes)       (model prob)     (SHAP reason codes)  (bands+rules+       │
        │                                                         watchlist flag)     │
        │                                                                            │
        │  Decision = { score, band: APPROVE|REVIEW|BLOCK, reason_codes[],           │
        │               flags[], model_version }                                     │
        └───────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
                       MONITORING:  PSI drift  +  Prometheus /metrics
```

**Build-time.** A top-level `mlops-orchestrator` subagent owns `params.yaml` and
`config.py` and coordinates five specialist subagents (`data-engineer`,
`feature-engineer`, `model-trainer`, `evaluator`, `deployer`). Each owns a subpackage and a
set of DVC stages; the orchestrator integrates and verifies their work. See
[`.claude/agents/`](.claude/agents).

**Run-time.** A transaction posted to `/score` flows through the orchestrator to the
scoring agent (loads the registered model, produces a fraud probability), the explanation
agent (SHAP reason codes), and the review agent (applies decision bands, the high-amount
rule, and the sanctions-style watchlist) to yield a structured, auditable decision.

## Requirements

- **Python 3.11** (project targets 3.10+; CI runs 3.11).
- A **Kaggle account** with API credentials (`kaggle.json`) — required only to download data
  and to run remote GPU training. Serving and tests do **not** need Kaggle.
- **Git** and (optionally) **DVC** for the reproducible pipeline.
- **Docker** (optional) for containerized serving.
- Core Python dependencies (see [`requirements.txt`](requirements.txt)): `numpy`, `pandas`,
  `pyarrow`, `scikit-learn`, `xgboost`, `torch`, `mlflow`, `dvc`, `kaggle`, `shap`,
  `fastapi`, `uvicorn`, `pydantic` / `pydantic-settings`, `prometheus-client`, `PyYAML`,
  `python-dotenv`, `joblib`, plus `pytest`, `httpx`, `ruff` for development.

> **Note:** `torch` and `xgboost` are large wheels. The first install may take several
> minutes and a few GB of disk.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<you>/fraud-scoring.git
cd fraud-scoring

# 2. Create a virtualenv and install dependencies
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Kaggle credentials (needed only for data download + GPU training)
#    Download kaggle.json from https://www.kaggle.com/settings/account
#    and place it at ~/.kaggle/kaggle.json  (Windows: %USERPROFILE%\.kaggle\kaggle.json)
#    Or export KAGGLE_USERNAME / KAGGLE_KEY. Never commit this file.

# 4. Get the data
python -m fraud_scoring.data.download

# 5. Build the train/test splits (stratified) + fit the scaler
python -m fraud_scoring.data.preprocess

# 6. Train on Kaggle GPU (pushes the kernel, polls, pulls artifacts into ./models)
python -m fraud_scoring.pipeline.run_kaggle_gpu

# 7. Benchmark the two models and register the winner (by PR-AUC) in MLflow
python -m fraud_scoring.models.benchmark
python -m fraud_scoring.models.registry

# 8. Serve
uvicorn fraud_scoring.serving.api:app --host 0.0.0.0 --port 8000
```

Score a transaction (fields are `Time`, `V1..V28`, `Amount`):

```bash
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
        "Time": 40000,
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
        "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
        "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
        "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
        "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
      }'
```

## Configuration

Configuration has three layers, merged by [`src/fraud_scoring/config.py`](src/fraud_scoring/config.py):
static hyperparameters in `params.yaml`, runtime/secret settings from `.env` (prefix
`FRAUD_`), and Kaggle credentials in `kaggle.json`.

### `params.yaml` (static hyperparameters — DVC-tracked)

| Key | Example | Meaning |
| --- | --- | --- |
| `seed` | `42` | Global RNG seed for reproducible splits/training. |
| `data.kaggle_dataset` | `mlg-ulb/creditcardfraud` | Kaggle dataset slug to download. |
| `data.raw_csv` | `creditcard.csv` | Raw file name under `data/raw/`. |
| `data.target` | `Class` | Label column (`1` = fraud). |
| `data.test_size` | `0.2` | Stratified hold-out fraction. |
| `data.scale_cols` | `[Amount, Time]` | Requested scale cols (only `Amount` is scaled; `Time` is kept raw for `Hour`). |
| `features.log_amount` | `true` | Add `Amount_log = log1p(Amount)`. |
| `features.hour_of_day` | `true` | Add `Hour = (Time // 3600) % 24`. |
| `features.amount_zscore` | `true` | Add `Amount_z` from fitted Amount mean/std. |
| `xgboost.*` | `n_estimators=600, max_depth=6, ...` | XGBoost hyperparameters (`eval_metric=aucpr`). |
| `torch.*` | `hidden_dims=[128,64,32], epochs=25, ...` | MLP architecture / training. |
| `torch.pos_weight_cap` | `50.0` | Cap on class-imbalance weighting for MLP stability. |
| `thresholds.approve_below` | `0.30` | `score < 0.30` → `APPROVE`. |
| `thresholds.block_above` | `0.85` | `score >= 0.85` → `BLOCK`; between → `REVIEW`. |
| `thresholds.high_amount_review` | `5000.0` | Large amounts force at least `REVIEW`. |
| `monitoring.psi_warn` / `psi_alert` | `0.10` / `0.25` | PSI drift warning / alert bands. |
| `mlflow.experiment_name` | `fraud-scoring` | MLflow experiment. |
| `mlflow.registered_model_name` | `fraud-scorer` | Registered model name. |
| `kaggle_kernel.*` | `kernel_name, enable_gpu, poll_*` | Remote GPU kernel settings. |

### `.env` (runtime settings — never committed)

Copy [`.env.example`](.env.example) to `.env`. All keys use the `FRAUD_` prefix:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRAUD_MLFLOW_TRACKING_URI` | `file:///./mlruns` | MLflow tracking store. |
| `FRAUD_MODEL_PATH` | `models/model_xgb.json` | Model bundle the API loads. |
| `FRAUD_KAGGLE_USERNAME` | *(unset)* | Kaggle username (else read from `kaggle.json`). |
| `FRAUD_APPROVE_BELOW` | *(unset)* | Override `thresholds.approve_below` without editing `params.yaml`. |
| `FRAUD_BLOCK_ABOVE` | *(unset)* | Override `thresholds.block_above`. |

Kaggle also reads `KAGGLE_USERNAME` / `KAGGLE_KEY` if you prefer env vars over the file.

### `kaggle.json` (API credentials — never committed)

```json
{ "username": "your_kaggle_username", "key": "your_kaggle_api_key" }
```

Place at `~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\kaggle.json`). It is
listed in `.gitignore` — keep it that way.

## Usage

### CLI entry points

| Command | What it does |
| --- | --- |
| `python -m fraud_scoring.data.download` | Download `creditcard.csv` into `data/raw/` via the Kaggle CLI. |
| `python -m fraud_scoring.data.preprocess` | Stratified split, scale `Amount`, write `train/test.parquet` + `scaler.joblib`. |
| `python -m fraud_scoring.pipeline.run_kaggle_gpu` | Push the GPU kernel, poll to completion, pull `models/*` artifacts. |
| `python -m fraud_scoring.models.benchmark` | Compare XGBoost vs MLP on the test split by PR-AUC. |
| `python -m fraud_scoring.models.registry` | Register the winning model as `fraud-scorer` in MLflow. |
| `python -m fraud_scoring.evaluate.evaluate` | Compute governed metrics + decision-band report → `metrics/eval.json`. |
| `python -m fraud_scoring.monitoring.drift` | Compute PSI drift vs a reference distribution. |
| `uvicorn fraud_scoring.serving.api:app` | Launch the multi-agent scoring API. |

### Reproducible pipeline (DVC)

The full DAG lives in [`dvc.yaml`](dvc.yaml). Rebuild everything that is stale:

```bash
dvc repro            # runs download → preprocess → train_gpu → evaluate as needed
dvc dag              # visualize the stage graph
dvc metrics show     # show tracked metrics (preprocess_summary.json, eval.json)
```

### Example `/score` request and response

Request body (a single transaction):

```json
{
  "Time": 40000,
  "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
  "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
  "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
  "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
  "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
  "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
}
```

Response (the multi-agent decision):

```json
{
  "score": 0.9312,
  "band": "BLOCK",
  "reason_codes": [
    { "feature": "V14", "contribution": -0.41, "direction": "increases_risk" },
    { "feature": "V4",  "contribution":  0.28, "direction": "increases_risk" },
    { "feature": "Amount_z", "contribution": 0.12, "direction": "increases_risk" }
  ],
  "flags": ["high_score"],
  "model_version": "fraud-scorer:3"
}
```

`GET /health` returns liveness/model-load status; `GET /metrics` exposes Prometheus
counters and latency histograms.

## Example Use Cases

- **Real-time authorization scoring** — call `/score` inline in the authorization path; use
  the `band` (`APPROVE` / `REVIEW` / `BLOCK`) to approve, step-up, or decline a transaction
  within the network's latency budget.
- **Batch review queue** — score a day's transactions and route everything in the `REVIEW`
  band, sorted by `score`, into an analyst work queue; the high-amount rule guarantees large
  transactions never auto-approve.
- **Adverse-action reason codes for regulators** — when a transaction is declined, the
  `reason_codes[]` (SHAP contributions) provide the human-readable "principal reasons" that
  regulations such as adverse-action notices require.
- **Sanctions-screening integration** — the review agent's watchlist hook flags accounts or
  entities that match a sanctions-style list, forcing a `REVIEW`/`BLOCK` and adding a
  `flags` entry regardless of the model score — a pattern that dovetails with AML/sanctions
  compliance.
- **Model drift alerting** — schedule `monitoring.drift` to compute PSI between live traffic
  and the training reference; breach of `psi_warn` / `psi_alert` raises an alert (and can
  gate an auto-retrain), while `/metrics` feeds Prometheus/Grafana dashboards.

## Testing

```bash
pytest            # runs the full suite (pytest is configured via pyproject.toml)
ruff check .      # lint
```

Tests are designed to run **without Kaggle, without a GPU, and without the real trained
model**:

- A **tiny-model fixture** trains a minimal classifier on a handful of synthetic rows (with
  a few injected positives) so scoring and the agent chain can be exercised in milliseconds.
- **Feature-contract tests** assert `build_features` always emits exactly the 32 columns
  `V1..V28, Amount, Amount_log, Hour, Amount_z` in order, and that fit-vs-apply stats
  prevent leakage.
- **Serving tests** drive the FastAPI app with `httpx`/`TestClient`, asserting the `/score`
  response schema (`score`, `band`, `reason_codes`, `flags`, `model_version`) and that
  decision bands and the high-amount rule behave as configured.
- **Preprocess/config tests** check the stratified split, that only `Amount` is scaled, and
  that `.env`/`params.yaml` overrides merge correctly.

Because everything hangs off the tiny-model fixture, the suite is fast and CI stays green
without any external credentials.

## Project Structure

```
fraud_scoring/
├── README.md                     # this file
├── Dockerfile                    # slim serving image (uvicorn, non-root)
├── docker-compose.yml            # serving (+ optional MLflow UI) services
├── requirements.txt              # Python dependencies
├── pyproject.toml                # build metadata, ruff + pytest config (pythonpath=src)
├── params.yaml                   # static hyperparameters (DVC-tracked)
├── dvc.yaml                      # pipeline DAG: download→preprocess→train_gpu→evaluate
├── .env.example                  # template for FRAUD_* runtime settings
├── .gitignore / .dvcignore       # ignore data/models/secrets
├── .github/
│   └── workflows/ci.yml          # CI: ruff + pytest + docker build
├── scripts/
│   └── run_all.sh                # convenience end-to-end runner
├── .claude/
│   └── agents/                   # six Claude Code build/maintain subagents
│       ├── mlops-orchestrator.md
│       ├── data-engineer.md
│       ├── feature-engineer.md
│       ├── model-trainer.md
│       ├── evaluator.md
│       └── deployer.md
├── kaggle/
│   ├── train_kernel.py           # self-contained GPU kernel (XGBoost + PyTorch MLP)
│   └── kernel-metadata.json      # Kaggle Kernels API metadata
└── src/fraud_scoring/
    ├── config.py                 # merges params.yaml + .env + paths
    ├── data/
    │   ├── download.py           # Kaggle download  (entry point)
    │   └── preprocess.py         # split + scale     (entry point)
    ├── features/
    │   └── build_features.py     # THE feature contract (32 cols)
    ├── models/
    │   ├── benchmark.py          # XGBoost vs MLP by PR-AUC (entry point)
    │   └── registry.py           # MLflow model registry   (entry point)
    ├── evaluate/
    │   └── evaluate.py           # governed metrics + bands (entry point)
    ├── monitoring/
    │   └── drift.py              # PSI drift monitoring     (entry point)
    ├── pipeline/
    │   └── run_kaggle_gpu.py     # remote GPU orchestrator  (entry point)
    └── serving/
        ├── api.py                # FastAPI app (/score, /health, /metrics)
        └── agents/               # orchestrator, scoring, explanation, review agents
```

> `data/`, `models/`, `metrics/`, and `mlruns/` are created at runtime and are DVC-tracked,
> not committed to Git.

## Learning Handbook

A short, opinionated tour of the ideas this repo implements, with pointers to the files that
implement each.

### Imbalanced classification: PR-AUC vs ROC-AUC, `scale_pos_weight`

With a ~0.17% positive rate, a model that predicts "never fraud" scores 99.83% accuracy and
a deceptively high **ROC-AUC**, because ROC-AUC is dominated by the huge negative class.
**PR-AUC** (average precision) focuses on the positive class and is far more sensitive to how
well fraud is actually ranked — which is why the benchmark selects the winner by PR-AUC.
To make the models care about the rare class, XGBoost uses `scale_pos_weight = n_neg / n_pos`
and the MLP uses a capped `pos_weight` in `BCEWithLogitsLoss` (`torch.pos_weight_cap`).
*See `kaggle/train_kernel.py` (`compute_metrics`, `train_xgboost`, `train_torch`) and
`params.yaml`.*

### GPU training on Kaggle kernels

You do not need a local GPU. `pipeline/run_kaggle_gpu.py` finalizes `kernel-metadata.json`,
**pushes** `kaggle/train_kernel.py` to Kaggle Kernels, **polls** status until terminal, and
**pulls** the output artifacts back into `models/`. The kernel is deliberately
self-contained (no `fraud_scoring` imports) because Kaggle has no access to your repo.
*See `src/fraud_scoring/pipeline/run_kaggle_gpu.py` and `kaggle/train_kernel.py`.*

### Experiment tracking & the MLflow registry

Training params, metrics, and artifacts are logged to MLflow; the best model is registered
under `fraud-scorer` so serving can resolve a specific, versioned model. The registry gives
you promotion stages, lineage, and a clear `model_version` to stamp on every decision.
*See `src/fraud_scoring/models/registry.py`, `params.yaml:mlflow.*`.*

### DVC reproducibility

`dvc.yaml` encodes the pipeline as a DAG with explicit deps, params, outs, and metrics.
`dvc repro` reruns only stale stages; because the params are hashed, changing a hyperparameter
invalidates exactly the downstream stages it affects — no more "which data made this model?"
*See `dvc.yaml`, `.dvcignore`.*

### Explainability / SHAP & adverse action

A score alone is not enough in a regulated setting. The explanation agent computes SHAP
contributions to produce per-transaction **reason codes** — the principal features pushing a
decision toward risk — which map directly onto the "principal reasons" an adverse-action
notice must disclose.
*See `src/fraud_scoring/serving/agents/` (explanation agent) and the `reason_codes[]` in the
`/score` response.*

### Drift & PSI

Models decay when live traffic drifts from training data. The **Population Stability Index
(PSI)** compares the distribution of a feature (or the score) now vs a reference; `psi_warn`
(0.10) and `psi_alert` (0.25) bands turn that into actionable alerts and can gate retraining.
*See `src/fraud_scoring/monitoring/drift.py`, `params.yaml:monitoring.*`.*

### The multi-agent serving pattern

Rather than one monolith, serving is a chain of small, single-responsibility agents:
**orchestrator** (routing/assembly) → **scoring** (model probability) → **explanation**
(SHAP reason codes) → **review** (decision bands, high-amount rule, sanctions watchlist).
Each agent is independently testable, and the review agent centralizes governance so policy
changes never touch model code.
*See `src/fraud_scoring/serving/agents/` and `src/fraud_scoring/serving/api.py`.*

## Security Notes

- **Never commit secrets.** `kaggle.json` and `.env` are in `.gitignore`; keep them there.
  Prefer `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars in CI over checked-in files.
- **PII / regulatory considerations.** The public dataset is anonymized (PCA features), but a
  real deployment scores cardholder data. Treat inputs as sensitive: encrypt in transit
  (TLS), avoid logging raw features, apply retention limits, and scope access.
- **Model governance.** Every decision carries a `model_version`; models are versioned in the
  MLflow registry and gated on evaluation metrics before promotion, so declines are traceable
  to an auditable model + config.
- **Sanctions watchlist hook.** The review agent supports a sanctions-style watchlist that can
  force `REVIEW`/`BLOCK` independent of the model score — keep the list source-controlled
  separately, access-restricted, and audited.
- **Container hardening.** The image runs as a **non-root** user, copies only `src/` and
  `params.yaml`, and mounts models **read-only**. In production, pin base images by digest,
  scan images (e.g. Trivy), drop capabilities, and run with a read-only root filesystem.
- **Least privilege.** The serving container needs no Kaggle credentials — only a trained
  model. Do not pass training/data secrets into the serving runtime.

## Roadmap

- [ ] **Feature store** for consistent offline/online feature definitions.
- [ ] **Online inference store** (low-latency lookups for account-level aggregates).
- [ ] **Canary / shadow deploys** with automated rollback on metric regression.
- [ ] **Real streaming ingestion** (Kafka) for continuous scoring.
- [ ] **Auto-retrain on drift** — PSI breach triggers a Kaggle GPU retrain + gated promotion.
- [ ] **SHAP dashboards** for portfolio-level explainability and reason-code monitoring.
- [ ] **Threshold optimization** by expected cost (FP vs FN business cost curves).
- [ ] **Model cards & datasheet** auto-generated per registered version.
- [ ] **Multi-model ensembling** and champion/challenger evaluation.
- [ ] **Kubernetes/Helm** deployment with HPA and Prometheus/Grafana wiring.

## Contributing

We keep the workflow simple and the `main` branch always green.

**Dev setup**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Before you push**

```bash
ruff check .      # lint (config in pyproject.toml; line-length 100)
pytest            # tests must pass without Kaggle/GPU (tiny-model fixture)
```

**Conventions**

- Branch from `main` as `feature/<short-name>` or `fix/<short-name>`; open a PR into `main`.
- Keep commits small and messages neutral and in the first person (e.g. "Add PSI drift
  bands"). One logical change per PR.
- Respect the **feature contract**: any change to model inputs must update
  `features/build_features.py` **and** `kaggle/train_kernel.py` together, with a test.
- New config goes in `params.yaml` (static) or `.env` (runtime), never hard-coded.
- Add or update tests for behavior changes; CI runs `ruff` + `pytest` on every push and PR.
```
