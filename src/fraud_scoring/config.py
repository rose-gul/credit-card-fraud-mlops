"""Central configuration for the fraud-scoring platform.

Loads static hyperparameters from ``params.yaml`` and runtime/secret settings from
the environment (``.env``). Every module imports :func:`get_config` / :data:`PARAMS`
instead of hard-coding paths or magic numbers, so the whole pipeline stays
reproducible and DVC-friendly.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Repository layout -------------------------------------------------------
# config.py lives at src/fraud_scoring/config.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "metrics"
MLRUNS_DIR = REPO_ROOT / "mlruns"
KAGGLE_DIR = REPO_ROOT / "kaggle"


def _load_params() -> dict[str, Any]:
    with PARAMS_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Static hyperparameters (single source of truth, mirrors params.yaml).
PARAMS: dict[str, Any] = _load_params()


class Settings(BaseSettings):
    """Runtime settings sourced from environment / ``.env`` (never committed)."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_prefix="FRAUD_",
        extra="ignore",
        protected_namespaces=(),
    )

    # MLflow tracking URI. A local SQLite backend is used by default: recent
    # MLflow requires a database backend (not the bare file store) for the model
    # registry. Artifacts still land under ./mlruns. Override via FRAUD_MLFLOW_TRACKING_URI.
    mlflow_tracking_uri: str = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
    # Path to a registered/exported model bundle used by the serving layer.
    model_path: str = str(MODELS_DIR / "model_xgb.json")
    # Kaggle username (defaults resolved from ~/.kaggle/kaggle.json at runtime).
    kaggle_username: str | None = None
    # Serving decision thresholds may be overridden without touching params.yaml.
    approve_below: float | None = None
    block_above: float | None = None


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Merged view: static params + resolved runtime settings + key paths."""
    settings = get_settings()
    thresholds = dict(PARAMS["thresholds"])
    if settings.approve_below is not None:
        thresholds["approve_below"] = settings.approve_below
    if settings.block_above is not None:
        thresholds["block_above"] = settings.block_above

    return {
        "params": PARAMS,
        "settings": settings,
        "thresholds": thresholds,
        "paths": {
            "repo_root": REPO_ROOT,
            "data": DATA_DIR,
            "raw": RAW_DIR,
            "processed": PROCESSED_DIR,
            "models": MODELS_DIR,
            "metrics": METRICS_DIR,
            "mlruns": MLRUNS_DIR,
            "kaggle": KAGGLE_DIR,
        },
    }


def ensure_dirs() -> None:
    """Create the standard output directories if missing (idempotent)."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, METRICS_DIR, MLRUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":  # quick sanity check
    import json

    cfg = get_config()
    print(json.dumps({k: str(v) for k, v in cfg["paths"].items()}, indent=2))
    print("thresholds:", cfg["thresholds"])
