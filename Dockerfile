# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# fraud-scoring serving image.
# Builds a lean-ish container that runs ONLY the FastAPI scoring service.
# Training (Kaggle GPU) and data download are NOT part of this image.
#
# Note: torch + xgboost are large wheels, so the image is a few GB. That is
# acceptable for a model-serving container. To slim it further you could:
#   * install a CPU-only torch wheel (--extra-index-url download.pytorch.org/whl/cpu)
#   * use a serving-only requirements file (drop dvc, kaggle, mlflow client, shap)
#   * adopt a multi-stage build that copies only site-packages into a fresh base
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Fail fast, no .pyc, unbuffered logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Minimal OS deps some ML wheels expect at runtime (libgomp for xgboost/OpenMP).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy only what serving needs: the package source and the static params.
COPY src/ ./src/
COPY params.yaml ./params.yaml

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Basic container healthcheck against the API's health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "fraud_scoring.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
