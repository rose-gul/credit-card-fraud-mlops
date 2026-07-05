"""FastAPI serving layer for the multi-agent fraud-scoring service.

Endpoints
---------
- ``POST /score``   -> full decision object
- ``POST /explain`` -> reason codes only
- ``GET  /health``  -> {status, model_ready, model_version}
- ``GET  /metrics`` -> Prometheus exposition text

A single :class:`FraudOrchestrator` is created at startup via the lifespan hook and
degrades gracefully when the trained model is missing.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import ConfigDict, create_model

from fraud_scoring.serving.agents.orchestrator import FraudOrchestrator

# --------------------------------------------------------------------------- #
# Request schema — generated programmatically to avoid typing 28 V-fields.
# --------------------------------------------------------------------------- #
_txn_fields: dict = {
    "transaction_id": (str | None, None),
    "Time": (float, ...),
    "Amount": (float, ...),
}
for _i in range(1, 29):
    _txn_fields[f"V{_i}"] = (float, ...)

# ``extra="allow"`` keeps caller-supplied ids (account_id, merchant_id, ...).
Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="allow"),
    **_txn_fields,
)

# --------------------------------------------------------------------------- #
# Prometheus metrics (own registry so re-import in tests is safe).
# --------------------------------------------------------------------------- #
REGISTRY = CollectorRegistry()
SCORED_TOTAL = Counter(
    "fraud_scored_total",
    "Total number of transactions scored.",
    registry=REGISTRY,
)
SCORED_BY_BAND = Counter(
    "fraud_scored_by_band_total",
    "Transactions scored, partitioned by decision band.",
    ["band"],
    registry=REGISTRY,
)
SCORE_LATENCY = Histogram(
    "fraud_score_latency_seconds",
    "Latency of the /score endpoint in seconds.",
    registry=REGISTRY,
)


# --------------------------------------------------------------------------- #
# App + lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the orchestrator once; lazy/graceful when the model is absent.
    app.state.orchestrator = FraudOrchestrator()
    yield


app = FastAPI(
    title="Fraud Scoring Service",
    version="0.1.0",
    description="Runtime multi-agent fraud scoring (scoring -> explanation -> review).",
    lifespan=lifespan,
)


def _orchestrator(request: Request) -> FraudOrchestrator:
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:  # defensive: build on demand if lifespan didn't run
        orch = FraudOrchestrator()
        request.app.state.orchestrator = orch
    return orch


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health(request: Request) -> dict:
    orch = _orchestrator(request)
    return {
        "status": "ok",
        "model_ready": bool(orch.is_ready),
        "model_version": orch.model_version,
    }


@app.post("/score")
def score(txn: Transaction, request: Request) -> dict:
    orch = _orchestrator(request)
    start = time.perf_counter()
    decision = orch.assess(txn.model_dump())
    SCORE_LATENCY.observe(time.perf_counter() - start)
    SCORED_TOTAL.inc()
    SCORED_BY_BAND.labels(band=decision["band"]).inc()
    return decision


@app.post("/explain")
def explain(txn: Transaction, request: Request) -> dict:
    orch = _orchestrator(request)
    return {"reason_codes": orch.explain(txn.model_dump())}


@app.get("/metrics")
def metrics() -> Response:
    data = generate_latest(REGISTRY)
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
