"""Tests for the FastAPI serving layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraud_scoring.serving.api import app


@pytest.fixture()
def client():
    # ``with`` triggers the lifespan hook (builds the orchestrator).
    with TestClient(app) as c:
        yield c


def _payload(sample_txn: dict) -> dict:
    return dict(sample_txn)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert isinstance(body["model_version"], str)


def test_score_returns_valid_decision(client, sample_txn):
    resp = client.post("/score", json=_payload(sample_txn))
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == "txn-0001"
    assert body["band"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert 0.0 <= body["score"] <= 1.0
    assert isinstance(body["reason_codes"], list)
    assert isinstance(body["flags"], list)


def test_explain_returns_reason_codes(client, sample_txn):
    resp = client.post("/explain", json=_payload(sample_txn))
    assert resp.status_code == 200
    codes = resp.json()["reason_codes"]
    assert isinstance(codes, list) and codes
    assert {"feature", "contribution", "direction"} <= set(codes[0])


def test_metrics_endpoint_is_prometheus_text(client, sample_txn):
    # Score once so counters are non-zero.
    client.post("/score", json=_payload(sample_txn))
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "fraud_scored_total" in text
    assert "fraud_score_latency_seconds" in text


def test_score_rejects_bad_payload(client):
    # Missing required Amount / V-fields -> 422 validation error.
    resp = client.post("/score", json={"Time": 1.0})
    assert resp.status_code == 422
