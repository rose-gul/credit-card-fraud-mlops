"""Tests for the runtime multi-agent chain."""

from __future__ import annotations

from fraud_scoring.serving.agents.explanation_agent import ExplanationAgent
from fraud_scoring.serving.agents.orchestrator import FraudOrchestrator
from fraud_scoring.serving.agents.review_agent import (
    APPROVE,
    BLOCK,
    REVIEW,
    ReviewAgent,
)
from fraud_scoring.serving.agents.scoring_agent import ScoringAgent


# --------------------------------------------------------------------------- #
# Scoring agent
# --------------------------------------------------------------------------- #
def test_scoring_agent_ready_and_range(sample_txn):
    agent = ScoringAgent()
    assert agent.is_ready is True
    score = agent.score(sample_txn)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_scoring_agent_missing_model_raises(tmp_path):
    agent = ScoringAgent(
        model_path=tmp_path / "nope.json",
        metrics_path=tmp_path / "nope_metrics.json",
    )
    assert agent.is_ready is False
    import pytest

    with pytest.raises(RuntimeError):
        agent.score({"Time": 0.0, "Amount": 1.0})


# --------------------------------------------------------------------------- #
# Review agent (pure decision logic)
# --------------------------------------------------------------------------- #
def _thresholds():
    return {"approve_below": 0.30, "block_above": 0.85, "high_amount_review": 5000.0}


def test_review_banding():
    agent = ReviewAgent(thresholds=_thresholds())
    assert agent.review(0.10, {"Amount": 10.0})["band"] == APPROVE
    assert agent.review(0.50, {"Amount": 10.0})["band"] == REVIEW
    assert agent.review(0.90, {"Amount": 10.0})["band"] == BLOCK


def test_review_high_amount_forces_review():
    agent = ReviewAgent(thresholds=_thresholds())
    out = agent.review(0.05, {"Amount": 9000.0})
    assert out["band"] == REVIEW  # escalated from APPROVE
    assert "HIGH_AMOUNT" in out["flags"]


def test_review_high_amount_does_not_downgrade_block():
    agent = ReviewAgent(thresholds=_thresholds())
    out = agent.review(0.95, {"Amount": 9000.0})
    assert out["band"] == BLOCK
    assert "HIGH_AMOUNT" in out["flags"]


def test_review_sanctions_watchlist_escalates_to_block():
    agent = ReviewAgent(thresholds=_thresholds(), watchlist={"acct-bad"})
    out = agent.review(0.01, {"Amount": 5.0, "account_id": "acct-bad"})
    assert out["band"] == BLOCK
    assert "SANCTIONS_WATCHLIST" in out["flags"]


def test_review_watchlist_per_call_extension():
    agent = ReviewAgent(thresholds=_thresholds())
    out = agent.review(
        0.01,
        {"Amount": 5.0, "merchant_id": "m-42"},
        watchlist={"m-42"},
    )
    assert out["band"] == BLOCK
    assert "SANCTIONS_WATCHLIST" in out["flags"]


def test_review_degraded_score_none():
    agent = ReviewAgent(thresholds=_thresholds())
    out = agent.review(None, {"Amount": 10.0})
    assert out["band"] == REVIEW


# --------------------------------------------------------------------------- #
# Explanation agent
# --------------------------------------------------------------------------- #
def test_explanation_returns_reason_codes(sample_txn):
    scorer = ScoringAgent()
    agent = ExplanationAgent(scoring_agent=scorer)
    codes = agent.explain(sample_txn, top_n=5)
    assert isinstance(codes, list)
    assert 1 <= len(codes) <= 5
    first = codes[0]
    assert set(first) == {"feature", "contribution", "direction"}
    assert first["direction"] in {"increases_risk", "decreases_risk"}
    # Sorted by |contribution| descending.
    mags = [abs(c["contribution"]) for c in codes]
    assert mags == sorted(mags, reverse=True)


def test_explanation_empty_when_no_model():
    agent = ExplanationAgent(booster=None, feature_columns=["V1", "Amount"])
    assert agent.explain({"Time": 0.0, "Amount": 1.0}) == []


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def test_orchestrator_full_decision(sample_txn):
    orch = FraudOrchestrator()
    assert orch.is_ready is True
    decision = orch.assess(sample_txn)
    assert decision["transaction_id"] == "txn-0001"
    assert decision["band"] in {APPROVE, REVIEW, BLOCK}
    assert 0.0 <= decision["score"] <= 1.0
    assert isinstance(decision["reason_codes"], list) and decision["reason_codes"]
    assert isinstance(decision["flags"], list)
    assert isinstance(decision["model_version"], str)


def test_orchestrator_sanctions_escalation(sample_txn):
    orch = FraudOrchestrator(watchlist={"acct-123"})
    decision = orch.assess(sample_txn)
    assert decision["band"] == BLOCK
    assert "SANCTIONS_WATCHLIST" in decision["flags"]


def test_orchestrator_degraded_mode(tmp_path, sample_txn):
    orch = FraudOrchestrator(
        model_path=tmp_path / "missing.json",
        metrics_path=tmp_path / "missing_metrics.json",
    )
    assert orch.is_ready is False
    decision = orch.assess(sample_txn)
    assert decision["score"] is None
    assert decision["band"] == REVIEW
    assert "MODEL_UNAVAILABLE" in decision["flags"]
    assert decision["reason_codes"] == []
