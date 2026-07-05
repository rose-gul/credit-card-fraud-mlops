"""Fraud orchestrator: wires scoring -> explanation -> review into one decision.

Chain::

    orchestrator -> scoring_agent -> explanation_agent -> review_agent -> decision

Returns a stable decision object::

    {"transaction_id", "score", "band", "reason_codes", "flags", "model_version"}

If the model is absent the orchestrator runs in DEGRADED mode: ``score=None``,
``band="REVIEW"`` and a ``MODEL_UNAVAILABLE`` flag — the API still responds and rule
flags (high-amount, sanctions) still apply.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from pathlib import Path

from fraud_scoring.serving.agents.explanation_agent import ExplanationAgent
from fraud_scoring.serving.agents.review_agent import ReviewAgent
from fraud_scoring.serving.agents.scoring_agent import ScoringAgent


class FraudOrchestrator:
    """Coordinate the three deterministic agents behind a single ``assess`` call."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
        watchlist: Iterable[str] | None = None,
        top_n_reasons: int = 5,
    ) -> None:
        self.scoring_agent = ScoringAgent(model_path=model_path, metrics_path=metrics_path)
        self.explanation_agent = ExplanationAgent(scoring_agent=self.scoring_agent)
        self.review_agent = ReviewAgent(watchlist=watchlist)
        self.top_n_reasons = top_n_reasons

        self.model_version = self.scoring_agent.model_version

    @property
    def is_ready(self) -> bool:
        return self.scoring_agent.is_ready

    # ------------------------------------------------------------------ #
    @staticmethod
    def _txn_id(txn: dict) -> str:
        tid = txn.get("transaction_id")
        return str(tid) if tid else uuid.uuid4().hex

    def assess(self, txn: dict, watchlist: Iterable[str] | None = None) -> dict:
        """Run the full chain and return the decision object."""
        transaction_id = self._txn_id(txn)

        # ---- DEGRADED: no model available ------------------------------
        if not self.scoring_agent.is_ready:
            review = self.review_agent.review(None, txn, [], watchlist=watchlist)
            flags = ["MODEL_UNAVAILABLE", *review["flags"]]
            return {
                "transaction_id": transaction_id,
                "score": None,
                "band": review["band"],
                "reason_codes": [],
                "flags": flags,
                "model_version": self.model_version,
            }

        # ---- NORMAL path ----------------------------------------------
        score = self.scoring_agent.score(txn)
        reason_codes = self.explanation_agent.explain(txn, top_n=self.top_n_reasons)
        review = self.review_agent.review(score, txn, reason_codes, watchlist=watchlist)

        return {
            "transaction_id": transaction_id,
            "score": score,
            "band": review["band"],
            "reason_codes": reason_codes,
            "flags": review["flags"],
            "model_version": self.model_version,
        }

    def explain(self, txn: dict) -> list[dict]:
        """Convenience: reason codes only (empty when the model is unavailable)."""
        if not self.scoring_agent.is_ready:
            return []
        return self.explanation_agent.explain(txn, top_n=self.top_n_reasons)
