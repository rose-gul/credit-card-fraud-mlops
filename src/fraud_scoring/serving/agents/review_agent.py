"""Review agent: pure, deterministic decision logic.

Maps a fraud ``score`` (plus rule-based checks on the raw transaction) to a decision
band and a set of flags. No model or I/O here — just policy — so it is trivially
unit-testable and safe to run even in DEGRADED mode (``score is None``).
"""

from __future__ import annotations

from collections.abc import Iterable

from fraud_scoring.config import get_config

APPROVE = "APPROVE"
REVIEW = "REVIEW"
BLOCK = "BLOCK"

# Transaction id fields checked against the sanctions/watchlist set.
WATCHLIST_ID_FIELDS = ("account_id", "merchant_id", "customer_id", "card_id")


class ReviewAgent:
    """Deterministic banding + rule flags."""

    def __init__(
        self,
        thresholds: dict | None = None,
        watchlist: Iterable[str] | None = None,
    ) -> None:
        cfg = thresholds if thresholds is not None else get_config()["thresholds"]
        self.approve_below = float(cfg["approve_below"])
        self.block_above = float(cfg["block_above"])
        self.high_amount_review = float(cfg["high_amount_review"])
        self.watchlist: set[str] = {str(x) for x in (watchlist or set())}

    # ------------------------------------------------------------------ #
    def _band_from_score(self, score: float | None) -> str:
        if score is None:  # degraded mode: cannot approve without a model
            return REVIEW
        if score < self.approve_below:
            return APPROVE
        if score >= self.block_above:
            return BLOCK
        return REVIEW

    def _watchlist_hit(self, txn: dict, extra: set[str]) -> bool:
        active = self.watchlist | extra
        if not active:
            return False
        for field in WATCHLIST_ID_FIELDS:
            value = txn.get(field)
            if value is not None and str(value) in active:
                return True
        return False

    # ------------------------------------------------------------------ #
    def review(
        self,
        score: float | None,
        txn: dict,
        reason_codes: list | None = None,  # noqa: ARG002 - accepted for a stable API
        watchlist: Iterable[str] | None = None,
    ) -> dict:
        """Return ``{"band", "flags"}`` for a scored transaction."""
        flags: list[str] = []
        band = self._band_from_score(score)

        # Rule: large amounts force at least REVIEW.
        amount = float(txn.get("Amount", 0.0) or 0.0)
        if amount >= self.high_amount_review:
            flags.append("HIGH_AMOUNT")
            if band == APPROVE:
                band = REVIEW

        # Hook: sanctions / watchlist screening escalates straight to BLOCK.
        extra = {str(x) for x in (watchlist or set())}
        if self._watchlist_hit(txn, extra):
            flags.append("SANCTIONS_WATCHLIST")
            band = BLOCK

        return {"band": band, "flags": flags}
