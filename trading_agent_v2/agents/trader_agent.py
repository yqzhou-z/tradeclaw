from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from trading_agent_v2.schemas import (
    AnalystView,
    FinalDecision,
    RiskReport,
    TradeProposal,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TraderAgent:
    def __init__(
        self,
        buy_threshold: float = 0.20,
        sell_threshold: float = -0.20,
        min_confidence: float = 0.55,
        default_buy_size_pct: float = 0.10,
        default_sell_size_pct: float = 0.50,
        default_stop_loss_pct: float = 0.03,
        default_take_profit_pct: float = 0.06,
        analyst_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Parameters
        ----------
        buy_threshold : float
            Net score above this value -> buy proposal.
        sell_threshold : float
            Net score below this value -> sell proposal.
        min_confidence : float
            Minimum normalized confidence to allow buy/sell proposal.
        default_buy_size_pct : float
            Used when proposal action is buy.
        default_sell_size_pct : float
            Used when proposal action is sell.
        default_stop_loss_pct : float
            Default stop loss for directional trades.
        default_take_profit_pct : float
            Default take profit for directional trades.
        analyst_weights : dict[str, float] | None
            Example:
            {
                "market_analyst": 1.2,
                "news_analyst": 0.8,
            }
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence
        self.default_buy_size_pct = default_buy_size_pct
        self.default_sell_size_pct = default_sell_size_pct
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.analyst_weights = analyst_weights or {}

    # =========================================================
    # Public API
    # =========================================================

    def generate_proposal(
        self,
        symbol: str,
        analyst_views: List[AnalystView],
        portfolio: dict,
        recent_memory: Optional[List[dict]] = None,
        strategy_memory: Optional[dict] = None,
    ) -> TradeProposal:
        """
        Aggregate multiple analyst views into a preliminary trade proposal.
        """
        timestamp = utc_now_iso()
        recent_memory = recent_memory or []
        strategy_memory = strategy_memory or {}

        if not analyst_views:
            return TradeProposal(
                symbol=symbol,
                timestamp=timestamp,
                action="hold",
                confidence=0.0,
                thesis="No analyst views available.",
                supporting_factors=[],
                conflicting_factors=["missing_analyst_views"],
                suggested_size_pct=0.0,
                stop_loss_pct=None,
                take_profit_pct=None,
                metadata={},
            )

        fusion = self._fuse_analyst_views(analyst_views)

        action = self._decide_action_from_score(
            net_score=fusion["net_score"],
            confidence=fusion["normalized_confidence"],
        )

        suggested_size_pct = 0.0
        stop_loss_pct = None
        take_profit_pct = None

        if action == "buy":
            suggested_size_pct = self.default_buy_size_pct
            stop_loss_pct = self.default_stop_loss_pct
            take_profit_pct = self.default_take_profit_pct
        elif action == "sell":
            suggested_size_pct = self.default_sell_size_pct
            stop_loss_pct = self.default_stop_loss_pct
            take_profit_pct = self.default_take_profit_pct

        supporting_factors = self._collect_supporting_factors(analyst_views, action)
        conflicting_factors = self._collect_conflicting_factors(analyst_views, action)

        thesis = self._build_thesis(
            analyst_views=analyst_views,
            action=action,
            fusion=fusion,
        )

        return TradeProposal(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            confidence=fusion["normalized_confidence"],
            thesis=thesis,
            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
            suggested_size_pct=suggested_size_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            metadata={
                "net_score": fusion["net_score"],
                "bullish_score": fusion["bullish_score"],
                "bearish_score": fusion["bearish_score"],
                "neutral_score": fusion["neutral_score"],
                "analyst_breakdown": fusion["analyst_breakdown"],
                "recent_memory_count": len(recent_memory),
                "strategy_memory_keys": list(strategy_memory.keys()),
                "current_positions": list((portfolio or {}).get("positions", {}).keys())
                if isinstance(portfolio, dict)
                else [],
            },
        )

    def make_final_decision(
        self,
        proposal: TradeProposal,
        risk_report: RiskReport,
    ) -> FinalDecision:
        """
        Convert a proposal + risk report into a final executable decision.
        """
        timestamp = utc_now_iso()

        if not risk_report.approved:
            reason = risk_report.rejection_reason or "Trade blocked by risk manager."
            if risk_report.warnings:
                reason += f" Warnings: {', '.join(risk_report.warnings)}"

            return FinalDecision(
                symbol=proposal.symbol,
                timestamp=timestamp,
                action="hold",
                reason=reason,
                size_pct=0.0,
                order_type="market",
                stop_loss_pct=None,
                take_profit_pct=None,
                metadata={
                    "proposal_action": proposal.action,
                    "proposal_confidence": proposal.confidence,
                    "risk_score": risk_report.risk_score,
                    "risk_summary": risk_report.summary,
                },
            )

        if proposal.action == "hold":
            return FinalDecision(
                symbol=proposal.symbol,
                timestamp=timestamp,
                action="hold",
                reason=f"Proposal indicates hold. {proposal.thesis}",
                size_pct=0.0,
                order_type="market",
                stop_loss_pct=None,
                take_profit_pct=None,
                metadata={
                    "proposal_confidence": proposal.confidence,
                    "risk_score": risk_report.risk_score,
                    "risk_summary": risk_report.summary,
                },
            )

        final_size_pct = (
            risk_report.adjusted_size_pct
            if risk_report.adjusted_size_pct is not None
            else proposal.suggested_size_pct
        )
        final_stop_loss_pct = (
            risk_report.adjusted_stop_loss_pct
            if risk_report.adjusted_stop_loss_pct is not None
            else proposal.stop_loss_pct
        )
        final_take_profit_pct = (
            risk_report.adjusted_take_profit_pct
            if risk_report.adjusted_take_profit_pct is not None
            else proposal.take_profit_pct
        )

        final_reason = f"{proposal.thesis} Approved by risk manager."
        if risk_report.warnings:
            final_reason += f" Warnings: {', '.join(risk_report.warnings)}"

        return FinalDecision(
            symbol=proposal.symbol,
            timestamp=timestamp,
            action=proposal.action,
            reason=final_reason,
            size_pct=final_size_pct,
            order_type="market",
            stop_loss_pct=final_stop_loss_pct,
            take_profit_pct=final_take_profit_pct,
            metadata={
                "proposal_confidence": proposal.confidence,
                "proposal_action": proposal.action,
                "risk_score": risk_report.risk_score,
                "risk_summary": risk_report.summary,
                "supporting_factors": proposal.supporting_factors,
                "conflicting_factors": proposal.conflicting_factors,
            },
        )

    # =========================================================
    # Fusion logic
    # =========================================================

    def _fuse_analyst_views(self, analyst_views: List[AnalystView]) -> Dict[str, object]:
        bullish_score = 0.0
        bearish_score = 0.0
        neutral_score = 0.0
        total_weighted_confidence = 0.0
        total_weight = 0.0
        analyst_breakdown = []

        for view in analyst_views:
            weight = float(self.analyst_weights.get(view.analyst_name, 1.0))
            confidence = max(0.0, min(1.0, float(view.confidence)))
            weighted_confidence = weight * confidence
            bias = (view.bias or "").lower().strip()

            total_weight += weight
            total_weighted_confidence += weighted_confidence

            if bias == "bullish":
                bullish_score += weighted_confidence
            elif bias == "bearish":
                bearish_score += weighted_confidence
            else:
                neutral_score += weighted_confidence

            analyst_breakdown.append(
                {
                    "analyst_name": view.analyst_name,
                    "bias": bias,
                    "confidence": confidence,
                    "weight": weight,
                    "weighted_confidence": weighted_confidence,
                    "summary": view.summary,
                }
            )

        normalized_confidence = (
            total_weighted_confidence / total_weight if total_weight > 0 else 0.0
        )
        net_score = bullish_score - bearish_score

        return {
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "neutral_score": neutral_score,
            "net_score": net_score,
            "normalized_confidence": normalized_confidence,
            "analyst_breakdown": analyst_breakdown,
        }

    def _decide_action_from_score(
        self,
        net_score: float,
        confidence: float,
    ) -> str:
        if confidence < self.min_confidence:
            return "hold"

        if net_score >= self.buy_threshold:
            return "buy"

        if net_score <= self.sell_threshold:
            return "sell"

        return "hold"

    # =========================================================
    # Explanation helpers
    # =========================================================

    def _collect_supporting_factors(
        self,
        analyst_views: List[AnalystView],
        action: str,
    ) -> List[str]:
        if action == "hold":
            return []

        desired_bias = "bullish" if action == "buy" else "bearish"
        factors: List[str] = []

        for view in analyst_views:
            if (view.bias or "").lower().strip() == desired_bias:
                factors.append(f"{view.analyst_name}_{desired_bias}")
                factors.extend(
                    [f"{view.analyst_name}:{signal}" for signal in view.supporting_signals]
                )

        return self._dedupe_preserve_order(factors)

    def _collect_conflicting_factors(
        self,
        analyst_views: List[AnalystView],
        action: str,
    ) -> List[str]:
        if action == "hold":
            biases = [((view.bias or "").lower().strip(), view.analyst_name) for view in analyst_views]
            unique_biases = {bias for bias, _ in biases if bias}
            if len(unique_biases) > 1:
                return [f"mixed_signals:{name}:{bias}" for bias, name in biases]
            return []

        conflicting_bias = "bearish" if action == "buy" else "bullish"
        factors: List[str] = []

        for view in analyst_views:
            bias = (view.bias or "").lower().strip()
            if bias == conflicting_bias or bias == "neutral":
                factors.append(f"{view.analyst_name}_{bias}")
                factors.extend(
                    [f"{view.analyst_name}:risk:{flag}" for flag in view.risk_flags]
                )

        return self._dedupe_preserve_order(factors)

    def _build_thesis(
        self,
        analyst_views: List[AnalystView],
        action: str,
        fusion: Dict[str, object],
    ) -> str:
        bullish_count = sum(
            1 for view in analyst_views if (view.bias or "").lower().strip() == "bullish"
        )
        bearish_count = sum(
            1 for view in analyst_views if (view.bias or "").lower().strip() == "bearish"
        )
        neutral_count = sum(
            1 for view in analyst_views if (view.bias or "").lower().strip() == "neutral"
        )

        net_score = float(fusion["net_score"])
        confidence = float(fusion["normalized_confidence"])

        if action == "buy":
            return (
                f"Analyst fusion is bullish "
                f"(bullish={bullish_count}, bearish={bearish_count}, neutral={neutral_count}, "
                f"net_score={net_score:.3f}, confidence={confidence:.2f})."
            )
        if action == "sell":
            return (
                f"Analyst fusion is bearish "
                f"(bullish={bullish_count}, bearish={bearish_count}, neutral={neutral_count}, "
                f"net_score={net_score:.3f}, confidence={confidence:.2f})."
            )
        return (
            f"No clear directional edge from analyst fusion "
            f"(bullish={bullish_count}, bearish={bearish_count}, neutral={neutral_count}, "
            f"net_score={net_score:.3f}, confidence={confidence:.2f})."
        )

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result