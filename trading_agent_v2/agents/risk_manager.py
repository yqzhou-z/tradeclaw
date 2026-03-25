from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_agent_v2.schemas import PortfolioState, RiskReport, TradeProposal


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class RiskManager:
    def __init__(
        self,
        max_total_invested_pct: float = 0.90,
        max_single_position_pct: float = 0.45,
        min_proposal_confidence: float = 0.50,
        max_conflicting_factors: int = 4,
        max_loss_streak: int = 4,
        volatility_size_cut_ratio: float = 0.70,
        high_risk_score_threshold: float = 0.85,
        medium_risk_score_threshold: float = 0.60,
        elevated_risk_size_multiplier: float = 0.85,
        conflict_size_cut_ratio: float = 0.90,
        volatility_stop_loss_tighten_ratio: float = 0.92,
    ):
        """
        Parameters
        ----------
        max_total_invested_pct : float
            Max fraction of total equity allowed to be invested.
        max_single_position_pct : float
            Max fraction of total equity allowed in one symbol.
        min_proposal_confidence : float
            Minimum proposal confidence required for directional trades.
        max_conflicting_factors : int
            Too many conflicts -> reject.
        max_loss_streak : int
            If recent consecutive losses exceed this, reject or heavily restrict.
        volatility_size_cut_ratio : float
            If volatility risk is detected, multiply size by this ratio.
        high_risk_score_threshold : float
            Above this threshold, reject.
        medium_risk_score_threshold : float
            Above this threshold, allow but adjust size/SL more conservatively.
        elevated_risk_size_multiplier : float
            Size multiplier applied when risk_score >= medium_risk_score_threshold.
        conflict_size_cut_ratio : float
            Additional size multiplier when conflicting signals are present.
        volatility_stop_loss_tighten_ratio : float
            Stop-loss tightening multiplier under volatility risk.
        """
        self.max_total_invested_pct = max_total_invested_pct
        self.max_single_position_pct = max_single_position_pct
        self.min_proposal_confidence = min_proposal_confidence
        self.max_conflicting_factors = max_conflicting_factors
        self.max_loss_streak = max_loss_streak
        self.volatility_size_cut_ratio = volatility_size_cut_ratio
        self.high_risk_score_threshold = high_risk_score_threshold
        self.medium_risk_score_threshold = medium_risk_score_threshold
        self.elevated_risk_size_multiplier = elevated_risk_size_multiplier
        self.conflict_size_cut_ratio = conflict_size_cut_ratio
        self.volatility_stop_loss_tighten_ratio = volatility_stop_loss_tighten_ratio

    # =========================================================
    # Proposal compatibility helpers
    # =========================================================

    def _proposal_get(self, proposal: Any, key: str, default=None):
        if isinstance(proposal, dict):
            return proposal.get(key, default)
        return getattr(proposal, key, default)

    def _proposal_size_pct(self, proposal: Any) -> float:
        value = self._proposal_get(
            proposal,
            "suggested_size_pct",
            self._proposal_get(proposal, "size_pct", 0.0),
        )
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _proposal_action(self, proposal: Any) -> str:
        return str(self._proposal_get(proposal, "action", "") or "").lower().strip()

    def _proposal_symbol(self, proposal: Any) -> str:
        return str(self._proposal_get(proposal, "symbol", "") or "")

    def _proposal_confidence(self, proposal: Any) -> float:
        try:
            return float(self._proposal_get(proposal, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # =========================================================
    # Public API
    # =========================================================

    def evaluate(
        self,
        proposal: TradeProposal | dict,
        portfolio: PortfolioState,
        recent_memory: Optional[List[dict]] = None,
        strategy_memory: Optional[dict] = None,
    ) -> RiskReport:
        recent_memory = recent_memory or []
        strategy_memory = strategy_memory or {}

        timestamp = utc_now_iso()
        action = self._proposal_action(proposal)
        symbol = self._proposal_symbol(proposal)
        confidence = self._proposal_confidence(proposal)

        if action == "hold":
            return RiskReport(
                symbol=symbol,
                timestamp=timestamp,
                approved=True,
                risk_score=0.0,
                summary="Hold proposal carries no execution risk.",
                warnings=[],
                rejection_reason=None,
                adjusted_size_pct=0.0,
                adjusted_stop_loss_pct=None,
                adjusted_take_profit_pct=None,
                metadata={
                    "proposal_action": self._proposal_get(proposal, "action", ""),
                    "proposal_confidence": confidence,
                },
            )

        warnings: List[str] = []
        rejection_reasons: List[str] = []

        risk_score = 0.0

        # 1. proposal-level checks
        proposal_risk, proposal_warnings, proposal_rejections = self._evaluate_proposal_quality(proposal)
        risk_score += proposal_risk
        warnings.extend(proposal_warnings)
        rejection_reasons.extend(proposal_rejections)

        # 2. portfolio exposure checks
        exposure_risk, exposure_warnings, exposure_rejections, exposure_info = self._evaluate_portfolio_exposure(
            proposal=proposal,
            portfolio=portfolio,
        )
        risk_score += exposure_risk
        warnings.extend(exposure_warnings)
        rejection_reasons.extend(exposure_rejections)

        # 3. behavioral / recent memory checks
        behavior_risk, behavior_warnings, behavior_rejections, behavior_info = self._evaluate_recent_behavior(
            recent_memory=recent_memory,
        )
        risk_score += behavior_risk
        warnings.extend(behavior_warnings)
        rejection_reasons.extend(behavior_rejections)

        # 4. strategy memory checks (light v1)
        strategy_risk, strategy_warnings = self._evaluate_strategy_memory(strategy_memory)
        risk_score += strategy_risk
        warnings.extend(strategy_warnings)

        # clamp
        risk_score = max(0.0, min(1.0, risk_score))

        # 5. adjustment logic
        adjusted_size_pct = self._proposal_size_pct(proposal)
        adjusted_stop_loss_pct = self._proposal_get(proposal, "stop_loss_pct", None)
        adjusted_take_profit_pct = self._proposal_get(proposal, "take_profit_pct", None)

        adjusted_size_pct, adjusted_stop_loss_pct, size_warnings = self._apply_risk_adjustments(
            proposal=proposal,
            current_size_pct=adjusted_size_pct,
            current_stop_loss_pct=adjusted_stop_loss_pct,
            warnings=warnings,
            risk_score=risk_score,
        )
        warnings.extend(size_warnings)

        # 6. approval decision
        approved = True
        rejection_reason = None

        if rejection_reasons:
            approved = False
            rejection_reason = " | ".join(self._dedupe_preserve_order(rejection_reasons))
        elif risk_score >= self.high_risk_score_threshold:
            approved = False
            rejection_reason = (
                f"Risk score too high ({risk_score:.2f}) for execution."
            )

        summary = self._build_summary(
            approved=approved,
            proposal=proposal,
            risk_score=risk_score,
            warnings=warnings,
            rejection_reason=rejection_reason,
        )

        return RiskReport(
            symbol=symbol,
            timestamp=timestamp,
            approved=approved,
            risk_score=risk_score,
            summary=summary,
            warnings=self._dedupe_preserve_order(warnings),
            rejection_reason=rejection_reason,
            adjusted_size_pct=adjusted_size_pct if approved else None,
            adjusted_stop_loss_pct=adjusted_stop_loss_pct if approved else None,
            adjusted_take_profit_pct=adjusted_take_profit_pct if approved else None,
            metadata={
                "proposal_action": self._proposal_get(proposal, "action", ""),
                "proposal_confidence": confidence,
                "exposure_info": exposure_info,
                "behavior_info": behavior_info,
                "strategy_memory_keys": list(strategy_memory.keys()),
            },
        )

    # =========================================================
    # Proposal checks
    # =========================================================

    def _evaluate_proposal_quality(
        self,
        proposal: TradeProposal | dict,
    ) -> tuple[float, List[str], List[str]]:
        risk = 0.0
        warnings: List[str] = []
        rejections: List[str] = []

        action = self._proposal_action(proposal)
        confidence = self._proposal_confidence(proposal)
        size_pct = self._proposal_size_pct(proposal)
        num_conflicts = len(self._proposal_get(proposal, "conflicting_factors", []) or [])

        if action not in {"buy", "sell", "hold"}:
            rejections.append(
                f"Unsupported proposal action: {self._proposal_get(proposal, 'action', '')}"
            )
            risk += 0.40

        if action in {"buy", "sell"} and confidence < self.min_proposal_confidence:
            rejections.append(
                f"Proposal confidence too low ({confidence:.2f} < {self.min_proposal_confidence:.2f})."
            )
            risk += 0.30

        if action in {"buy", "sell"} and size_pct <= 0:
            rejections.append("Directional proposal has non-positive suggested_size_pct.")
            risk += 0.30

        if num_conflicts > 0:
            risk += min(0.20, 0.04 * num_conflicts)
            warnings.append(f"Proposal has {num_conflicts} conflicting factor(s).")

        if num_conflicts > self.max_conflicting_factors:
            rejections.append(
                f"Too many conflicting factors ({num_conflicts} > {self.max_conflicting_factors})."
            )
            risk += 0.20

        return risk, warnings, rejections

    # =========================================================
    # Portfolio checks
    # =========================================================

    def _evaluate_portfolio_exposure(
        self,
        proposal: TradeProposal | dict,
        portfolio: PortfolioState,
    ) -> tuple[float, List[str], List[str], Dict[str, Any]]:
        risk = 0.0
        warnings: List[str] = []
        rejections: List[str] = []

        symbol = self._proposal_symbol(proposal)
        action = self._proposal_action(proposal)
        proposal_size_pct = self._proposal_size_pct(proposal)

        total_equity = float(portfolio.total_equity or 0.0)
        cash = float(portfolio.cash or 0.0)
        positions = portfolio.positions or {}

        total_market_value = sum(
            float(pos.get("market_value", 0.0)) for pos in positions.values()
        )
        current_total_invested_pct = (
            total_market_value / total_equity if total_equity > 0 else 0.0
        )

        symbol_market_value = float(positions.get(symbol, {}).get("market_value", 0.0))
        current_symbol_exposure_pct = (
            symbol_market_value / total_equity if total_equity > 0 else 0.0
        )

        projected_total_invested_pct = current_total_invested_pct
        projected_symbol_exposure_pct = current_symbol_exposure_pct

        if action == "buy":
            add_value = cash * proposal_size_pct
            projected_total_invested_pct = (
                (total_market_value + add_value) / total_equity if total_equity > 0 else 1.0
            )
            projected_symbol_exposure_pct = (
                (symbol_market_value + add_value) / total_equity if total_equity > 0 else 1.0
            )
        elif action == "sell":
            reduce_value = symbol_market_value * proposal_size_pct
            projected_total_invested_pct = (
                max(0.0, total_market_value - reduce_value) / total_equity if total_equity > 0 else 0.0
            )
            projected_symbol_exposure_pct = (
                max(0.0, symbol_market_value - reduce_value) / total_equity if total_equity > 0 else 0.0
            )

        if action == "buy":
            if projected_total_invested_pct > self.max_total_invested_pct:
                rejections.append(
                    f"Projected total invested exposure too high "
                    f"({projected_total_invested_pct:.2%} > {self.max_total_invested_pct:.2%})."
                )
                risk += 0.35
            elif projected_total_invested_pct > self.max_total_invested_pct * 0.9:
                warnings.append(
                    f"Projected total invested exposure is near limit ({projected_total_invested_pct:.2%})."
                )
                risk += 0.12

            if projected_symbol_exposure_pct > self.max_single_position_pct:
                rejections.append(
                    f"Projected {symbol} exposure too high "
                    f"({projected_symbol_exposure_pct:.2%} > {self.max_single_position_pct:.2%})."
                )
                risk += 0.35
            elif projected_symbol_exposure_pct > self.max_single_position_pct * 0.85:
                warnings.append(
                    f"Projected {symbol} exposure is near limit ({projected_symbol_exposure_pct:.2%})."
                )
                risk += 0.10

        if action == "sell":
            current_qty = float(positions.get(symbol, {}).get("quantity", 0.0))
            if current_qty <= 0:
                rejections.append(f"No existing position found for sell action on {symbol}.")
                risk += 0.35

        info = {
            "cash": cash,
            "total_equity": total_equity,
            "current_total_invested_pct": current_total_invested_pct,
            "projected_total_invested_pct": projected_total_invested_pct,
            "current_symbol_exposure_pct": current_symbol_exposure_pct,
            "projected_symbol_exposure_pct": projected_symbol_exposure_pct,
        }

        return risk, warnings, rejections, info

    # =========================================================
    # Recent memory / behavior checks
    # =========================================================

    def _evaluate_recent_behavior(
        self,
        recent_memory: List[dict],
    ) -> tuple[float, List[str], List[str], Dict[str, Any]]:
        risk = 0.0
        warnings: List[str] = []
        rejections: List[str] = []

        loss_streak = self._compute_recent_loss_streak(recent_memory)
        recent_trade_count = self._count_recent_executed_trades(recent_memory)

        if loss_streak > 0:
            warnings.append(f"Recent loss streak detected: {loss_streak}.")
            risk += min(0.25, 0.06 * loss_streak)

        if loss_streak >= self.max_loss_streak:
            rejections.append(
                f"Recent loss streak too large ({loss_streak} >= {self.max_loss_streak})."
            )
            risk += 0.25

        if recent_trade_count >= 5:
            warnings.append(f"High recent trade frequency detected: {recent_trade_count}.")
            risk += 0.10

        if recent_trade_count >= 8:
            warnings.append("Possible overtrading behavior detected.")
            risk += 0.10

        info = {
            "loss_streak": loss_streak,
            "recent_trade_count": recent_trade_count,
            "recent_memory_count": len(recent_memory),
        }

        return risk, warnings, rejections, info

    def _compute_recent_loss_streak(self, recent_memory: List[dict]) -> int:
        """
        Simple v1 heuristic:
        Walk backward through recent episodes.
        Count consecutive losing executed trades until first non-loss.
        """
        streak = 0

        for episode in reversed(recent_memory):
            execution = episode.get("execution_result") or {}
            status = str(execution.get("status", "")).lower()
            if status != "filled":
                continue

            portfolio_snapshot = episode.get("portfolio_snapshot") or {}
            realized_pnl = portfolio_snapshot.get("realized_pnl")

            exec_meta = execution.get("metadata") or {}
            pnl_hint = exec_meta.get("realized_pnl")

            pnl_value = None
            if pnl_hint is not None:
                try:
                    pnl_value = float(pnl_hint)
                except (TypeError, ValueError):
                    pnl_value = None

            if pnl_value is None and realized_pnl is not None:
                try:
                    pnl_value = float(realized_pnl)
                except (TypeError, ValueError):
                    pnl_value = None

            # v1: if we cannot infer pnl reliably, stop streak scan
            if pnl_value is None:
                break

            if pnl_value < 0:
                streak += 1
            else:
                break

        return streak

    def _count_recent_executed_trades(self, recent_memory: List[dict]) -> int:
        count = 0
        for episode in recent_memory:
            execution = episode.get("execution_result") or {}
            if str(execution.get("status", "")).lower() == "filled":
                count += 1
        return count

    # =========================================================
    # Strategy memory checks
    # =========================================================

    def _evaluate_strategy_memory(
        self,
        strategy_memory: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        risk = 0.0
        warnings: List[str] = []

        risk_adjustments = strategy_memory.get("risk_adjustments", {})
        if not isinstance(risk_adjustments, dict):
            return risk, warnings

        if risk_adjustments.get("reduce_risk", False):
            warnings.append("Strategy memory suggests reduced risk mode.")
            risk += 0.10

        if risk_adjustments.get("high_volatility_mode", False):
            warnings.append("Strategy memory indicates high volatility regime.")
            risk += 0.10

        return risk, warnings

    # =========================================================
    # Adjustment logic
    # =========================================================

    def _apply_risk_adjustments(
        self,
        proposal: TradeProposal | dict,
        current_size_pct: Optional[float],
        current_stop_loss_pct: Optional[float],
        warnings: List[str],
        risk_score: float,
    ) -> tuple[Optional[float], Optional[float], List[str]]:
        size_pct = current_size_pct
        stop_loss_pct = current_stop_loss_pct
        new_warnings: List[str] = []

        if size_pct is None:
            return size_pct, stop_loss_pct, new_warnings

        warning_text = " ".join(warnings).lower()

        # high/medium risk based sizing
        if risk_score >= self.medium_risk_score_threshold:
            size_pct *= self.elevated_risk_size_multiplier
            new_warnings.append("Position size reduced due to elevated overall risk.")

        # volatility-specific cuts
        if "volatility" in warning_text:
            size_pct *= self.volatility_size_cut_ratio
            new_warnings.append("Position size reduced due to volatility-related risk.")

            if stop_loss_pct is not None:
                stop_loss_pct *= self.volatility_stop_loss_tighten_ratio
                new_warnings.append("Stop loss tightened due to volatility-related risk.")

        # conflicting factors -> additional cut
        if "conflicting factor" in warning_text or "mixed_signals" in warning_text:
            size_pct *= self.conflict_size_cut_ratio
            new_warnings.append("Position size reduced due to conflicting signals.")

        # clamp lower bound
        size_pct = max(0.0, size_pct)

        return size_pct, stop_loss_pct, new_warnings

    # =========================================================
    # Summary helpers
    # =========================================================

    def _build_summary(
        self,
        approved: bool,
        proposal: TradeProposal | dict,
        risk_score: float,
        warnings: List[str],
        rejection_reason: Optional[str],
    ) -> str:
        action = self._proposal_action(proposal)

        if approved:
            if warnings:
                return (
                    f"{action.upper()} approved with risk adjustments. "
                    f"risk_score={risk_score:.2f}. warnings={len(warnings)}."
                )
            return f"{action.upper()} approved. risk_score={risk_score:.2f}."

        return (
            f"{action.upper()} rejected. risk_score={risk_score:.2f}. "
            f"reason={rejection_reason or 'unspecified'}"
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
