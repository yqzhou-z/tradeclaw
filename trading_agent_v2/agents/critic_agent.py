from __future__ import annotations

from typing import Any, Dict, List


class CriticAgent:
    """
    Explainable critic.
    Evaluates each proposal through auditable checks and exposes score impacts.
    """

    def review(
        self,
        proposals: List[Dict[str, Any]],
        features: Dict[str, Any],
        similar_cases: List[Dict[str, Any]] | None = None,
        strategy_memory: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        reviewed = []
        similar_cases = similar_cases or []
        strategy_memory = strategy_memory or {}
        pattern_stats = strategy_memory.get("pattern_stats", {}) or {}

        for proposal in proposals:
            reviewed.append(
                self._review_single(
                    proposal=proposal,
                    features=features,
                    similar_cases=similar_cases,
                    pattern_stats=pattern_stats,
                )
            )

        reviewed.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return reviewed

    def _review_single(
        self,
        proposal: Dict[str, Any],
        features: Dict[str, Any],
        similar_cases: List[Dict[str, Any]],
        pattern_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        p = proposal.copy()
        action = str(p.get("action", "hold")).lower()
        size_pct = self._safe_float(p.get("size_pct"), 0.0)
        start_conf = self._safe_float(p.get("confidence"), 0.50)

        checks: List[Dict[str, Any]] = []
        checks.extend(self._check_directional_alignment(action, features))
        checks.extend(self._check_sentiment_alignment(action, features))
        checks.extend(self._check_size_and_volatility(action, size_pct, features))
        checks.extend(self._check_similar_cases(action, similar_cases))
        checks.extend(self._check_pattern_regime(action, features, pattern_stats))
        checks.extend(self._check_reasoning_trace(p))

        total_delta = sum(self._safe_float(item.get("impact"), 0.0) for item in checks)
        final_conf = max(0.0, min(1.0, start_conf + total_delta))

        weaknesses = [item["message"] for item in checks if item.get("impact", 0.0) < 0]
        strengths = [item["message"] for item in checks if item.get("impact", 0.0) > 0]

        p["critic_checks"] = checks
        p["critic_strengths"] = strengths
        p["critic_weaknesses"] = weaknesses
        p["critic_score"] = final_conf
        p["confidence"] = final_conf
        p["critic_reasoning"] = {
            "critic_type": "explainable_critic_v1",
            "start_confidence": round(start_conf, 6),
            "total_delta": round(total_delta, 6),
            "final_confidence": round(final_conf, 6),
            "penalty_count": len(weaknesses),
            "credit_count": len(strengths),
        }
        return p

    def _check_directional_alignment(
        self,
        action: str,
        features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        trend = str(features.get("trend", "neutral")).lower()
        if action not in {"buy", "sell"}:
            return [
                self._check(
                    name="direction_alignment",
                    impact=0.01,
                    message="Non-directional action avoids trend mismatch risk.",
                    category="market",
                )
            ]

        if action == "buy" and trend == "down":
            return [
                self._check(
                    name="direction_alignment",
                    impact=-0.12,
                    message="BUY conflicts with downtrend.",
                    category="market",
                )
            ]
        if action == "sell" and trend == "up":
            return [
                self._check(
                    name="direction_alignment",
                    impact=-0.12,
                    message="SELL conflicts with uptrend.",
                    category="market",
                )
            ]
        return [
            self._check(
                name="direction_alignment",
                impact=0.02,
                message=f"{action.upper()} aligns with current trend.",
                category="market",
            )
        ]

    def _check_sentiment_alignment(
        self,
        action: str,
        features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if action not in {"buy", "sell"}:
            return []

        news_sent = self._safe_float(features.get("news_sentiment"), 0.0)
        social_sent = self._safe_float(features.get("social_sentiment"), 0.0)
        blended = news_sent * 0.6 + social_sent * 0.4

        if action == "buy" and blended < -0.12:
            return [
                self._check(
                    name="sentiment_alignment",
                    impact=-0.08,
                    message=f"BUY faces adverse sentiment (blended={blended:.3f}).",
                    category="sentiment",
                )
            ]
        if action == "sell" and blended > 0.12:
            return [
                self._check(
                    name="sentiment_alignment",
                    impact=-0.08,
                    message=f"SELL opposes constructive sentiment (blended={blended:.3f}).",
                    category="sentiment",
                )
            ]

        if abs(blended) < 0.08:
            return [
                self._check(
                    name="sentiment_alignment",
                    impact=0.0,
                    message=f"Sentiment is neutral (blended={blended:.3f}).",
                    category="sentiment",
                )
            ]
        return [
            self._check(
                name="sentiment_alignment",
                impact=0.02,
                message=f"Sentiment supports {action.upper()} (blended={blended:.3f}).",
                category="sentiment",
            )
        ]

    def _check_size_and_volatility(
        self,
        action: str,
        size_pct: float,
        features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        checks: List[Dict[str, Any]] = []
        if action not in {"buy", "sell"}:
            return checks

        atr = self._safe_float(features.get("atr"), 0.0)
        if size_pct > 0.12:
            checks.append(
                self._check(
                    name="size_aggressiveness",
                    impact=-0.05,
                    message=f"Position size {size_pct:.3f} is aggressive.",
                    category="risk",
                )
            )
        elif size_pct <= 0.10:
            checks.append(
                self._check(
                    name="size_aggressiveness",
                    impact=0.01,
                    message=f"Position size {size_pct:.3f} is conservative.",
                    category="risk",
                )
            )

        if atr >= 0.03 and size_pct >= 0.12:
            checks.append(
                self._check(
                    name="volatility_sizing",
                    impact=-0.06,
                    message=f"High ATR ({atr:.4f}) with large size increases execution risk.",
                    category="risk",
                )
            )
        return checks

    def _check_similar_cases(
        self,
        action: str,
        similar_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if action not in {"buy", "sell"}:
            return []

        matched = [
            case for case in similar_cases
            if str(case.get("action", "")).lower() == action
        ]
        if len(matched) < 2:
            return []

        wins = sum(1 for case in matched if str(case.get("outcome", "")).lower() == "win")
        losses = sum(1 for case in matched if str(case.get("outcome", "")).lower() == "loss")
        outcome_count = wins + losses
        if outcome_count < 2:
            return []

        loss_ratio = losses / outcome_count
        win_ratio = wins / outcome_count
        if loss_ratio >= 0.6:
            return [
                self._check(
                    name="similar_case_outcome",
                    impact=-0.10,
                    message=f"Similar {action.upper()} cases underperformed (loss_ratio={loss_ratio:.2f}).",
                    category="memory",
                )
            ]
        if win_ratio >= 0.6:
            return [
                self._check(
                    name="similar_case_outcome",
                    impact=0.03,
                    message=f"Similar {action.upper()} cases were supportive (win_ratio={win_ratio:.2f}).",
                    category="memory",
                )
            ]
        return []

    def _check_pattern_regime(
        self,
        action: str,
        features: Dict[str, Any],
        pattern_stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if action not in {"buy", "sell"}:
            return []
        if not isinstance(pattern_stats, dict):
            return []

        key = self._build_pattern_key(features=features, action=action)
        stats = pattern_stats.get(key, {})
        if not isinstance(stats, dict):
            return []

        count = int(stats.get("count", 0))
        if count < 3:
            return []
        wins = self._safe_float(stats.get("wins"), 0.0)
        losses = self._safe_float(stats.get("losses"), 0.0)
        total = wins + losses
        if total <= 0:
            return []
        win_rate = wins / total
        if losses > wins:
            return [
                self._check(
                    name="pattern_regime",
                    impact=-0.10,
                    message=f"Pattern memory disfavors regime ({key}, win_rate={win_rate:.2f}).",
                    category="memory",
                )
            ]
        if wins > losses and win_rate >= 0.60:
            return [
                self._check(
                    name="pattern_regime",
                    impact=0.03,
                    message=f"Pattern memory supports regime ({key}, win_rate={win_rate:.2f}).",
                    category="memory",
                )
            ]
        return []

    def _check_reasoning_trace(self, proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        trace = proposal.get("reasoning_trace", {})
        if not isinstance(trace, dict):
            return []

        conflict = self._safe_float(trace.get("conflict_index"), 0.0)
        confidence = self._safe_float(trace.get("confidence"), 0.5)
        action = str(proposal.get("action", "hold")).lower()

        checks: List[Dict[str, Any]] = []
        if action in {"buy", "sell"} and conflict >= 0.70:
            checks.append(
                self._check(
                    name="planner_conflict_index",
                    impact=-0.04,
                    message=f"Planner evidence is highly conflicting (conflict={conflict:.2f}).",
                    category="planner_trace",
                )
            )
        if action in {"buy", "sell"} and confidence >= 0.65 and conflict <= 0.45:
            checks.append(
                self._check(
                    name="planner_conflict_index",
                    impact=0.02,
                    message="Planner trace shows clear and coherent evidence.",
                    category="planner_trace",
                )
            )
        return checks

    def _build_pattern_key(self, features: Dict[str, Any], action: str) -> str:
        trend = str(features.get("trend", "down")).lower()
        sentiment_bucket = self._sentiment_bucket(features.get("news_sentiment", 0.0))
        return f"trend={trend}|sentiment={sentiment_bucket}|action={action}"

    def _sentiment_bucket(self, value: Any) -> str:
        numeric = self._safe_float(value, 0.0)
        if numeric >= 0.2:
            return "positive"
        if numeric <= -0.2:
            return "negative"
        return "neutral"

    def _check(
        self,
        name: str,
        impact: float,
        message: str,
        category: str,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "category": category,
            "impact": round(impact, 6),
            "message": message,
        }

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

