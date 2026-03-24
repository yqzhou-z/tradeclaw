# trading_agent_v2/agents/critic_agent.py

from typing import List, Dict, Any


class CriticAgent:
    def review(
        self,
        proposals: List[Dict[str, Any]],
        features: Dict[str, Any],
        similar_cases: List[Dict[str, Any]] | None = None,
        strategy_memory: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        reviewed = []
        strategy_memory = strategy_memory or {}
        pattern_stats = strategy_memory.get("pattern_stats", {}) or {}

        trend = features.get("trend")
        news_sentiment = features.get("news_sentiment", 0)

        for proposal in proposals:
            p = proposal.copy()
            weaknesses = []

            action = p.get("action")
            size_pct = float(p.get("size_pct", 0.0))
            confidence = float(p.get("confidence", 0.5))

            if action == "buy" and trend == "down":
                weaknesses.append("Buy proposal conflicts with downward market trend.")
                confidence -= 0.12

            if action == "sell" and trend == "up":
                weaknesses.append("Sell proposal conflicts with upward market trend.")
                confidence -= 0.12

            if action == "buy" and news_sentiment is not None and news_sentiment < 0:
                weaknesses.append("Buy proposal faces negative news sentiment.")
                confidence -= 0.08

            if action == "sell" and news_sentiment is not None and news_sentiment > 0:
                weaknesses.append("Sell proposal faces positive news sentiment.")
                confidence -= 0.08

            if size_pct > 0.12:
                weaknesses.append("Position size is relatively aggressive.")
                confidence -= 0.05

            # similar case check (same action only)
            if similar_cases:
                matched_cases = [
                    case for case in similar_cases
                    if str(case.get("action", "")).lower() == action
                ]
                bad_count = sum(
                    1 for case in matched_cases
                    if str(case.get("outcome", "")).lower() == "loss"
                )
                if matched_cases and bad_count >= max(2, len(matched_cases) // 2):
                    weaknesses.append("Recent similar cases underperformed for this action.")
                    confidence -= 0.10

            # pattern memory check (aligned with planner regime key)
            pattern_key = self._build_pattern_key(features=features, action=action)
            stats = pattern_stats.get(pattern_key, {}) if isinstance(pattern_stats, dict) else {}
            if isinstance(stats, dict):
                count = int(stats.get("count", 0))
                wins = int(stats.get("wins", 0))
                losses = int(stats.get("losses", 0))
                win_rate = (wins / count) if count > 0 else 0.0

                if count >= 3 and losses > wins:
                    weaknesses.append(
                        f"Pattern memory underperforms in current regime ({pattern_key}, win_rate={win_rate:.2f})."
                    )
                    confidence -= 0.10
                elif count >= 3 and wins > losses and win_rate >= 0.60:
                    confidence += 0.03

            p["critic_weaknesses"] = weaknesses
            p["critic_score"] = max(0.0, min(1.0, confidence))
            p["confidence"] = max(0.0, min(1.0, confidence))

            reviewed.append(p)

        reviewed.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return reviewed

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

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
