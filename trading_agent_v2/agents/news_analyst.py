from __future__ import annotations

from dataclasses import dataclass

from trading_agent_v2.schemas import AnalystView, RawContext


@dataclass
class NewsAnalystConfig:
    bullish_sentiment_threshold: float = 0.58
    bearish_sentiment_threshold: float = 0.42
    high_mentions_threshold: float = 0.25


class NewsAnalyst:
    def __init__(self, config: NewsAnalystConfig | None = None):
        self.config = config or NewsAnalystConfig()

    def analyze(self, raw_context: RawContext) -> AnalystView:
        news_data = raw_context.news_data or []
        social = raw_context.social_data or {}
        onchain = raw_context.onchain_data or {}

        article_count = len(news_data)
        sentiment_score = float(social.get("sentiment_score", 0.5))
        mentions_change_pct = float(social.get("mentions_change_pct", 0.0))
        stablecoin_flow = str(onchain.get("stablecoin_flow", "neutral")).lower()

        bias = "neutral"
        confidence = 0.50
        summary = "News flow is balanced."
        supporting_signals: list[str] = []
        risk_flags: list[str] = []

        if sentiment_score >= self.config.bullish_sentiment_threshold and article_count > 0:
            bias = "bullish"
            confidence += 0.12
            supporting_signals.extend(["positive_headlines", "constructive_social_sentiment"])

        if sentiment_score <= self.config.bearish_sentiment_threshold and article_count > 0:
            bias = "bearish"
            confidence += 0.12
            supporting_signals.extend(["negative_headlines", "weak_social_sentiment"])

        if stablecoin_flow == "inflow":
            confidence += 0.04
            supporting_signals.append("liquidity_support")
        elif stablecoin_flow == "outflow":
            confidence -= 0.04
            risk_flags.append("liquidity_drain_risk")

        if abs(mentions_change_pct) >= self.config.high_mentions_threshold:
            risk_flags.append("headline_reversal_risk")
            confidence -= 0.06

        if article_count == 0:
            risk_flags.append("missing_news_coverage")
            confidence -= 0.08

        confidence = max(0.0, min(1.0, confidence))

        if bias == "bullish":
            summary = "News and sentiment are mildly supportive."
        elif bias == "bearish":
            summary = "News tone and social sentiment are risk-off."

        return AnalystView(
            analyst_name="news_analyst",
            symbol=raw_context.symbol,
            timestamp=raw_context.timestamp,
            bias=bias,
            confidence=confidence,
            summary=summary,
            supporting_signals=self._dedupe_preserve_order(supporting_signals),
            risk_flags=self._dedupe_preserve_order(risk_flags),
            details={
                "article_count": article_count,
                "sentiment_score": sentiment_score,
                "mentions_change_pct": mentions_change_pct,
                "stablecoin_flow": stablecoin_flow,
            },
        )

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
