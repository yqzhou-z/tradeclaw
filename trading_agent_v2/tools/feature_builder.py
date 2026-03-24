from __future__ import annotations

from typing import Any, Dict


class FeatureBuilder:
    def build(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        news_data: Dict[str, Any],
        onchain_data: Dict[str, Any],
        social_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        items = news_data.get("items", []) or []
        headlines = [str(item.get("title", "")) for item in items if isinstance(item, dict)]
        news_summary = news_data.get("summary")
        if not news_summary:
            news_summary = "; ".join(headlines[:3]) if headlines else "No recent headlines."

        market_price = self._safe_float(market_data.get("price"), 0.0)
        rsi = self._safe_float(market_data.get("rsi"), 50.0)
        atr_pct = self._safe_float(market_data.get("atr_pct"), 0.0)
        trend = self._infer_trend(market_data)

        news_sentiment = self._normalize_sentiment(
            self._pick_first_non_none(
                news_data.get("sentiment"),
                news_data.get("sentiment_score"),
                social_data.get("sentiment_score"),
            )
        )
        social_sentiment = self._normalize_sentiment(
            self._pick_first_non_none(
                social_data.get("sentiment"),
                social_data.get("sentiment_score"),
                news_data.get("sentiment"),
            )
        )

        mentions_change_pct = self._safe_float(
            self._pick_first_non_none(
                social_data.get("mentions_change_pct"),
                social_data.get("mentions"),
            ),
            0.0,
        )
        engagement_score = self._safe_float(
            self._pick_first_non_none(
                social_data.get("engagement_score"),
                social_data.get("score"),
            ),
            0.0,
        )

        onchain_signal = self._infer_onchain_signal(onchain_data)
        onchain_score = self._score_onchain_signal(onchain_data)

        return {
            "symbol": symbol,
            "price": market_price,
            "trend": trend,
            "rsi": rsi,
            "atr": atr_pct,
            "volatility": atr_pct,
            "news_sentiment": news_sentiment,
            "news_headlines": headlines,
            "news_summary": news_summary,
            "onchain_signal": onchain_signal,
            "onchain_score": onchain_score,
            "social_sentiment": social_sentiment,
            "social_mentions": mentions_change_pct,
            "social_score": engagement_score,
        }

    def _infer_trend(self, market_data: Dict[str, Any]) -> str:
        trend = market_data.get("trend")
        if isinstance(trend, str) and trend:
            return trend
        if bool(market_data.get("ema_fast_above_slow", False)):
            return "up"
        return "down"

    def _infer_onchain_signal(self, onchain_data: Dict[str, Any]) -> str:
        outflow = str(onchain_data.get("exchange_outflow", "moderate")).lower()
        whale = str(onchain_data.get("whale_activity", "neutral")).lower()
        stablecoin = str(onchain_data.get("stablecoin_flow", "neutral")).lower()

        if whale == "accumulation" and outflow in {"strong", "moderate"} and stablecoin == "inflow":
            return "bullish"
        if whale == "distribution" and stablecoin == "outflow":
            return "bearish"
        return "neutral"

    def _score_onchain_signal(self, onchain_data: Dict[str, Any]) -> float:
        signal = self._infer_onchain_signal(onchain_data)
        if signal == "bullish":
            return 0.8
        if signal == "bearish":
            return -0.8
        return 0.0

    @staticmethod
    def _pick_first_non_none(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _normalize_sentiment(self, value: Any) -> float:
        numeric = self._safe_float(value, 0.5)
        if 0.0 <= numeric <= 1.0:
            return round((numeric - 0.5) * 2.0, 4)
        return round(max(-1.0, min(1.0, numeric)), 4)
