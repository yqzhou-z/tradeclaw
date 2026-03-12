from __future__ import annotations

from datetime import datetime, timezone


class SocialTools:
    def get_social_snapshot(self, symbol: str, news_data: list[dict] | None = None) -> dict:
        news_data = news_data or []
        base_coin = symbol.split("/")[0].upper()
        now = datetime.now(timezone.utc)

        sentiment_map = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
        if news_data:
            scores = [
                sentiment_map.get(str(item.get("sentiment", "neutral")).lower(), 0.5)
                for item in news_data
            ]
            news_score = sum(scores) / len(scores)
        else:
            news_score = 0.50

        buzz_factor = ((now.minute % 30) - 15) / 100.0
        sentiment_score = max(0.0, min(1.0, news_score + buzz_factor * 0.15))
        mentions_change_pct = max(-0.5, min(0.8, buzz_factor + 0.12))

        return {
            "asset": base_coin,
            "sentiment_score": round(sentiment_score, 4),
            "mentions_change_pct": round(mentions_change_pct, 4),
            "engagement_score": round(40 + sentiment_score * 50 + abs(mentions_change_pct) * 30, 2),
        }
