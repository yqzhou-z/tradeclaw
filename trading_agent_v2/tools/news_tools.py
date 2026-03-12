from __future__ import annotations

from datetime import datetime, timedelta, timezone


class NewsTools:
    def get_latest_news(self, symbol: str, limit: int = 3) -> list[dict]:
        """
        Mock news feed with alternating tone by hour.
        """
        base_coin = symbol.split("/")[0].upper()
        now = datetime.now(timezone.utc)

        positive_templates = [
            f"{base_coin} spot demand remains resilient amid macro uncertainty",
            f"Institutional desk reports steady accumulation in {base_coin}",
            f"{base_coin} derivative funding normalizes after recent volatility",
        ]
        negative_templates = [
            f"{base_coin} sees elevated leverage and liquidation risk",
            f"Short-term traders rotate away from {base_coin} into stablecoins",
            f"{base_coin} momentum slows as breakout volume fades",
        ]
        neutral_templates = [
            f"{base_coin} trades in range while market awaits macro catalysts",
            f"Analysts remain split on near-term direction for {base_coin}",
            f"{base_coin} volatility cools after prior impulsive move",
        ]

        regime = now.hour % 3
        if regime == 0:
            sentiments = ["positive", "positive", "neutral"]
        elif regime == 1:
            sentiments = ["neutral", "neutral", "positive"]
        else:
            sentiments = ["negative", "neutral", "negative"]

        entries: list[dict] = []
        for idx in range(limit):
            sentiment = sentiments[idx % len(sentiments)]
            if sentiment == "positive":
                title = positive_templates[idx % len(positive_templates)]
            elif sentiment == "negative":
                title = negative_templates[idx % len(negative_templates)]
            else:
                title = neutral_templates[idx % len(neutral_templates)]

            published_at = (now - timedelta(minutes=20 * idx)).replace(microsecond=0).isoformat()
            entries.append(
                {
                    "title": title,
                    "summary": title,
                    "source": "mock_news",
                    "published_at": published_at,
                    "sentiment": sentiment,
                }
            )

        return entries

    def summarize_sentiment(self, news_data: list[dict]) -> dict:
        if not news_data:
            return {
                "sentiment_score": 0.50,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "article_count": 0,
            }

        score_map = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
        scores = [score_map.get(str(item.get("sentiment", "neutral")).lower(), 0.5) for item in news_data]
        sentiment_score = sum(scores) / len(scores)

        positive_count = sum(1 for s in scores if s > 0.66)
        negative_count = sum(1 for s in scores if s < 0.34)
        total = len(scores)

        return {
            "sentiment_score": round(sentiment_score, 4),
            "positive_ratio": round(positive_count / total, 4),
            "negative_ratio": round(negative_count / total, 4),
            "article_count": total,
        }
