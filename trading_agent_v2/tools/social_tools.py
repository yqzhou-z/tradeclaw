from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import requests


class SocialTools:
    COINGECKO_HISTORY_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/history"
    COIN_ID_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
    }

    def __init__(self, timeout_sec: int = 10):
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self._historical_coin_cache: dict[tuple[str, str], dict[str, Any] | None] = {}

    def get_social_snapshot(self, symbol: str, news_data: list[dict] | None = None) -> dict:
        news_data = news_data or []
        base_coin = symbol.split("/")[0].upper()
        coin_id = self.COIN_ID_MAP.get(base_coin, base_coin.lower())

        coin_data = self._fetch_coin_community(coin_id)
        news_sentiment = self._compute_news_sentiment(news_data)

        if coin_data:
            community = coin_data.get("community_data") or {}
            upvote_pct = self._safe_float(coin_data.get("sentiment_votes_up_percentage"), 50.0)
            community_sentiment = max(0.0, min(1.0, upvote_pct / 100.0))
            sentiment_score = self._blend_sentiment(community_sentiment, news_sentiment)

            twitter_followers = self._safe_float(community.get("twitter_followers"), 0.0)
            reddit_subscribers = self._safe_float(community.get("reddit_subscribers"), 0.0)
            reddit_posts_48h = self._safe_float(community.get("reddit_average_posts_48h"), 0.0)
            reddit_comments_48h = self._safe_float(community.get("reddit_average_comments_48h"), 0.0)

            mentions_change_pct = self._estimate_mentions_change_pct(
                reddit_posts_48h=reddit_posts_48h,
                reddit_comments_48h=reddit_comments_48h,
            )
            engagement_score = self._estimate_engagement_score(
                twitter_followers=twitter_followers,
                reddit_subscribers=reddit_subscribers,
                reddit_posts_48h=reddit_posts_48h,
                reddit_comments_48h=reddit_comments_48h,
            )

            return {
                "asset": base_coin,
                "sentiment_score": round(sentiment_score, 4),
                "mentions_change_pct": round(mentions_change_pct, 4),
                "engagement_score": round(engagement_score, 2),
                "twitter_followers": int(twitter_followers),
                "reddit_subscribers": int(reddit_subscribers),
                "source": "coingecko_community",
                "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }

        # fallback: derive social profile from news if community endpoint is unavailable
        mentions_change_pct = min(0.8, max(-0.5, len(news_data) * 0.04 - 0.05))
        engagement_score = 40 + (news_sentiment * 50) + abs(mentions_change_pct) * 20
        return {
            "asset": base_coin,
            "sentiment_score": round(news_sentiment, 4),
            "mentions_change_pct": round(mentions_change_pct, 4),
            "engagement_score": round(engagement_score, 2),
            "source": "fallback:news_derived",
            "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

    def get_historical_social_snapshot(
        self,
        symbol: str,
        as_of: str | datetime,
        *,
        news_data: list[dict] | None = None,
    ) -> dict:
        news_data = news_data or []
        as_of_dt = self._parse_datetime(as_of)
        if as_of_dt is None:
            return self.get_social_snapshot(symbol, news_data=news_data)

        base_coin = symbol.split("/")[0].upper()
        coin_id = self.COIN_ID_MAP.get(base_coin, base_coin.lower())

        current_data = self._fetch_coin_history(coin_id, as_of_dt)
        previous_data = self._fetch_coin_history(coin_id, as_of_dt - timedelta(days=1))
        news_sentiment = self._compute_news_sentiment(news_data)

        if current_data:
            community = current_data.get("community_data") or {}
            prev_community = (previous_data or {}).get("community_data") or {}
            upvote_pct = self._safe_float(current_data.get("sentiment_votes_up_percentage"), 50.0)
            community_sentiment = max(0.0, min(1.0, upvote_pct / 100.0))
            sentiment_score = self._blend_sentiment(community_sentiment, news_sentiment)

            twitter_followers = self._safe_float(community.get("twitter_followers"), 0.0)
            reddit_subscribers = self._safe_float(community.get("reddit_subscribers"), 0.0)
            reddit_posts_48h = self._safe_float(community.get("reddit_average_posts_48h"), 0.0)
            reddit_comments_48h = self._safe_float(community.get("reddit_average_comments_48h"), 0.0)

            prev_reddit_subscribers = self._safe_float(prev_community.get("reddit_subscribers"), 0.0)
            prev_activity = self._safe_float(prev_community.get("reddit_average_posts_48h"), 0.0) + (
                self._safe_float(prev_community.get("reddit_average_comments_48h"), 0.0) / 5.0
            )
            current_activity = reddit_posts_48h + (reddit_comments_48h / 5.0)

            mentions_change_pct = self._compute_mentions_change(
                current_subscribers=reddit_subscribers,
                previous_subscribers=prev_reddit_subscribers,
                current_activity=current_activity,
                previous_activity=prev_activity,
            )
            engagement_score = self._estimate_engagement_score(
                twitter_followers=twitter_followers,
                reddit_subscribers=reddit_subscribers,
                reddit_posts_48h=reddit_posts_48h,
                reddit_comments_48h=reddit_comments_48h,
            )

            return {
                "asset": base_coin,
                "sentiment_score": round(sentiment_score, 4),
                "mentions_change_pct": round(mentions_change_pct, 4),
                "engagement_score": round(engagement_score, 2),
                "twitter_followers": int(twitter_followers),
                "reddit_subscribers": int(reddit_subscribers),
                "source": "coingecko_history_community",
                "fetched_at": as_of_dt.replace(microsecond=0).isoformat(),
            }

        mentions_change_pct = min(0.8, max(-0.5, len(news_data) * 0.04 - 0.05))
        engagement_score = 40 + (news_sentiment * 50) + abs(mentions_change_pct) * 20
        return {
            "asset": base_coin,
            "sentiment_score": round(news_sentiment, 4),
            "mentions_change_pct": round(mentions_change_pct, 4),
            "engagement_score": round(engagement_score, 2),
            "source": "fallback:historical_news_derived",
            "fetched_at": as_of_dt.replace(microsecond=0).isoformat(),
        }

    def _fetch_coin_community(self, coin_id: str) -> dict[str, Any] | None:
        try:
            response = self.session.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "false",
                    "community_data": "true",
                    "developer_data": "false",
                    "sparkline": "false",
                },
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return None
            return payload
        except Exception:
            return None

    def _fetch_coin_history(self, coin_id: str, as_of_dt: datetime) -> dict[str, Any] | None:
        date_key = as_of_dt.astimezone(timezone.utc).strftime("%d-%m-%Y")
        cache_key = (coin_id, date_key)
        if cache_key in self._historical_coin_cache:
            return self._historical_coin_cache[cache_key]

        try:
            response = self.session.get(
                self.COINGECKO_HISTORY_URL.format(coin_id=coin_id),
                params={
                    "date": date_key,
                    "localization": "false",
                },
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                payload = None
        except Exception:
            payload = None

        self._historical_coin_cache[cache_key] = payload
        return payload

    def _compute_news_sentiment(self, news_data: list[dict]) -> float:
        if not news_data:
            return 0.5
        sentiment_map = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
        values = [
            sentiment_map.get(str(item.get("sentiment", "neutral")).lower(), 0.5)
            for item in news_data
        ]
        return sum(values) / len(values)

    def _blend_sentiment(self, community_sentiment: float, news_sentiment: float) -> float:
        # community votes are noisy; combine with headline tone
        return max(0.0, min(1.0, community_sentiment * 0.6 + news_sentiment * 0.4))

    def _estimate_mentions_change_pct(self, reddit_posts_48h: float, reddit_comments_48h: float) -> float:
        activity = reddit_posts_48h + (reddit_comments_48h / 5.0)
        scaled = math.tanh(activity / 25.0) * 0.5
        return max(-0.5, min(0.8, scaled))

    def _compute_mentions_change(
        self,
        current_subscribers: float,
        previous_subscribers: float,
        current_activity: float,
        previous_activity: float,
    ) -> float:
        subscriber_change = 0.0
        if previous_subscribers > 0:
            subscriber_change = (current_subscribers - previous_subscribers) / previous_subscribers

        activity_change = 0.0
        if previous_activity > 0:
            activity_change = (current_activity - previous_activity) / previous_activity
        elif current_activity > 0:
            activity_change = 1.0

        blended = subscriber_change * 0.4 + activity_change * 0.6
        return max(-0.8, min(1.2, blended))

    def _estimate_engagement_score(
        self,
        twitter_followers: float,
        reddit_subscribers: float,
        reddit_posts_48h: float,
        reddit_comments_48h: float,
    ) -> float:
        size_component = math.log1p(max(0.0, twitter_followers) + max(0.0, reddit_subscribers)) / 12.0
        activity_component = math.log1p(max(0.0, reddit_posts_48h) + max(0.0, reddit_comments_48h)) / 6.0
        raw = 30 + size_component * 45 + activity_component * 25
        return max(0.0, min(100.0, raw))

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_datetime(self, value: str | datetime) -> datetime | None:
        if isinstance(value, datetime):
            dt = value
        else:
            text = str(value or "").strip()
            if not text:
                return None
            candidates = [text]
            if text.endswith("Z"):
                candidates.append(text[:-1] + "+00:00")
            dt = None
            for candidate in candidates:
                try:
                    dt = datetime.fromisoformat(candidate)
                    break
                except ValueError:
                    continue
            if dt is None:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
