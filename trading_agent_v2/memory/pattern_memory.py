from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PatternMemoryStore:
    """
    Rule-based pattern memory.
    Aggregates historical outcomes by compact regime keys to support
    future proposal filtering or sizing adjustments.
    """

    def __init__(self, memory_file: str):
        self.memory_file = memory_file

    def load(self) -> dict[str, Any]:
        if not os.path.exists(self.memory_file):
            return self._empty_store()

        with open(self.memory_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return self._empty_store()

        if not isinstance(data, dict):
            return self._empty_store()

        data.setdefault("updated_at", utc_now_iso())
        data.setdefault("patterns", {})
        data.setdefault("metadata", {})
        return data

    def save(self, payload: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

    def refresh_from_episodes(
        self,
        episodes: list[dict[str, Any]],
        min_count_for_signal: int = 3,
    ) -> dict[str, Any]:
        patterns: dict[str, dict[str, Any]] = {}
        realized_series = self._extract_realized_series(episodes)

        for idx, episode in enumerate(episodes):
            key = self._build_pattern_key(episode)
            if not key:
                continue

            stats = patterns.setdefault(
                key,
                {
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "flats": 0,
                    "avg_pnl_delta": 0.0,
                    "total_pnl_delta": 0.0,
                },
            )
            stats["count"] += 1

            pnl_delta = self._episode_realized_delta(idx, realized_series)
            if pnl_delta > 0:
                stats["wins"] += 1
            elif pnl_delta < 0:
                stats["losses"] += 1
            else:
                stats["flats"] += 1

            stats["total_pnl_delta"] += pnl_delta
            stats["avg_pnl_delta"] = stats["total_pnl_delta"] / stats["count"]

        insights = self._build_pattern_insights(patterns, min_count=min_count_for_signal)
        payload = {
            "updated_at": utc_now_iso(),
            "patterns": patterns,
            "metadata": {
                "episode_sample_size": len(episodes),
                "insights": insights,
                "min_count_for_signal": min_count_for_signal,
            },
        }
        self.save(payload)
        return payload

    def lookup(
        self,
        trend: str,
        sentiment_bucket: str,
        action: str,
    ) -> dict[str, Any] | None:
        store = self.load()
        key = f"trend={trend}|sentiment={sentiment_bucket}|action={action}"
        return store.get("patterns", {}).get(key)

    def _build_pattern_key(self, episode: dict[str, Any]) -> str | None:
        raw_context = episode.get("raw_context") or {}
        market = raw_context.get("market_data") or {}
        social = raw_context.get("social_data") or {}
        final_decision = episode.get("final_decision") or {}

        action = str(final_decision.get("action", "hold")).lower()
        if action not in {"buy", "sell", "hold"}:
            return None

        trend = "up" if bool(market.get("ema_fast_above_slow", False)) else "down"
        sentiment_score = self._safe_float(social.get("sentiment_score"), 0.5)
        sentiment_bucket = self._sentiment_bucket(sentiment_score)
        return f"trend={trend}|sentiment={sentiment_bucket}|action={action}"

    def _extract_realized_series(self, episodes: list[dict[str, Any]]) -> list[float]:
        series: list[float] = []
        for episode in episodes:
            snapshot = episode.get("portfolio_snapshot") or {}
            value = self._safe_float(snapshot.get("realized_pnl"), 0.0)
            series.append(value)
        return series

    def _episode_realized_delta(self, idx: int, realized_series: list[float]) -> float:
        if not realized_series:
            return 0.0
        if idx <= 0:
            return 0.0
        if idx >= len(realized_series):
            return 0.0
        return realized_series[idx] - realized_series[idx - 1]

    def _build_pattern_insights(
        self,
        patterns: dict[str, dict[str, Any]],
        min_count: int,
    ) -> list[str]:
        insights: list[str] = []

        for key, stats in patterns.items():
            count = int(stats.get("count", 0))
            if count < min_count:
                continue

            wins = int(stats.get("wins", 0))
            losses = int(stats.get("losses", 0))
            win_rate = wins / count if count > 0 else 0.0

            if win_rate >= 0.65 and wins > losses:
                insights.append(f"{key} shows positive edge (win_rate={win_rate:.2f}).")
            elif win_rate <= 0.35 and losses > wins:
                insights.append(f"{key} underperforms (win_rate={win_rate:.2f}).")

        if not insights:
            insights.append("No robust pattern signal yet; keep baseline policy.")
        return insights

    @staticmethod
    def _sentiment_bucket(score_0_to_1: float) -> str:
        if score_0_to_1 >= 0.62:
            return "positive"
        if score_0_to_1 <= 0.38:
            return "negative"
        return "neutral"

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _empty_store() -> dict[str, Any]:
        return {
            "updated_at": utc_now_iso(),
            "patterns": {},
            "metadata": {},
        }
