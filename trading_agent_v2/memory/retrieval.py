from __future__ import annotations

from typing import Any


class MemoryRetriever:
    """
    Lightweight in-memory retrieval for recent episodes.
    Uses heuristic similarity so it can run without vector DB dependencies.
    """

    def retrieve_similar_cases(
        self,
        features: dict[str, Any],
        episodes: list[dict[str, Any]],
        top_k: int = 5,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        if not episodes:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for episode in episodes:
            if symbol and episode.get("symbol") != symbol:
                continue
            score = self._similarity_score(features, episode)
            scored.append((score, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        output: list[dict[str, Any]] = []
        for score, episode in scored[: max(1, top_k)]:
            execution = episode.get("execution_result") or {}
            outcome = self._infer_outcome(episode)
            output.append(
                {
                    "symbol": episode.get("symbol"),
                    "timestamp": episode.get("timestamp"),
                    "action": (episode.get("final_decision") or {}).get("action"),
                    "status": execution.get("status"),
                    "outcome": outcome,
                    "similarity": round(score, 4),
                    "risk_score": (episode.get("risk_report") or {}).get("risk_score"),
                    "thesis": (episode.get("proposal") or {}).get("thesis"),
                }
            )
        return output

    def retrieve_recent_failures(
        self,
        episodes: list[dict[str, Any]],
        limit: int = 5,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for episode in reversed(episodes):
            if symbol and episode.get("symbol") != symbol:
                continue
            outcome = self._infer_outcome(episode)
            if outcome in {"loss", "rejected", "failed"}:
                rows.append(
                    {
                        "symbol": episode.get("symbol"),
                        "timestamp": episode.get("timestamp"),
                        "action": (episode.get("final_decision") or {}).get("action"),
                        "outcome": outcome,
                        "risk_score": (episode.get("risk_report") or {}).get("risk_score"),
                    }
                )
            if len(rows) >= limit:
                break
        return rows

    def build_context_bundle(
        self,
        features: dict[str, Any],
        episodes: list[dict[str, Any]],
        symbol: str,
        similar_k: int = 5,
        recent_failure_k: int = 3,
    ) -> dict[str, Any]:
        similar_cases = self.retrieve_similar_cases(
            features=features,
            episodes=episodes,
            top_k=similar_k,
            symbol=symbol,
        )
        recent_failures = self.retrieve_recent_failures(
            episodes=episodes,
            limit=recent_failure_k,
            symbol=symbol,
        )
        return {
            "symbol": symbol,
            "similar_cases": similar_cases,
            "recent_failures": recent_failures,
            "sample_size": len(episodes),
        }

    def _similarity_score(self, features: dict[str, Any], episode: dict[str, Any]) -> float:
        raw_context = episode.get("raw_context") or {}
        market = raw_context.get("market_data") or {}
        social = raw_context.get("social_data") or {}

        score = 0.0
        weight_sum = 0.0

        target_trend = str(features.get("trend", "")).lower()
        hist_trend = "up" if bool(market.get("ema_fast_above_slow", False)) else "down"
        if target_trend:
            weight_sum += 0.30
            score += 0.30 if target_trend == hist_trend else 0.0

        target_rsi = self._safe_float(features.get("rsi"), None)
        hist_rsi = self._safe_float(market.get("rsi"), None)
        if target_rsi is not None and hist_rsi is not None:
            weight_sum += 0.25
            score += 0.25 * self._distance_similarity(target_rsi, hist_rsi, max_range=40.0)

        target_atr = self._safe_float(features.get("atr"), None)
        hist_atr = self._safe_float(market.get("atr_pct"), None)
        if target_atr is not None and hist_atr is not None:
            weight_sum += 0.20
            score += 0.20 * self._distance_similarity(target_atr, hist_atr, max_range=0.05)

        target_sentiment = self._safe_float(features.get("social_sentiment"), None)
        hist_sentiment_raw = self._safe_float(social.get("sentiment_score"), None)
        hist_sentiment = None
        if hist_sentiment_raw is not None:
            hist_sentiment = (hist_sentiment_raw - 0.5) * 2.0
        if target_sentiment is not None and hist_sentiment is not None:
            weight_sum += 0.25
            score += 0.25 * self._distance_similarity(target_sentiment, hist_sentiment, max_range=2.0)

        if weight_sum <= 0:
            return 0.0
        return score / weight_sum

    def _infer_outcome(self, episode: dict[str, Any]) -> str:
        execution = episode.get("execution_result") or {}
        status = str(execution.get("status", "")).lower()

        if status in {"rejected", "failed"}:
            return status
        if status != "filled":
            return "flat"

        pnl = self._safe_float((execution.get("metadata") or {}).get("realized_pnl"), None)
        if pnl is None:
            snapshot = episode.get("portfolio_snapshot") or {}
            pnl = self._safe_float(snapshot.get("realized_pnl"), None)
        if pnl is None:
            return "filled_unknown"
        if pnl > 0:
            return "win"
        if pnl < 0:
            return "loss"
        return "flat"

    @staticmethod
    def _distance_similarity(a: float, b: float, max_range: float) -> float:
        if max_range <= 0:
            return 0.0
        distance = abs(a - b)
        return max(0.0, min(1.0, 1.0 - distance / max_range))

    @staticmethod
    def _safe_float(value: Any, default: float | None) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
