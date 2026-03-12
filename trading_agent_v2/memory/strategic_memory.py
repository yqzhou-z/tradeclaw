from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from trading_agent_v2.schemas import StrategyMemory


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class StrategicMemoryStore:
    def __init__(self, memory_file: str):
        self.memory_file = memory_file

    def load(self) -> StrategyMemory:
        if not os.path.exists(self.memory_file):
            return StrategyMemory(updated_at=utc_now_iso())

        with open(self.memory_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return StrategyMemory(updated_at=utc_now_iso())

        return StrategyMemory(
            updated_at=str(data.get("updated_at", "")),
            active_insights=list(data.get("active_insights", [])),
            risk_adjustments=dict(data.get("risk_adjustments", {})),
            performance_summary=dict(data.get("performance_summary", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, memory: StrategyMemory) -> None:
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(memory.to_dict(), f, indent=4, ensure_ascii=False)

    def refresh_from_recent_episodes(self, episodes: list[dict]) -> StrategyMemory:
        memory = self.load()

        performance_summary = self._build_performance_summary(episodes)
        risk_adjustments = self._derive_risk_adjustments(performance_summary)
        active_insights = self._build_active_insights(performance_summary, risk_adjustments)

        memory.updated_at = utc_now_iso()
        memory.performance_summary = performance_summary
        memory.risk_adjustments = risk_adjustments
        memory.active_insights = active_insights
        memory.metadata = {
            "episode_sample_size": len(episodes),
        }

        self.save(memory)
        return memory

    def _build_performance_summary(self, episodes: list[dict]) -> dict:
        realized_series: list[float] = []
        filled_count = 0

        for episode in episodes:
            execution = episode.get("execution_result") or {}
            if str(execution.get("status", "")).lower() == "filled":
                filled_count += 1

            snapshot = episode.get("portfolio_snapshot") or {}
            realized_value = snapshot.get("realized_pnl")
            if realized_value is None:
                continue
            try:
                realized_series.append(float(realized_value))
            except (TypeError, ValueError):
                continue

        deltas = []
        for idx in range(1, len(realized_series)):
            deltas.append(realized_series[idx] - realized_series[idx - 1])

        win_count = sum(1 for value in deltas if value > 0)
        non_zero_outcomes = [value for value in deltas if value != 0]
        outcome_count = len(non_zero_outcomes)
        win_rate = (win_count / outcome_count) if outcome_count > 0 else 0.0

        loss_streak = 0
        for value in reversed(non_zero_outcomes):
            if value < 0:
                loss_streak += 1
            else:
                break

        avg_abs_change = (
            sum(abs(value) for value in non_zero_outcomes) / outcome_count
            if outcome_count > 0
            else 0.0
        )

        return {
            "filled_trade_count": filled_count,
            "outcome_count": outcome_count,
            "win_rate": round(win_rate, 4),
            "recent_loss_streak": loss_streak,
            "avg_abs_realized_pnl_change": round(avg_abs_change, 4),
            "last_realized_pnl": round(realized_series[-1], 4) if realized_series else 0.0,
        }

    def _derive_risk_adjustments(self, perf: dict) -> dict:
        filled_trade_count = int(perf.get("filled_trade_count", 0))
        win_rate = float(perf.get("win_rate", 0.0))
        loss_streak = int(perf.get("recent_loss_streak", 0))
        avg_abs_change = float(perf.get("avg_abs_realized_pnl_change", 0.0))

        reduce_risk = loss_streak >= 2 or (filled_trade_count >= 5 and win_rate < 0.40)
        high_volatility_mode = avg_abs_change >= 80.0

        position_scale = 1.0
        if reduce_risk:
            position_scale *= 0.75
        if high_volatility_mode:
            position_scale *= 0.80

        return {
            "reduce_risk": reduce_risk,
            "high_volatility_mode": high_volatility_mode,
            "position_scale": round(position_scale, 4),
        }

    def _build_active_insights(self, perf: dict, risk: dict) -> list[str]:
        insights: list[str] = []

        win_rate = float(perf.get("win_rate", 0.0))
        loss_streak = int(perf.get("recent_loss_streak", 0))

        if loss_streak >= 2:
            insights.append("Recent consecutive losses detected; prioritize capital protection.")
        if win_rate >= 0.60 and int(perf.get("outcome_count", 0)) >= 5:
            insights.append("Current process shows positive edge; keep disciplined sizing.")
        if risk.get("high_volatility_mode"):
            insights.append("High realized PnL variance detected; keep tighter risk limits.")
        if not insights:
            insights.append("No strong regime signal; maintain baseline risk profile.")

        return insights
