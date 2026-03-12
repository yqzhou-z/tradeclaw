from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from trading_agent_v2.schemas import ReflectionNote


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ReflectionEngine:
    def __init__(self, reflection_file: str):
        self.reflection_file = reflection_file

    def generate_reflection(
        self,
        episode: dict[str, Any],
        recent_episodes: list[dict[str, Any]] | None = None,
    ) -> ReflectionNote:
        recent_episodes = recent_episodes or []

        symbol = str(episode.get("symbol", "UNKNOWN"))
        timestamp = str(episode.get("timestamp", utc_now_iso()))

        execution = episode.get("execution_result") or {}
        final_decision = episode.get("final_decision") or {}
        risk_report = episode.get("risk_report") or {}
        proposal = episode.get("proposal") or {}

        status = str(execution.get("status", "unknown")).lower()
        action = str(final_decision.get("action", "hold")).lower()
        risk_score = float(risk_report.get("risk_score", 0.0) or 0.0)
        conflicting_count = len(proposal.get("conflicting_factors") or [])

        lesson = "Maintain process consistency and avoid impulsive overrides."
        mistake = None
        improvement = "Keep tracking outcome quality and adjust gradually."
        tags = ["process"]

        if status == "filled":
            lesson = "Execution succeeded under the full decision pipeline."
            tags.append("execution_success")
            if risk_score >= 0.55:
                lesson = "Trade executed in elevated-risk conditions; smaller size was appropriate."
                improvement = "Prioritize setups with cleaner confirmation in high-risk regimes."
                tags.append("elevated_risk")
            elif action in {"buy", "sell"} and conflicting_count > 0:
                improvement = "Reduce size earlier when signals conflict to lower variance."
                tags.append("signal_conflict")
        elif status in {"rejected", "failed"}:
            lesson = "Pipeline prevented unsafe execution."
            mistake = "Decision path reached an unexecutable state."
            improvement = "Tighten pre-trade checks so rejections happen earlier in the flow."
            tags.extend(["blocked_trade", "risk_control"])
        elif status == "skipped":
            lesson = "No trade was taken; preserving optionality can be correct."
            improvement = "Continue waiting for asymmetric setups with stronger agreement."
            tags.append("no_trade")

        if self._compute_recent_skip_ratio(recent_episodes) > 0.70:
            tags.append("high_skip_ratio")
            improvement = (
                f"{improvement} Review thresholds to confirm the system is not excessively conservative."
            )

        return ReflectionNote(
            symbol=symbol,
            timestamp=timestamp,
            lesson=lesson,
            mistake=mistake,
            improvement=improvement,
            tags=self._dedupe_preserve_order(tags),
            metadata={
                "status": status,
                "action": action,
                "risk_score": risk_score,
                "conflicting_factor_count": conflicting_count,
                "recent_episode_count": len(recent_episodes),
            },
        )

    def append_reflection(self, note: ReflectionNote | dict[str, Any]) -> None:
        payload = note.to_dict() if isinstance(note, ReflectionNote) else note
        os.makedirs(os.path.dirname(self.reflection_file), exist_ok=True)
        with open(self.reflection_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load_recent(self, limit: int = 50, symbol: str | None = None) -> list[dict]:
        if not os.path.exists(self.reflection_file):
            return []

        rows: list[dict] = []
        with open(self.reflection_file, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if symbol and item.get("symbol") != symbol:
                    continue
                rows.append(item)

        return rows[-limit:]

    def _compute_recent_skip_ratio(self, recent_episodes: list[dict]) -> float:
        if not recent_episodes:
            return 0.0

        count = 0
        skip_count = 0
        for episode in recent_episodes:
            status = str((episode.get("execution_result") or {}).get("status", "")).lower()
            if not status:
                continue
            count += 1
            if status in {"skipped", "rejected"}:
                skip_count += 1

        return (skip_count / count) if count > 0 else 0.0

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
