from __future__ import annotations

import json
import os
from typing import Any

from trading_agent_v2.schemas import EpisodeRecord
from trading_agent_v2.memory.time_utils import local_day_bounds, parse_timestamp


class EpisodicMemoryStore:
    def __init__(self, memory_file: str):
        self.memory_file = memory_file

    def append_episode(self, episode: EpisodeRecord | dict[str, Any]) -> None:
        payload = episode.to_dict() if isinstance(episode, EpisodeRecord) else episode
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _load_all(self, symbol: str | None = None) -> list[dict]:
        if not os.path.exists(self.memory_file):
            return []

        episodes: list[dict] = []
        with open(self.memory_file, "r", encoding="utf-8") as f:
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
                episodes.append(item)
        return episodes

    def load_all(self, symbol: str | None = None) -> list[dict]:
        return self._load_all(symbol=symbol)

    def load_recent(self, limit: int = 50, symbol: str | None = None) -> list[dict]:
        episodes = self._load_all(symbol=symbol)
        return episodes[-limit:]

    def load_for_local_date(
        self,
        local_date: str,
        timezone_name: str = "America/Los_Angeles",
        symbol: str | None = None,
    ) -> list[dict]:
        start_utc, end_utc = local_day_bounds(local_date, timezone_name=timezone_name)
        output: list[dict] = []

        for episode in self._load_all(symbol=symbol):
            timestamp = parse_timestamp(episode.get("timestamp"))
            if timestamp is None:
                continue
            if start_utc <= timestamp < end_utc:
                output.append(episode)

        return output

    def load_latest_before_local_date(
        self,
        local_date: str,
        timezone_name: str = "America/Los_Angeles",
        symbol: str | None = None,
    ) -> dict[str, Any] | None:
        start_utc, _ = local_day_bounds(local_date, timezone_name=timezone_name)
        latest_episode: dict[str, Any] | None = None
        latest_ts = None

        for episode in self._load_all(symbol=symbol):
            timestamp = parse_timestamp(episode.get("timestamp"))
            if timestamp is None or timestamp >= start_utc:
                continue
            if latest_ts is None or timestamp > latest_ts:
                latest_ts = timestamp
                latest_episode = episode

        return latest_episode
