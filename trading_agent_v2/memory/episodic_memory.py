from __future__ import annotations

import json
import os
from typing import Any

from trading_agent_v2.schemas import EpisodeRecord


class EpisodicMemoryStore:
    def __init__(self, memory_file: str):
        self.memory_file = memory_file

    def append_episode(self, episode: EpisodeRecord | dict[str, Any]) -> None:
        payload = episode.to_dict() if isinstance(episode, EpisodeRecord) else episode
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load_recent(self, limit: int = 50, symbol: str | None = None) -> list[dict]:
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

        return episodes[-limit:]
