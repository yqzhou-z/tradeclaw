from __future__ import annotations

import json
import os
from typing import Any


class DailyReviewStore:
    def __init__(self, review_file: str):
        self.review_file = review_file

    def append_review(self, review: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.review_file), exist_ok=True)
        with open(self.review_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(review, ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.review_file):
            return []

        reviews: list[dict[str, Any]] = []
        with open(self.review_file, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    reviews.append(item)
        return reviews

    def load_recent(
        self,
        limit: int = 5,
        dedupe_by_date: bool = True,
    ) -> list[dict[str, Any]]:
        reviews = self.load_all()
        if not dedupe_by_date:
            return reviews[-limit:]

        output: list[dict[str, Any]] = []
        seen_dates: set[str] = set()
        for review in reversed(reviews):
            review_date = str(review.get("review_date", "")).strip()
            if not review_date or review_date in seen_dates:
                continue
            seen_dates.add(review_date)
            output.append(review)
            if len(output) >= limit:
                break
        return list(reversed(output))
