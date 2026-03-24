from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree as ET

import requests


class NewsTools:
    API_URL = "https://min-api.cryptocompare.com/data/v2/news/"
    POSITIVE_KEYWORDS = {
        "surge", "up", "rally", "approval", "breakout", "bull", "bullish", "inflow", "adoption",
        "partnership", "launch", "growth", "accumulation", "record", "gain",
    }
    NEGATIVE_KEYWORDS = {
        "drop", "down", "hack", "exploit", "lawsuit", "bear", "bearish", "outflow", "liquidation",
        "risk", "decline", "selloff", "ban", "fraud", "loss",
    }
    RSS_FALLBACK_URLS = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]

    def __init__(self, timeout_sec: int = 10):
        self.timeout_sec = timeout_sec
        self.api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
        self.session = requests.Session()

    def get_latest_news(self, symbol: str, limit: int = 3) -> list[dict]:
        """
        Fetch latest crypto news from CryptoCompare.
        """
        base_coin = symbol.split("/")[0].upper()
        params = {
            "lang": "EN",
            "categories": base_coin,
        }
        headers = {}
        if self.api_key:
            headers["authorization"] = f"Apikey {self.api_key}"

        try:
            response = self.session.get(
                self.API_URL,
                params=params,
                headers=headers,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("Response", "")).lower() == "error":
                raise RuntimeError(payload.get("Message", "CryptoCompare error response"))

            raw_items = payload.get("Data", []) or []
            rows: list[dict] = []

            for item in raw_items:
                title = str(item.get("title", "")).strip()
                body = str(item.get("body", "")).strip()
                text = f"{title} {body}".strip()
                if not text:
                    continue

                published_on = self._safe_int(item.get("published_on"), 0)
                published_at = (
                    datetime.fromtimestamp(published_on, tz=timezone.utc).replace(microsecond=0).isoformat()
                    if published_on > 0
                    else datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                )
                sentiment = self._classify_sentiment(text)

                rows.append(
                    {
                        "title": title or body[:120],
                        "summary": body[:280] if body else title,
                        "source": ((item.get("source_info") or {}).get("name") or "cryptocompare"),
                        "published_at": published_at,
                        "sentiment": sentiment,
                        "url": item.get("url"),
                        "categories": item.get("categories"),
                    }
                )

            scoped = [row for row in rows if base_coin in (row.get("title", "") + " " + row.get("summary", "")).upper()]
            selected = scoped if scoped else rows
            if selected:
                return selected[: max(1, limit)]
            return self._fetch_rss_fallback(base_coin=base_coin, limit=limit)
        except Exception:
            return self._fetch_rss_fallback(base_coin=base_coin, limit=limit)

    def summarize_sentiment(self, news_data: list[dict]) -> dict:
        if not news_data:
            return {
                "sentiment_score": 0.50,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "article_count": 0,
                "summary": "No fresh news found from upstream feeds.",
            }

        score_map = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
        scores = [score_map.get(str(item.get("sentiment", "neutral")).lower(), 0.5) for item in news_data]
        sentiment_score = sum(scores) / len(scores)

        positive_count = sum(1 for s in scores if s > 0.66)
        negative_count = sum(1 for s in scores if s < 0.34)
        total = len(scores)

        top_titles = [str(item.get("title", "")).strip() for item in news_data[:3] if item.get("title")]
        summary_text = "; ".join(top_titles) if top_titles else "News fetched but titles unavailable."

        return {
            "sentiment_score": round(sentiment_score, 4),
            "positive_ratio": round(positive_count / total, 4),
            "negative_ratio": round(negative_count / total, 4),
            "article_count": total,
            "summary": summary_text,
        }

    def _classify_sentiment(self, text: str) -> str:
        lowered = text.lower()
        pos_hits = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in lowered)
        neg_hits = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in lowered)
        if pos_hits > neg_hits:
            return "positive"
        if neg_hits > pos_hits:
            return "negative"
        return "neutral"

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _fetch_rss_fallback(self, base_coin: str, limit: int) -> list[dict]:
        rows: list[dict] = []
        for url in self.RSS_FALLBACK_URLS:
            try:
                response = self.session.get(url, timeout=self.timeout_sec)
                response.raise_for_status()
                rows.extend(self._parse_rss_items(response.text, source=url))
            except Exception:
                continue

        scoped = [
            row for row in rows
            if base_coin in (row.get("title", "") + " " + row.get("summary", "")).upper()
        ]
        selected = scoped if scoped else rows
        return selected[: max(1, limit)]

    def _parse_rss_items(self, xml_text: str, source: str) -> list[dict]:
        output: list[dict] = []
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return output

        # Standard RSS path
        items = root.findall(".//channel/item")
        if not items:
            # Atom fallback
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for item in items[:20]:
            title_node = self._first_present(
                item.find("title"),
                item.find("{http://www.w3.org/2005/Atom}title"),
            )
            desc_node = self._first_present(
                item.find("description"),
                item.find("{http://www.w3.org/2005/Atom}summary"),
            )
            pub_node = self._first_present(
                item.find("pubDate"),
                item.find("{http://www.w3.org/2005/Atom}updated"),
            )
            link_node = item.find("link")

            title = (title_node.text or "").strip() if title_node is not None else ""
            summary = (desc_node.text or "").strip() if desc_node is not None else title
            published_at = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                if pub_node is None or not pub_node.text
                else str(pub_node.text).strip()
            )
            url = None
            if link_node is not None:
                url = link_node.text or link_node.attrib.get("href")

            text = f"{title} {summary}".strip()
            if not text:
                continue

            output.append(
                {
                    "title": title or summary[:120],
                    "summary": summary[:280] if summary else title,
                    "source": source,
                    "published_at": published_at,
                    "sentiment": self._classify_sentiment(text),
                    "url": url,
                    "categories": None,
                }
            )
        return output

    def _first_present(self, *values):
        for value in values:
            if value is not None:
                return value
        return None
