from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests


class OnchainTools:
    """
    Public-data on-chain proxy signals.
    Uses CoinGecko market endpoints as a no-key, real-data baseline.
    """

    COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
    COIN_ID_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
    }

    def __init__(self, timeout_sec: int = 10):
        self.timeout_sec = timeout_sec
        self.session = requests.Session()

    def get_onchain_snapshot(self, symbol: str) -> dict:
        base_coin = symbol.split("/")[0].upper()
        coin_id = self.COIN_ID_MAP.get(base_coin, base_coin.lower())

        asset_market = self._fetch_coin_market(coin_id)
        stablecoin_flow = self._infer_stablecoin_flow()

        if asset_market:
            return self._build_snapshot_from_market(
                base_coin=base_coin,
                stablecoin_flow=stablecoin_flow,
                asset_market=asset_market,
            )

        return {
            "asset": base_coin,
            "exchange_outflow": "moderate",
            "whale_activity": "neutral",
            "stablecoin_flow": stablecoin_flow,
            "active_addresses_change_pct": 0.0,
            "source": "fallback:onchain_proxy",
            "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

    def _fetch_coin_market(self, coin_id: str) -> dict[str, Any] | None:
        try:
            response = self.session.get(
                self.COINGECKO_MARKETS_URL,
                params={
                    "vs_currency": "usd",
                    "ids": coin_id,
                    "price_change_percentage": "24h",
                },
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload:
                return None
            if not isinstance(payload, list):
                return None
            if not payload[0]:
                return None
            return payload[0]
        except Exception:
            return None

    def _infer_stablecoin_flow(self) -> str:
        try:
            response = self.session.get(
                self.COINGECKO_MARKETS_URL,
                params={
                    "vs_currency": "usd",
                    "ids": "tether,usd-coin,dai",
                    "price_change_percentage": "24h",
                },
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            items = response.json()
            if not items or not isinstance(items, list):
                return "neutral"

            changes = []
            for row in items:
                changes.append(self._safe_float(row.get("market_cap_change_percentage_24h"), 0.0))
            avg_change = sum(changes) / len(changes) if changes else 0.0

            if avg_change >= 0.12:
                return "inflow"
            if avg_change <= -0.12:
                return "outflow"
            return "neutral"
        except Exception:
            return "neutral"

    def _build_snapshot_from_market(
        self,
        base_coin: str,
        stablecoin_flow: str,
        asset_market: dict[str, Any],
    ) -> dict:
        volume = self._safe_float(asset_market.get("total_volume"), 0.0)
        market_cap = self._safe_float(asset_market.get("market_cap"), 0.0)
        price_change_24h = self._safe_float(asset_market.get("price_change_percentage_24h"), 0.0)
        market_cap_change_24h = self._safe_float(asset_market.get("market_cap_change_percentage_24h"), 0.0)

        turnover = (volume / market_cap) if market_cap > 0 else 0.0
        if turnover < 0.04:
            exchange_outflow = "strong"
        elif turnover < 0.10:
            exchange_outflow = "moderate"
        else:
            exchange_outflow = "weak"

        if price_change_24h > 1.0 and turnover <= 0.12:
            whale_activity = "accumulation"
        elif price_change_24h < -1.0 and turnover >= 0.08:
            whale_activity = "distribution"
        else:
            whale_activity = "neutral"

        active_addresses_change_pct = max(
            -25.0,
            min(25.0, market_cap_change_24h * 1.2),
        )

        return {
            "asset": base_coin,
            "exchange_outflow": exchange_outflow,
            "whale_activity": whale_activity,
            "stablecoin_flow": stablecoin_flow,
            "active_addresses_change_pct": round(active_addresses_change_pct, 2),
            "source": "coingecko_proxy",
            "turnover_ratio": round(turnover, 6),
            "price_change_24h_pct": round(price_change_24h, 4),
            "market_cap_change_24h_pct": round(market_cap_change_24h, 4),
            "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

