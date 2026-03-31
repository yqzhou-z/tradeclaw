from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests


class OnchainTools:
    """
    Public-data on-chain proxy signals.
    Uses CoinGecko market endpoints as a no-key, real-data baseline.
    """

    COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
    COINGECKO_MARKET_RANGE_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    COIN_ID_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
    }

    def __init__(self, timeout_sec: int = 10):
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self._historical_range_cache: dict[tuple[str, int, int], dict[str, list[list[float]]]] = {}
        self._historical_stablecoin_cache: dict[tuple[int, int], dict[str, dict[str, list[list[float]]]]] = {}

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

    def preload_historical_onchain_series(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> dict[str, Any]:
        base_coin = symbol.split("/")[0].upper()
        coin_id = self.COIN_ID_MAP.get(base_coin, base_coin.lower())
        asset_series = self._fetch_market_chart_range(coin_id, start_ms, end_ms)
        stablecoin_series = self._fetch_stablecoin_market_caps(start_ms, end_ms)
        return {
            "coin_id": coin_id,
            "asset_series": asset_series,
            "stablecoin_series": stablecoin_series,
        }

    def get_historical_onchain_snapshot(
        self,
        symbol: str,
        as_of: str | datetime,
        *,
        lookback_hours: int = 48,
        preloaded_series: dict[str, Any] | None = None,
    ) -> dict:
        cutoff_dt = self._parse_datetime(as_of)
        if cutoff_dt is None:
            return self.get_onchain_snapshot(symbol)

        cutoff_ms = int(cutoff_dt.timestamp() * 1000)
        start_dt = cutoff_dt - timedelta(hours=max(24, lookback_hours))
        start_ms = int(start_dt.timestamp() * 1000)
        base_coin = symbol.split("/")[0].upper()

        if preloaded_series:
            asset_series = dict(preloaded_series.get("asset_series") or {})
            stablecoin_series = dict(preloaded_series.get("stablecoin_series") or {})
        else:
            preloaded = self.preload_historical_onchain_series(symbol, start_ms, cutoff_ms)
            asset_series = dict(preloaded.get("asset_series") or {})
            stablecoin_series = dict(preloaded.get("stablecoin_series") or {})

        if asset_series:
            return self._build_historical_snapshot_from_series(
                base_coin=base_coin,
                cutoff_ms=cutoff_ms,
                asset_series=asset_series,
                stablecoin_series=stablecoin_series,
            )

        return {
            "asset": base_coin,
            "exchange_outflow": "moderate",
            "whale_activity": "neutral",
            "stablecoin_flow": "neutral",
            "active_addresses_change_pct": 0.0,
            "source": "fallback:onchain_historical",
            "fetched_at": cutoff_dt.replace(microsecond=0).isoformat(),
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

    def _fetch_market_chart_range(
        self,
        coin_id: str,
        start_ms: int,
        end_ms: int,
    ) -> dict[str, list[list[float]]]:
        cache_key = (coin_id, int(start_ms), int(end_ms))
        cached = self._historical_range_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            response = self.session.get(
                self.COINGECKO_MARKET_RANGE_URL.format(coin_id=coin_id),
                params={
                    "vs_currency": "usd",
                    "from": max(0, int(start_ms // 1000)),
                    "to": max(0, int(end_ms // 1000)),
                },
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            payload = {}

        normalized = {
            "prices": self._normalize_series(payload.get("prices")),
            "market_caps": self._normalize_series(payload.get("market_caps")),
            "total_volumes": self._normalize_series(payload.get("total_volumes")),
        }
        self._historical_range_cache[cache_key] = normalized
        return normalized

    def _fetch_stablecoin_market_caps(
        self,
        start_ms: int,
        end_ms: int,
    ) -> dict[str, dict[str, list[list[float]]]]:
        cache_key = (int(start_ms), int(end_ms))
        cached = self._historical_stablecoin_cache.get(cache_key)
        if cached is not None:
            return cached

        output: dict[str, dict[str, list[list[float]]]] = {}
        for coin_id in ("tether", "usd-coin", "dai"):
            output[coin_id] = self._fetch_market_chart_range(coin_id, start_ms, end_ms)
        self._historical_stablecoin_cache[cache_key] = output
        return output

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

    def _build_historical_snapshot_from_series(
        self,
        base_coin: str,
        cutoff_ms: int,
        asset_series: dict[str, list[list[float]]],
        stablecoin_series: dict[str, dict[str, list[list[float]]]],
    ) -> dict:
        current_price = self._latest_value_before(asset_series.get("prices", []), cutoff_ms)
        current_market_cap = self._latest_value_before(asset_series.get("market_caps", []), cutoff_ms)
        current_volume = self._latest_value_before(asset_series.get("total_volumes", []), cutoff_ms)
        prev_cutoff_ms = cutoff_ms - 24 * 3600 * 1000
        previous_price = self._latest_value_before(asset_series.get("prices", []), prev_cutoff_ms)
        previous_market_cap = self._latest_value_before(asset_series.get("market_caps", []), prev_cutoff_ms)

        price_change_24h = self._pct_change(current_price, previous_price)
        market_cap_change_24h = self._pct_change(current_market_cap, previous_market_cap)
        turnover = (current_volume / current_market_cap) if current_market_cap > 0 else 0.0

        avg_stablecoin_change = 0.0
        stablecoin_changes: list[float] = []
        for payload in stablecoin_series.values():
            now_cap = self._latest_value_before((payload or {}).get("market_caps", []), cutoff_ms)
            prev_cap = self._latest_value_before((payload or {}).get("market_caps", []), prev_cutoff_ms)
            if now_cap > 0 and prev_cap > 0:
                stablecoin_changes.append(self._pct_change(now_cap, prev_cap))
        if stablecoin_changes:
            avg_stablecoin_change = sum(stablecoin_changes) / len(stablecoin_changes)

        if avg_stablecoin_change >= 0.12:
            stablecoin_flow = "inflow"
        elif avg_stablecoin_change <= -0.12:
            stablecoin_flow = "outflow"
        else:
            stablecoin_flow = "neutral"

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

        active_addresses_change_pct = max(-25.0, min(25.0, market_cap_change_24h * 1.2))

        return {
            "asset": base_coin,
            "exchange_outflow": exchange_outflow,
            "whale_activity": whale_activity,
            "stablecoin_flow": stablecoin_flow,
            "active_addresses_change_pct": round(active_addresses_change_pct, 2),
            "source": "coingecko_history_range",
            "turnover_ratio": round(turnover, 6),
            "price_change_24h_pct": round(price_change_24h, 4),
            "market_cap_change_24h_pct": round(market_cap_change_24h, 4),
            "fetched_at": datetime.fromtimestamp(cutoff_ms / 1000.0, tz=timezone.utc).replace(microsecond=0).isoformat(),
        }

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

    def _normalize_series(self, rows: Any) -> list[list[float]]:
        output: list[list[float]] = []
        for row in rows or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            ts = self._safe_float(row[0], -1.0)
            value = self._safe_float(row[1], 0.0)
            if ts < 0:
                continue
            output.append([int(ts), value])
        output.sort(key=lambda item: item[0])
        return output

    def _latest_value_before(self, rows: list[list[float]], cutoff_ms: int) -> float:
        latest_value = 0.0
        for row in rows or []:
            try:
                ts = int(row[0])
            except (TypeError, ValueError, IndexError):
                continue
            if ts > cutoff_ms:
                break
            latest_value = self._safe_float(row[1], latest_value)
        return latest_value

    def _pct_change(self, current: float, previous: float) -> float:
        if previous <= 0:
            return 0.0
        return ((current - previous) / previous) * 100.0

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
