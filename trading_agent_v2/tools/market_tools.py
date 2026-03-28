from __future__ import annotations

import math
from typing import Any

try:
    import ccxt
except Exception:  # pragma: no cover - optional dependency
    ccxt = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None
from datetime import datetime, timezone


class MarketTools:
    def __init__(
        self,
        exchange_id: str = "binanceus",
        fallback_exchange_id: str = "coinbase",
        timeframe: str = "1h",
        ohlcv_limit: int = 120,
        request_timeout_ms: int = 10000,
        default_prices: dict[str, float] | None = None,
    ):
        self.exchange = self._build_exchange(exchange_id, request_timeout_ms)
        self.fallback_exchange = self._build_exchange(fallback_exchange_id, request_timeout_ms)
        self.timeframe = timeframe
        self.ohlcv_limit = max(50, int(ohlcv_limit))
        self.default_prices = default_prices or {
            "BTC/USDT": 82000.0,
            "ETH/USDT": 4500.0,
            "SOL/USDT": 180.0,
        }

    def get_market_snapshot(self, symbol: str) -> dict:
        """
        Real market snapshot via exchanges and public market data APIs.
        Falls back to deterministic synthetic data when upstream is unavailable.
        """
        for exchange in (self.exchange, self.fallback_exchange):
            if exchange is None:
                continue
            snapshot = self._fetch_exchange_snapshot(exchange=exchange, symbol=symbol)
            if snapshot:
                return snapshot

        yf_snapshot = self._fetch_yfinance_snapshot(symbol)
        if yf_snapshot:
            return yf_snapshot

        return self._fallback_mock_snapshot(symbol)

    def scan_tradeable_candidates(
        self,
        quote_assets: list[str] | None = None,
        limit: int = 80,
    ) -> list[dict[str, Any]]:
        normalized_quotes = {
            str(item).upper().strip()
            for item in (quote_assets or ["USDT"])
            if str(item).strip()
        }
        max_candidates = max(1, int(limit))

        exchanges = [self.exchange]
        if self.exchange is None and self.fallback_exchange is not None:
            exchanges.append(self.fallback_exchange)

        for exchange in exchanges:
            if exchange is None:
                continue
            candidates = self._scan_exchange_candidates(
                exchange=exchange,
                quote_assets=normalized_quotes,
                limit=max_candidates,
            )
            if candidates:
                return candidates

        return self._fallback_candidate_scan(normalized_quotes, max_candidates)

    def _build_exchange(self, exchange_id: str, request_timeout_ms: int):
        if ccxt is None:
            return None
        try:
            exchange_cls = getattr(ccxt, exchange_id)
            return exchange_cls(
                {
                    "enableRateLimit": True,
                    "timeout": request_timeout_ms,
                }
            )
        except Exception:
            return None

    def _scan_exchange_candidates(
        self,
        exchange: Any,
        quote_assets: set[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            exchange.load_markets()
            tickers = exchange.fetch_tickers()
        except Exception:
            return []

        candidates: list[dict[str, Any]] = []
        for market_symbol, market in exchange.markets.items():
            if not isinstance(market, dict):
                continue
            if not market.get("spot", False):
                continue

            quote = str(market.get("quote", "")).upper().strip()
            if quote_assets and quote not in quote_assets:
                continue

            if market.get("active") is False:
                continue

            info = market.get("info") or {}
            state = str(info.get("state", "")).lower().strip()
            if state and state != "live":
                continue

            symbol = str(market.get("symbol") or market_symbol or "").strip()
            if not symbol or "/" not in symbol:
                continue

            ticker = tickers.get(market_symbol) or tickers.get(symbol)
            if not isinstance(ticker, dict):
                continue

            candidate = self._build_candidate_snapshot(
                exchange_id=getattr(exchange, "id", "unknown"),
                symbol=symbol,
                market=market,
                ticker=ticker,
            )
            if candidate is None:
                continue
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                float(item.get("scout_score", 0.0)),
                float(item.get("quote_volume_24h", 0.0)),
                float(item.get("range_pct_24h", 0.0)),
            ),
            reverse=True,
        )
        return candidates[:limit]

    def _build_candidate_snapshot(
        self,
        exchange_id: str,
        symbol: str,
        market: dict[str, Any],
        ticker: dict[str, Any],
    ) -> dict[str, Any] | None:
        info = ticker.get("info") or {}
        last_price = self._safe_float(ticker.get("last"), 0.0)
        bid_price = self._safe_float(ticker.get("bid"), 0.0)
        ask_price = self._safe_float(ticker.get("ask"), 0.0)
        if last_price <= 0:
            last_price = max(bid_price, ask_price)
        if last_price <= 0:
            return None

        open_price = self._safe_float(ticker.get("open"), 0.0)
        high_price = self._safe_float(ticker.get("high"), 0.0)
        low_price = self._safe_float(ticker.get("low"), 0.0)
        percentage_24h = self._safe_float(ticker.get("percentage"), 0.0)
        if abs(percentage_24h) <= 1e-9 and open_price > 0:
            percentage_24h = ((last_price - open_price) / open_price) * 100.0

        quote_volume_24h = self._safe_float(ticker.get("quoteVolume"), 0.0)
        if quote_volume_24h <= 0:
            quote_volume_24h = self._safe_float(info.get("volCcy24h"), 0.0)

        base_volume_24h = self._safe_float(ticker.get("baseVolume"), 0.0)
        if base_volume_24h <= 0:
            base_volume_24h = self._safe_float(info.get("vol24h"), 0.0)

        midpoint = (bid_price + ask_price) / 2.0 if bid_price > 0 and ask_price > 0 else last_price
        spread_pct = ((ask_price - bid_price) / midpoint) if midpoint > 0 and ask_price >= bid_price > 0 else 0.0
        range_pct_24h = ((high_price - low_price) / last_price) if high_price > 0 and low_price > 0 else abs(percentage_24h) / 100.0

        list_time_raw = (
            (market.get("info") or {}).get("listTime")
            or info.get("listTime")
            or ""
        )
        listed_at = self._parse_exchange_timestamp(list_time_raw)
        listing_age_hours = 99999.0
        if listed_at is not None:
            listing_age_hours = max(0.0, (datetime.now(timezone.utc) - listed_at).total_seconds() / 3600.0)

        movement_component = min(1.0, abs(percentage_24h) / 25.0 + range_pct_24h * 4.0)
        activity_component = min(1.0, math.log10(max(1.0, quote_volume_24h) + 1.0) / 8.0)
        tightness_component = max(0.0, 1.0 - min(1.0, spread_pct / 0.025))
        freshness_component = 0.15 if listing_age_hours <= 72 else 0.0
        scout_score = (
            movement_component * 0.46
            + activity_component * 0.36
            + tightness_component * 0.18
            + freshness_component
        )

        return {
            "symbol": symbol,
            "base": str(market.get("base", "")).upper().strip(),
            "quote": str(market.get("quote", "")).upper().strip(),
            "last_price": round(last_price, 10),
            "bid_price": round(bid_price, 10),
            "ask_price": round(ask_price, 10),
            "spread_pct": round(spread_pct, 6),
            "pct_change_24h": round(percentage_24h, 4),
            "range_pct_24h": round(range_pct_24h, 6),
            "quote_volume_24h": round(quote_volume_24h, 4),
            "base_volume_24h": round(base_volume_24h, 4),
            "listing_age_hours": round(listing_age_hours, 2),
            "is_recent_listing": bool(listing_age_hours <= 72),
            "state": str((market.get("info") or {}).get("state", "live") or "live").lower(),
            "market_source": f"ccxt:{exchange_id}",
            "scout_score": round(scout_score, 6),
        }

    def _fallback_candidate_scan(self, quote_assets: set[str], limit: int) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for symbol, price in self.default_prices.items():
            base, quote = symbol.split("/")
            if quote_assets and quote.upper() not in quote_assets:
                continue
            output.append(
                {
                    "symbol": symbol,
                    "base": base.upper(),
                    "quote": quote.upper(),
                    "last_price": round(float(price), 10),
                    "bid_price": round(float(price) * 0.999, 10),
                    "ask_price": round(float(price) * 1.001, 10),
                    "spread_pct": 0.002,
                    "pct_change_24h": 0.0,
                    "range_pct_24h": 0.01,
                    "quote_volume_24h": 0.0,
                    "base_volume_24h": 0.0,
                    "listing_age_hours": 99999.0,
                    "is_recent_listing": False,
                    "state": "live",
                    "market_source": "fallback:default_prices",
                    "scout_score": 0.1,
                }
            )
        return output[:limit]

    def _parse_exchange_timestamp(self, value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            millis = int(float(text))
            return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return None

    def _fetch_exchange_snapshot(self, exchange: Any, symbol: str) -> dict | None:
        try:
            exchange.load_markets()
            if symbol not in exchange.markets:
                return None

            ticker = exchange.fetch_ticker(symbol)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.ohlcv_limit)
            if not ohlcv:
                return None

            closes = [float(c[4]) for c in ohlcv]
            price = self._safe_float(ticker.get("last"), closes[-1])
            open_price = self._safe_float(ticker.get("open"), float(ohlcv[-1][1]))
            high_price = self._safe_float(ticker.get("high"), float(ohlcv[-1][2]))
            low_price = self._safe_float(ticker.get("low"), float(ohlcv[-1][3]))
            volume = self._safe_float(
                ticker.get("baseVolume"),
                float(ohlcv[-1][5]),
            )

            rsi = self._compute_rsi(closes, period=14)
            atr_pct = self._compute_atr_pct(ohlcv, period=14)
            ema_fast = self._compute_ema(closes, period=9)
            ema_slow = self._compute_ema(closes, period=21)

            return {
                "price": round(price, 6),
                "open": round(open_price, 6),
                "high": round(high_price, 6),
                "low": round(max(0.000001, low_price), 6),
                "volume": round(volume, 2),
                "rsi": round(rsi, 2),
                "atr_pct": round(atr_pct, 4),
                "ema_fast_above_slow": bool(ema_fast >= ema_slow),
                "source": f"ccxt:{exchange.id}",
                "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        except Exception:
            return None

    def _fetch_yfinance_snapshot(self, symbol: str) -> dict | None:
        if yf is None:
            return None
        try:
            yf_symbol = self._to_yfinance_symbol(symbol)
            history = yf.Ticker(yf_symbol).history(period="7d", interval="1h")
            if history is None or history.empty:
                return None

            closes = [float(v) for v in history["Close"].dropna().tolist()]
            highs = [float(v) for v in history["High"].dropna().tolist()]
            lows = [float(v) for v in history["Low"].dropna().tolist()]
            opens = [float(v) for v in history["Open"].dropna().tolist()]
            volumes = [float(v) for v in history["Volume"].fillna(0).tolist()]
            if not closes:
                return None

            ohlcv = []
            min_len = min(len(opens), len(highs), len(lows), len(closes), len(volumes))
            for idx in range(min_len):
                ohlcv.append([0, opens[idx], highs[idx], lows[idx], closes[idx], volumes[idx]])

            rsi = self._compute_rsi(closes, period=14)
            atr_pct = self._compute_atr_pct(ohlcv, period=14)
            ema_fast = self._compute_ema(closes, period=9)
            ema_slow = self._compute_ema(closes, period=21)

            return {
                "price": round(closes[-1], 6),
                "open": round(opens[-1], 6),
                "high": round(highs[-1], 6),
                "low": round(max(0.000001, lows[-1]), 6),
                "volume": round(volumes[-1], 2),
                "rsi": round(rsi, 2),
                "atr_pct": round(atr_pct, 4),
                "ema_fast_above_slow": bool(ema_fast >= ema_slow),
                "source": "yfinance",
                "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        except Exception:
            return None

    def _fallback_mock_snapshot(self, symbol: str) -> dict:
        now = datetime.now(timezone.utc)
        minute_phase = (now.minute + now.second / 60.0) / 60.0
        phase = minute_phase * 2 * math.pi

        base_price = float(self.default_prices.get(symbol, 100.0))
        wave = math.sin(phase)
        trend_wave = math.cos(phase * 0.5)

        price = base_price * (1.0 + 0.004 * wave + 0.0015 * trend_wave)
        open_price = base_price * (1.0 - 0.0010 * trend_wave)
        day_range = base_price * (0.010 + abs(wave) * 0.015)

        high_price = price + day_range * 0.50
        low_price = max(0.000001, price - day_range * 0.50)
        volume = 100000 + 45000 * abs(wave)

        rsi = max(20.0, min(80.0, 50.0 + 18.0 * wave))
        atr_pct = 0.015 + abs(wave) * 0.020

        return {
            "price": round(price, 6),
            "open": round(open_price, 6),
            "high": round(high_price, 6),
            "low": round(low_price, 6),
            "volume": round(volume, 2),
            "rsi": round(rsi, 2),
            "atr_pct": round(atr_pct, 4),
            "ema_fast_above_slow": wave >= -0.10,
            "source": "fallback:synthetic",
            "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

    def _to_yfinance_symbol(self, symbol: str) -> str:
        base, quote = symbol.split("/")
        if quote.upper() == "USDT":
            quote = "USD"
        return f"{base.upper()}-{quote.upper()}"

    def _compute_rsi(self, closes: list[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        gains = []
        losses = []
        for i in range(-period, 0):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_atr_pct(self, ohlcv: list[list[float]], period: int = 14) -> float:
        if len(ohlcv) < period + 1:
            return 0.02
        trs = []
        for i in range(-period, 0):
            high = float(ohlcv[i][2])
            low = float(ohlcv[i][3])
            prev_close = float(ohlcv[i - 1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        atr = sum(trs) / period
        last_close = float(ohlcv[-1][4])
        if last_close <= 0:
            return 0.02
        return atr / last_close

    def _compute_ema(self, values: list[float], period: int) -> float:
        if not values:
            return 0.0
        alpha = 2 / (period + 1)
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
