from __future__ import annotations

import math
from datetime import datetime, timezone


class MarketTools:
    def __init__(self, default_prices: dict[str, float] | None = None):
        self.default_prices = default_prices or {
            "BTC/USDT": 82000.0,
            "ETH/USDT": 4500.0,
            "SOL/USDT": 180.0,
        }

    def get_market_snapshot(self, symbol: str) -> dict:
        """
        Lightweight market data generator for local development.
        It keeps the interface stable so we can swap to live APIs later.
        """
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
        }
