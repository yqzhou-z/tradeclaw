from __future__ import annotations

from dataclasses import dataclass

from trading_agent_v2.schemas import AnalystView, RawContext


@dataclass
class MarketAnalystConfig:
    bullish_rsi_floor: float = 50.0
    bullish_rsi_ceiling: float = 70.0
    bearish_rsi_ceiling: float = 44.0
    high_volatility_atr_pct: float = 0.03


class MarketAnalyst:
    def __init__(self, config: MarketAnalystConfig | None = None):
        self.config = config or MarketAnalystConfig()

    def analyze(self, raw_context: RawContext) -> AnalystView:
        market = raw_context.market_data or {}
        onchain = raw_context.onchain_data or {}

        price = float(market.get("price", 0.0))
        rsi = float(market.get("rsi", 50.0))
        atr_pct = float(market.get("atr_pct", 0.0))
        trend_ok = bool(market.get("ema_fast_above_slow", False))

        exchange_outflow = str(onchain.get("exchange_outflow", "moderate")).lower()
        whale_activity = str(onchain.get("whale_activity", "neutral")).lower()

        confidence = 0.50
        bias = "neutral"
        summary = "Market structure and momentum are mixed."
        supporting_signals: list[str] = []
        risk_flags: list[str] = []

        if trend_ok and self.config.bullish_rsi_floor < rsi < self.config.bullish_rsi_ceiling:
            bias = "bullish"
            confidence += 0.18
            supporting_signals.extend(["trend_alignment", "rsi_supportive"])

        if (not trend_ok) and rsi <= self.config.bearish_rsi_ceiling:
            bias = "bearish"
            confidence += 0.15
            supporting_signals.extend(["trend_breakdown", "rsi_weak"])

        if exchange_outflow in {"strong", "moderate"} and whale_activity == "accumulation":
            if bias == "bearish":
                confidence -= 0.08
            else:
                bias = "bullish"
                confidence += 0.08
            supporting_signals.append("onchain_accumulation")

        if whale_activity == "distribution":
            risk_flags.append("whale_distribution")
            if bias == "bullish":
                confidence -= 0.10

        if atr_pct >= self.config.high_volatility_atr_pct:
            risk_flags.append("high_volatility")
            confidence -= 0.08

        confidence = max(0.0, min(1.0, confidence))

        if bias == "bullish":
            summary = "Trend and momentum are supportive with acceptable structure."
        elif bias == "bearish":
            summary = "Momentum and structure indicate downside pressure."

        return AnalystView(
            analyst_name="market_analyst",
            symbol=raw_context.symbol,
            timestamp=raw_context.timestamp,
            bias=bias,
            confidence=confidence,
            summary=summary,
            supporting_signals=self._dedupe_preserve_order(supporting_signals),
            risk_flags=self._dedupe_preserve_order(risk_flags),
            details={
                "price": price,
                "rsi": rsi,
                "atr_pct": atr_pct,
                "ema_fast_above_slow": trend_ok,
                "exchange_outflow": exchange_outflow,
                "whale_activity": whale_activity,
            },
        )

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
