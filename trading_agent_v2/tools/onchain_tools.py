from __future__ import annotations

from datetime import datetime, timezone


class OnchainTools:
    def get_onchain_snapshot(self, symbol: str) -> dict:
        """
        Mock on-chain signal block.
        """
        now = datetime.now(timezone.utc)
        base_coin = symbol.split("/")[0].upper()

        regime = (now.minute // 10) % 3
        if regime == 0:
            exchange_outflow = "strong"
            whale_activity = "accumulation"
            stablecoin_flow = "inflow"
        elif regime == 1:
            exchange_outflow = "moderate"
            whale_activity = "neutral"
            stablecoin_flow = "neutral"
        else:
            exchange_outflow = "weak"
            whale_activity = "distribution"
            stablecoin_flow = "outflow"

        return {
            "asset": base_coin,
            "exchange_outflow": exchange_outflow,
            "whale_activity": whale_activity,
            "stablecoin_flow": stablecoin_flow,
            "active_addresses_change_pct": round(((now.minute % 12) - 6) * 0.6, 2),
        }
