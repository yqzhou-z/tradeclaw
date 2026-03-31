from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from trading_agent_v2.schemas import ExecutionResult, FinalDecision, PortfolioState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PaperExecutor:
    def __init__(
        self,
        trading_fee_rate: float = 0.0,
        slippage_rate: float = 0.0,
    ):
        """
        Parameters
        ----------
        trading_fee_rate : float
            Example: 0.001 means 0.1% fee.
        slippage_rate : float
            Example: 0.0005 means 0.05% slippage.
        """
        self.trading_fee_rate = trading_fee_rate
        self.slippage_rate = slippage_rate

    # =========================================================
    # Public API
    # =========================================================

    def execute(
        self,
        decision: FinalDecision,
        portfolio: PortfolioState,
        market_prices: Optional[Dict[str, float]] = None,
    ) -> ExecutionResult:
        """
        Simulate execution for a paper trade.

        Assumptions in v1:
        - buy: size_pct means % of available cash
        - sell: size_pct means % of current position
        - order_type is treated as market order
        """
        action = (decision.action or "").lower().strip()
        symbol = decision.symbol
        timestamp = decision.timestamp or utc_now_iso()

        if action == "hold":
            return ExecutionResult(
                symbol=symbol,
                timestamp=timestamp,
                status="skipped",
                action="hold",
                message="Hold decision: no execution performed.",
            )

        market_price = self._get_market_price(symbol, market_prices)
        if market_price is None or market_price <= 0:
            return ExecutionResult(
                symbol=symbol,
                timestamp=timestamp,
                status="failed",
                action=action,
                message=f"Missing valid market price for {symbol}.",
            )

        try:
            execution_price = self._apply_slippage(
                action=action,
                market_price=market_price,
            )

            if action == "buy":
                notional_value, filled_qty, fees = self._execute_buy(
                    decision=decision,
                    portfolio=portfolio,
                    execution_price=execution_price,
                )
            elif action == "sell":
                notional_value, filled_qty, fees = self._execute_sell(
                    decision=decision,
                    portfolio=portfolio,
                    execution_price=execution_price,
                )
            else:
                return ExecutionResult(
                    symbol=symbol,
                    timestamp=timestamp,
                    status="failed",
                    action=action,
                    message=f"Unsupported action: {decision.action}",
                )

            if filled_qty <= 0 or notional_value <= 0:
                return ExecutionResult(
                    symbol=symbol,
                    timestamp=timestamp,
                    status="failed",
                    action=action,
                    message="Computed quantity or notional is non-positive.",
                )

            return ExecutionResult(
                symbol=symbol,
                timestamp=timestamp,
                status="filled",
                action=action,
                filled_price=execution_price,
                filled_qty=filled_qty,
                notional_value=notional_value,
                fees=fees,
                message="Paper order executed successfully.",
                metadata={
                    "market_price": market_price,
                    "slippage_rate": self.slippage_rate,
                    "trading_fee_rate": self.trading_fee_rate,
                    "order_type": decision.order_type,
                    "reason": decision.reason,
                },
            )

        except Exception as e:
            return ExecutionResult(
                symbol=symbol,
                timestamp=timestamp,
                status="failed",
                action=action,
                message=f"Paper execution failed: {str(e)}",
            )

    # =========================================================
    # Buy / sell calculation
    # =========================================================

    def _execute_buy(
        self,
        decision: FinalDecision,
        portfolio: PortfolioState,
        execution_price: float,
    ) -> tuple[float, float, float]:
        """
        Buy amount is based on available cash.
        size_pct = fraction of current cash to deploy.
        """
        cash = float(portfolio.cash)
        size_pct = float(decision.size_pct or 0.0)

        gross_notional = cash * size_pct
        if gross_notional <= 0:
            raise ValueError("Buy gross_notional must be positive.")

        fees = gross_notional * self.trading_fee_rate
        net_notional_for_asset = gross_notional - fees

        if net_notional_for_asset <= 0:
            raise ValueError("Net notional after fees must be positive.")

        filled_qty = net_notional_for_asset / execution_price

        if gross_notional > cash + 1e-12:
            raise ValueError(
                f"Insufficient cash for buy. cash={cash}, required={gross_notional}"
            )

        return gross_notional, filled_qty, fees

    def _execute_sell(
        self,
        decision: FinalDecision,
        portfolio: PortfolioState,
        execution_price: float,
    ) -> tuple[float, float, float]:
        """
        Sell amount is based on current position size.
        size_pct = fraction of current position to sell.
        """
        symbol = decision.symbol
        size_pct = float(decision.size_pct or 0.0)

        position = portfolio.positions.get(symbol)
        if not position:
            raise ValueError(f"No position found for symbol {symbol}.")

        current_qty = float(position.get("quantity", 0.0))
        if current_qty <= 0:
            raise ValueError(f"Current position quantity is zero for {symbol}.")

        filled_qty = current_qty * size_pct
        if filled_qty <= 0:
            raise ValueError("Sell quantity must be positive.")

        if filled_qty > current_qty + 1e-12:
            raise ValueError(
                f"Sell quantity exceeds current holding. have={current_qty}, sell={filled_qty}"
            )

        gross_notional = filled_qty * execution_price
        fees = gross_notional * self.trading_fee_rate
        net_notional = gross_notional - fees

        if net_notional <= 0:
            raise ValueError("Net proceeds after fees must be positive.")

        return net_notional, filled_qty, fees

    # =========================================================
    # Helpers
    # =========================================================

    def _get_market_price(
        self,
        symbol: str,
        market_prices: Optional[Dict[str, float]],
    ) -> Optional[float]:
        if not market_prices:
            return None
        price = market_prices.get(symbol)
        if price is None:
            return None
        return float(price)

    def _apply_slippage(
        self,
        action: str,
        market_price: float,
    ) -> float:
        """
        Simple slippage model:
        - buy pays slightly more
        - sell receives slightly less
        """
        if action == "buy":
            return market_price * (1.0 + self.slippage_rate)
        elif action == "sell":
            return market_price * (1.0 - self.slippage_rate)
        return market_price
