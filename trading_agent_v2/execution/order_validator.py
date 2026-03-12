from __future__ import annotations

from typing import Dict, Optional

from trading_agent_v2.schemas import FinalDecision, PortfolioState, ValidationResult


class OrderValidator:
    def __init__(
        self,
        min_size_pct: float = 0.0,
        max_size_pct: float = 1.0,
        allow_short: bool = False,
    ):
        self.min_size_pct = min_size_pct
        self.max_size_pct = max_size_pct
        self.allow_short = allow_short

    # =========================================================
    # Public API
    # =========================================================

    def validate(
        self,
        decision: FinalDecision,
        portfolio: PortfolioState,
        market_prices: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """
        Validate whether a final decision can be executed safely.

        Parameters
        ----------
        decision : FinalDecision
            Final decision from trader_agent.
        portfolio : PortfolioState
            Current portfolio state.
        market_prices : dict[str, float] | None
            Latest market prices keyed by symbol, e.g. {"BTC/USDT": 82000.0}

        Returns
        -------
        ValidationResult
        """
        errors = []
        warnings = []

        action = (decision.action or "").lower().strip()
        symbol = decision.symbol
        size_pct = float(decision.size_pct or 0.0)

        # 1. action legality
        if action not in {"buy", "sell", "hold"}:
            errors.append(f"Unsupported action: {decision.action}")

        # 2. symbol legality
        if not symbol or not isinstance(symbol, str):
            errors.append("Missing or invalid symbol.")

        # 3. hold branch
        if action == "hold":
            if size_pct != 0.0:
                warnings.append("Hold decision has non-zero size_pct; it will be ignored.")
            return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

        # 4. size_pct legality
        if size_pct <= 0:
            errors.append("size_pct must be > 0 for buy/sell decisions.")

        if size_pct < self.min_size_pct:
            errors.append(
                f"size_pct {size_pct:.6f} is below min_size_pct {self.min_size_pct:.6f}."
            )

        if size_pct > self.max_size_pct:
            errors.append(
                f"size_pct {size_pct:.6f} exceeds max_size_pct {self.max_size_pct:.6f}."
            )

        # 5. market price required for buy/sell validation
        market_price = self._get_market_price(symbol, market_prices)
        if market_price is None or market_price <= 0:
            errors.append(f"Missing valid market price for symbol: {symbol}")

        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        # 6. business validation by action
        if action == "buy":
            buy_errors, buy_warnings = self._validate_buy(
                decision=decision,
                portfolio=portfolio,
                market_price=market_price,
            )
            errors.extend(buy_errors)
            warnings.extend(buy_warnings)

        elif action == "sell":
            sell_errors, sell_warnings = self._validate_sell(
                decision=decision,
                portfolio=portfolio,
                market_price=market_price,
            )
            errors.extend(sell_errors)
            warnings.extend(sell_warnings)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    # =========================================================
    # Internal helpers
    # =========================================================

    def _validate_buy(
        self,
        decision: FinalDecision,
        portfolio: PortfolioState,
        market_price: float,
    ) -> tuple[list[str], list[str]]:
        errors = []
        warnings = []

        cash = float(portfolio.cash)
        size_pct = float(decision.size_pct)
        notional_value = cash * size_pct
        estimated_qty = notional_value / market_price if market_price > 0 else 0.0

        if cash <= 0:
            errors.append("No available cash to buy.")

        if notional_value <= 0:
            errors.append("Computed buy notional_value <= 0.")

        if estimated_qty <= 0:
            errors.append("Computed buy quantity <= 0.")

        if notional_value > cash:
            errors.append(
                f"Buy notional_value {notional_value:.6f} exceeds available cash {cash:.6f}."
            )

        # optional warning for very high concentration
        projected_symbol_value = self._get_position_market_value(portfolio, decision.symbol)
        projected_symbol_value += notional_value
        projected_equity = float(portfolio.total_equity) if portfolio.total_equity > 0 else cash
        if projected_equity > 0:
            projected_exposure = projected_symbol_value / projected_equity
            if projected_exposure > 0.5:
                warnings.append(
                    f"Projected exposure for {decision.symbol} may exceed 50% "
                    f"of total equity ({projected_exposure:.2%})."
                )

        return errors, warnings

    def _validate_sell(
        self,
        decision: FinalDecision,
        portfolio: PortfolioState,
        market_price: float,
    ) -> tuple[list[str], list[str]]:
        errors = []
        warnings = []

        symbol = decision.symbol
        size_pct = float(decision.size_pct)

        position = portfolio.positions.get(symbol)
        current_qty = float(position.get("quantity", 0.0)) if position else 0.0
        current_market_value = (
            float(position.get("market_value", 0.0)) if position else current_qty * market_price
        )

        if current_qty <= 0:
            errors.append(f"No existing position to sell for {symbol}.")
            return errors, warnings

        target_notional = current_market_value * size_pct
        estimated_qty = target_notional / market_price if market_price > 0 else 0.0

        if target_notional <= 0:
            errors.append("Computed sell notional_value <= 0.")

        if estimated_qty <= 0:
            errors.append("Computed sell quantity <= 0.")

        if not self.allow_short and estimated_qty > current_qty + 1e-12:
            errors.append(
                f"Sell quantity {estimated_qty:.12f} exceeds current position "
                f"quantity {current_qty:.12f}."
            )

        remaining_qty = current_qty - estimated_qty
        if remaining_qty < 0 and not self.allow_short:
            errors.append("Sell would result in negative quantity.")

        if size_pct > 1.0 and not self.allow_short:
            errors.append("size_pct > 1.0 is not allowed for sell without shorting enabled.")

        if 0 < remaining_qty < 1e-8:
            warnings.append(
                "Remaining quantity after sell would be extremely small; "
                "consider selling the full position."
            )

        return errors, warnings

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

    def _get_position_market_value(
        self,
        portfolio: PortfolioState,
        symbol: str,
    ) -> float:
        pos = portfolio.positions.get(symbol)
        if not pos:
            return 0.0
        return float(pos.get("market_value", 0.0))