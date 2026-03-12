from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from trading_agent_v2.schemas import ExecutionResult, PortfolioState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PortfolioManager:
    def __init__(self, portfolio_file: str):
        self.portfolio_file = portfolio_file

    # =========================================================
    # File / initialization
    # =========================================================

    def ensure_portfolio_exists(self, initial_cash: float = 10000.0) -> None:
        """
        Create a new paper portfolio file if it does not exist.
        """
        if not os.path.exists(self.portfolio_file):
            portfolio = PortfolioState(
                cash=initial_cash,
                total_equity=initial_cash,
                realized_pnl=0.0,
                positions={},
                updated_at=utc_now_iso(),
            )
            self.save_portfolio(portfolio)

    def load_portfolio(self) -> PortfolioState:
        """
        Load portfolio from JSON file.
        """
        self.ensure_portfolio_exists()

        with open(self.portfolio_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return PortfolioState(
            cash=float(data.get("cash", 10000.0)),
            total_equity=float(data.get("total_equity", data.get("cash", 10000.0))),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            positions=data.get("positions", {}),
            updated_at=data.get("updated_at", ""),
        )

    def save_portfolio(self, portfolio: PortfolioState) -> None:
        """
        Save portfolio to JSON file.
        """
        os.makedirs(os.path.dirname(self.portfolio_file), exist_ok=True)

        with open(self.portfolio_file, "w", encoding="utf-8") as f:
            json.dump(portfolio.to_dict(), f, indent=4, ensure_ascii=False)

    # =========================================================
    # Position helpers
    # =========================================================

    def _get_or_create_position(
        self,
        portfolio: PortfolioState,
        symbol: str,
    ) -> Dict[str, Any]:
        if symbol not in portfolio.positions:
            portfolio.positions[symbol] = {
                "symbol": symbol,
                "quantity": 0.0,
                "avg_entry_price": 0.0,
                "market_price": 0.0,
                "market_value": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "updated_at": utc_now_iso(),
            }
        return portfolio.positions[symbol]

    def _remove_empty_position_if_needed(
        self,
        portfolio: PortfolioState,
        symbol: str,
        eps: float = 1e-12,
    ) -> None:
        pos = portfolio.positions.get(symbol)
        if not pos:
            return

        if abs(float(pos.get("quantity", 0.0))) < eps:
            del portfolio.positions[symbol]

    # =========================================================
    # Mark-to-market / snapshot
    # =========================================================

    def mark_to_market(
        self,
        portfolio: PortfolioState,
        market_prices: Dict[str, float],
    ) -> PortfolioState:
        """
        Update market_price, market_value, unrealized_pnl, total_equity
        using latest market prices.
        """
        total_positions_value = 0.0

        for symbol, pos in portfolio.positions.items():
            qty = float(pos.get("quantity", 0.0))
            avg_entry = float(pos.get("avg_entry_price", 0.0))
            market_price = float(market_prices.get(symbol, pos.get("market_price", 0.0)))

            market_value = qty * market_price
            unrealized_pnl = (market_price - avg_entry) * qty

            pos["market_price"] = market_price
            pos["market_value"] = market_value
            pos["unrealized_pnl"] = unrealized_pnl
            pos["updated_at"] = utc_now_iso()

            total_positions_value += market_value

        portfolio.total_equity = float(portfolio.cash) + total_positions_value
        portfolio.updated_at = utc_now_iso()
        return portfolio

    def get_portfolio_snapshot(
        self,
        portfolio: PortfolioState,
        market_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Return a JSON-friendly portfolio snapshot.
        """
        if market_prices:
            portfolio = self.mark_to_market(portfolio, market_prices)

        positions_snapshot = {}
        for symbol, pos in portfolio.positions.items():
            positions_snapshot[symbol] = {
                "symbol": pos.get("symbol", symbol),
                "quantity": float(pos.get("quantity", 0.0)),
                "avg_entry_price": float(pos.get("avg_entry_price", 0.0)),
                "market_price": float(pos.get("market_price", 0.0)),
                "market_value": float(pos.get("market_value", 0.0)),
                "unrealized_pnl": float(pos.get("unrealized_pnl", 0.0)),
                "realized_pnl": float(pos.get("realized_pnl", 0.0)),
                "updated_at": pos.get("updated_at", ""),
            }

        total_market_value = sum(
            float(pos.get("market_value", 0.0)) for pos in portfolio.positions.values()
        )

        return {
            "cash": float(portfolio.cash),
            "total_equity": float(portfolio.total_equity),
            "realized_pnl": float(portfolio.realized_pnl),
            "total_market_value": total_market_value,
            "position_count": len(portfolio.positions),
            "positions": positions_snapshot,
            "updated_at": portfolio.updated_at,
        }

    # =========================================================
    # Execution application
    # =========================================================

    def apply_execution_result(
        self,
        portfolio: PortfolioState,
        execution_result: ExecutionResult,
    ) -> PortfolioState:
        """
        Apply a filled paper trade result to the portfolio.

        Only handles status='filled'. Other statuses will leave the
        portfolio unchanged except timestamp refresh.
        """
        portfolio.updated_at = utc_now_iso()

        if execution_result.status != "filled":
            return portfolio

        symbol = execution_result.symbol
        action = execution_result.action.lower()
        filled_price = float(execution_result.filled_price or 0.0)
        filled_qty = float(execution_result.filled_qty or 0.0)
        fees = float(execution_result.fees or 0.0)

        if filled_price <= 0 or filled_qty <= 0:
            raise ValueError("Filled execution must have positive filled_price and filled_qty.")

        position = self._get_or_create_position(portfolio, symbol)

        old_qty = float(position.get("quantity", 0.0))
        old_avg = float(position.get("avg_entry_price", 0.0))
        old_realized = float(position.get("realized_pnl", 0.0))

        if action == "buy":
            cost = filled_price * filled_qty + fees

            if portfolio.cash < cost:
                raise ValueError(
                    f"Insufficient cash to apply buy execution. "
                    f"cash={portfolio.cash}, required={cost}"
                )

            new_qty = old_qty + filled_qty

            if new_qty <= 0:
                raise ValueError("Invalid resulting quantity after buy.")

            if old_qty > 0:
                new_avg = ((old_qty * old_avg) + (filled_qty * filled_price)) / new_qty
            else:
                new_avg = filled_price

            portfolio.cash -= cost

            position["quantity"] = new_qty
            position["avg_entry_price"] = new_avg
            position["updated_at"] = utc_now_iso()

        elif action == "sell":
            if old_qty < filled_qty:
                raise ValueError(
                    f"Insufficient position to sell. have={old_qty}, trying_to_sell={filled_qty}"
                )

            proceeds = filled_price * filled_qty - fees
            realized_gain = (filled_price - old_avg) * filled_qty

            new_qty = old_qty - filled_qty

            portfolio.cash += proceeds
            portfolio.realized_pnl += realized_gain

            position["quantity"] = new_qty
            position["realized_pnl"] = old_realized + realized_gain
            position["updated_at"] = utc_now_iso()

            if new_qty == 0:
                position["avg_entry_price"] = 0.0

        else:
            raise ValueError(f"Unsupported execution action: {execution_result.action}")

        # refresh mark-to-market for this symbol at filled price
        latest_prices = {symbol: filled_price}
        self.mark_to_market(portfolio, latest_prices)

        # remove empty positions
        self._remove_empty_position_if_needed(portfolio, symbol)

        portfolio.updated_at = utc_now_iso()
        return portfolio

    # =========================================================
    # Exposure helpers
    # =========================================================

    def get_symbol_exposure_pct(
        self,
        portfolio: PortfolioState,
        symbol: str,
    ) -> float:
        """
        Exposure as a fraction of total equity.
        """
        if portfolio.total_equity <= 0:
            return 0.0

        pos = portfolio.positions.get(symbol)
        if not pos:
            return 0.0

        market_value = float(pos.get("market_value", 0.0))
        return market_value / float(portfolio.total_equity)

    def get_total_invested_pct(self, portfolio: PortfolioState) -> float:
        """
        Total invested capital as a fraction of total equity.
        """
        if portfolio.total_equity <= 0:
            return 0.0

        total_market_value = sum(
            float(pos.get("market_value", 0.0)) for pos in portfolio.positions.values()
        )
        return total_market_value / float(portfolio.total_equity)