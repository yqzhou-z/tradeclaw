from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

try:
    import ccxt
except Exception:  # pragma: no cover - optional dependency
    ccxt = None

from trading_agent_v2.schemas import ExecutionResult, FinalDecision, PortfolioState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class OkxExecutor:
    """
    Live execution adapter for OKX spot trading via CCXT.
    """

    def __init__(
        self,
        api_key: str,
        secret: str,
        passphrase: str,
        use_sandbox: bool = False,
        timeout_ms: int = 10000,
        enable_rate_limit: bool = True,
        td_mode: str = "cash",
    ):
        if ccxt is None:
            raise RuntimeError("ccxt is not available. Please install ccxt before using OKX execution.")
        if not api_key or not secret or not passphrase:
            raise ValueError(
                "Missing OKX credentials. Please set OKX_API_KEY, OKX_SECRET_KEY, and OKX_PASSPHRASE."
            )

        self.exchange = ccxt.okx(
            {
                "apiKey": api_key,
                "secret": secret,
                "password": passphrase,
                "enableRateLimit": bool(enable_rate_limit),
                "timeout": int(timeout_ms),
                "options": {"defaultType": "spot"},
            }
        )
        self.use_sandbox = bool(use_sandbox)
        if self.use_sandbox:
            self.exchange.set_sandbox_mode(True)

        self.td_mode = td_mode
        self._markets_loaded = False

    def execute(
        self,
        decision: FinalDecision,
        portfolio: PortfolioState,
        market_prices: Optional[Dict[str, float]] = None,
    ) -> ExecutionResult:
        del portfolio  # live execution sizes from exchange balances, not local portfolio.

        action = (decision.action or "").lower().strip()
        symbol = decision.symbol
        timestamp = decision.timestamp or utc_now_iso()

        if action == "hold":
            return ExecutionResult(
                symbol=symbol,
                timestamp=timestamp,
                status="skipped",
                action="hold",
                message="Hold decision: no live execution performed.",
                metadata={"exchange": "okx", "sandbox": self.use_sandbox},
            )

        if action not in {"buy", "sell"}:
            return ExecutionResult(
                symbol=symbol,
                timestamp=timestamp,
                status="failed",
                action=action or "unknown",
                message=f"Unsupported action for OKX execution: {decision.action}",
                metadata={"exchange": "okx", "sandbox": self.use_sandbox},
            )

        try:
            self._ensure_markets_loaded()
            resolved_symbol = self._resolve_symbol(symbol)
            base_asset, quote_asset = self._split_symbol(symbol)

            market_price = self._safe_float((market_prices or {}).get(symbol), 0.0)
            if market_price <= 0:
                market_price = self._fetch_last_price(resolved_symbol)
            if market_price <= 0:
                return ExecutionResult(
                    symbol=symbol,
                    timestamp=timestamp,
                    status="failed",
                    action=action,
                    message=f"Could not fetch valid market price for {symbol}.",
                    metadata={"exchange": "okx", "sandbox": self.use_sandbox},
                )

            size_pct = max(0.0, float(decision.size_pct or 0.0))
            if size_pct <= 0:
                return ExecutionResult(
                    symbol=symbol,
                    timestamp=timestamp,
                    status="failed",
                    action=action,
                    message="size_pct must be > 0 for buy/sell.",
                    metadata={"exchange": "okx", "sandbox": self.use_sandbox},
                )

            balance = self.exchange.fetch_balance()
            requested_notional = 0.0
            order_amount = 0.0
            available_quote = 0.0
            available_base = 0.0

            if action == "buy":
                available_quote = self._balance_amount(balance, quote_asset, prefer_free=True)
                requested_notional = available_quote * size_pct
                order_amount = requested_notional / market_price if market_price > 0 else 0.0
            elif action == "sell":
                available_base = self._balance_amount(balance, base_asset, prefer_free=True)
                order_amount = available_base * size_pct
                requested_notional = order_amount * market_price

            min_required_amount = self._min_order_amount(resolved_symbol=resolved_symbol, market_price=market_price)
            if min_required_amount > 0 and 0 < order_amount < min_required_amount:
                if action == "buy":
                    max_affordable_amount = available_quote / market_price if market_price > 0 else 0.0
                    if max_affordable_amount >= min_required_amount:
                        order_amount = min_required_amount
                        requested_notional = order_amount * market_price
                    else:
                        return ExecutionResult(
                            symbol=symbol,
                            timestamp=timestamp,
                            status="failed",
                            action=action,
                            message=(
                                "Insufficient quote balance to meet OKX minimum order amount. "
                                f"required_amount={min_required_amount:.8f}, "
                                f"max_affordable_amount={max_affordable_amount:.8f}."
                            ),
                            metadata={
                                "exchange": "okx",
                                "sandbox": self.use_sandbox,
                                "market_price": market_price,
                                "requested_size_pct": size_pct,
                            },
                        )
                else:
                    if available_base >= min_required_amount:
                        order_amount = min_required_amount
                        requested_notional = order_amount * market_price
                    else:
                        return ExecutionResult(
                            symbol=symbol,
                            timestamp=timestamp,
                            status="failed",
                            action=action,
                            message=(
                                "Insufficient base balance to meet OKX minimum order amount. "
                                f"required_amount={min_required_amount:.8f}, "
                                f"available_base={available_base:.8f}."
                            ),
                            metadata={
                                "exchange": "okx",
                                "sandbox": self.use_sandbox,
                                "market_price": market_price,
                                "requested_size_pct": size_pct,
                            },
                        )

            if order_amount <= 0:
                return ExecutionResult(
                    symbol=symbol,
                    timestamp=timestamp,
                    status="failed",
                    action=action,
                    message=(
                        f"Computed {action} amount is non-positive. "
                        "Check available balance and decision size_pct."
                    ),
                    metadata={
                        "exchange": "okx",
                        "sandbox": self.use_sandbox,
                        "requested_notional": requested_notional,
                        "market_price": market_price,
                    },
                )

            order = self.exchange.create_order(
                symbol=resolved_symbol,
                type="market",
                side=action,
                amount=order_amount,
                params={"tdMode": self.td_mode},
            )
            order = self._hydrate_order(order=order, symbol=resolved_symbol)

            filled_qty, avg_price, gross_cost, fees, order_status = self._extract_fill(
                order=order,
                fallback_price=market_price,
            )
            if filled_qty <= 0:
                return ExecutionResult(
                    symbol=symbol,
                    timestamp=utc_now_iso(),
                    status="failed",
                    action=action,
                    message="OKX order submitted but no fill quantity detected.",
                    metadata={
                        "exchange": "okx",
                        "sandbox": self.use_sandbox,
                        "resolved_symbol": resolved_symbol,
                        "order_id": str(order.get("id", "")),
                        "order_status": order_status,
                    },
                )

            if action == "buy":
                notional_value = max(0.0, gross_cost)
            else:
                notional_value = max(0.0, gross_cost - fees)

            return ExecutionResult(
                symbol=symbol,
                timestamp=utc_now_iso(),
                status="filled",
                action=action,
                filled_price=avg_price,
                filled_qty=filled_qty,
                notional_value=notional_value,
                fees=fees,
                message="OKX market order executed.",
                metadata={
                    "exchange": "okx",
                    "sandbox": self.use_sandbox,
                    "resolved_symbol": resolved_symbol,
                    "order_id": str(order.get("id", "")),
                    "order_status": order_status,
                    "requested_size_pct": size_pct,
                    "requested_notional": requested_notional,
                    "market_price": market_price,
                    "order_type": decision.order_type,
                    "reason": decision.reason,
                },
            )
        except Exception as exc:
            return ExecutionResult(
                symbol=symbol,
                timestamp=utc_now_iso(),
                status="failed",
                action=action,
                message=f"OKX execution failed: {str(exc)}",
                metadata={"exchange": "okx", "sandbox": self.use_sandbox},
            )

    def sync_portfolio_state(
        self,
        portfolio: PortfolioState,
        symbols: Iterable[str],
        market_prices: Optional[Dict[str, float]] = None,
    ) -> PortfolioState:
        """
        Sync local portfolio with OKX balances for tracked symbols.
        """
        self._ensure_markets_loaded()
        balance = self.exchange.fetch_balance()
        tracked_symbols = [s for s in symbols if isinstance(s, str) and "/" in s]
        if not tracked_symbols:
            return portfolio

        primary_quote = self._split_symbol(tracked_symbols[0])[1]
        cash = self._balance_amount(balance, primary_quote, prefer_free=True)
        if cash <= 0:
            cash = self._balance_amount(balance, primary_quote, prefer_free=False)
        portfolio.cash = cash

        updated_positions = dict(portfolio.positions or {})
        tracked_set = set(tracked_symbols)
        total_market_value = 0.0
        now = utc_now_iso()

        for symbol in tracked_symbols:
            base_asset, _ = self._split_symbol(symbol)
            qty = self._balance_amount(balance, base_asset, prefer_free=False)
            if qty <= 1e-12:
                updated_positions.pop(symbol, None)
                continue

            price = self._safe_float((market_prices or {}).get(symbol), 0.0)
            if price <= 0:
                resolved_symbol = self._resolve_symbol(symbol)
                price = self._fetch_last_price(resolved_symbol)

            existing = updated_positions.get(symbol, {})
            avg_entry = self._safe_float(existing.get("avg_entry_price"), price if price > 0 else 0.0)
            if avg_entry <= 0 and price > 0:
                avg_entry = price

            market_value = qty * price if price > 0 else 0.0
            unrealized_pnl = (price - avg_entry) * qty if price > 0 else 0.0
            realized_pnl = self._safe_float(existing.get("realized_pnl"), 0.0)

            updated_positions[symbol] = {
                "symbol": symbol,
                "quantity": qty,
                "avg_entry_price": avg_entry,
                "market_price": price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "updated_at": now,
            }
            total_market_value += market_value

        for symbol, pos in updated_positions.items():
            if symbol in tracked_set:
                continue
            total_market_value += self._safe_float(pos.get("market_value"), 0.0)

        portfolio.positions = updated_positions
        portfolio.total_equity = float(portfolio.cash) + total_market_value
        portfolio.updated_at = now
        return portfolio

    def _ensure_markets_loaded(self) -> None:
        if self._markets_loaded:
            return
        self.exchange.load_markets()
        self._markets_loaded = True

    def _resolve_symbol(self, symbol: str) -> str:
        if symbol in self.exchange.markets:
            return symbol

        base_asset, quote_asset = self._split_symbol(symbol)
        for market_symbol, market in self.exchange.markets.items():
            if not market.get("spot", False):
                continue
            if (
                str(market.get("base", "")).upper() == base_asset
                and str(market.get("quote", "")).upper() == quote_asset
            ):
                return market_symbol
        raise ValueError(f"Symbol {symbol} is not available on OKX spot markets.")

    def _fetch_last_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        return self._safe_float(ticker.get("last"), 0.0)

    def _min_order_amount(self, resolved_symbol: str, market_price: float) -> float:
        market = (self.exchange.markets or {}).get(resolved_symbol, {})
        if not isinstance(market, dict):
            return 0.0

        limits = market.get("limits", {})
        if not isinstance(limits, dict):
            return 0.0

        amount_limits = limits.get("amount", {})
        cost_limits = limits.get("cost", {})
        min_amount = self._safe_float(amount_limits.get("min"), 0.0) if isinstance(amount_limits, dict) else 0.0
        min_cost = self._safe_float(cost_limits.get("min"), 0.0) if isinstance(cost_limits, dict) else 0.0
        implied_min_amount = min_cost / market_price if (min_cost > 0 and market_price > 0) else 0.0
        return max(0.0, min_amount, implied_min_amount)

    def _hydrate_order(self, order: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        if not isinstance(order, dict):
            return {}
        order_id = str(order.get("id", "") or "")
        if not order_id:
            return order
        try:
            fetched = self.exchange.fetch_order(order_id, symbol)
            if isinstance(fetched, dict) and fetched:
                return fetched
        except Exception:
            pass
        return order

    def _extract_fill(
        self,
        order: Dict[str, Any],
        fallback_price: float,
    ) -> tuple[float, float, float, float, str]:
        filled_qty = self._safe_float(order.get("filled"), 0.0)
        avg_price = self._safe_float(order.get("average"), self._safe_float(order.get("price"), fallback_price))
        if avg_price <= 0:
            avg_price = fallback_price

        gross_cost = self._safe_float(order.get("cost"), 0.0)
        fee_obj = order.get("fee") or {}
        fees = self._safe_float(fee_obj.get("cost"), 0.0) if isinstance(fee_obj, dict) else 0.0
        order_status = str(order.get("status", "unknown"))

        if filled_qty <= 0 and gross_cost > 0 and avg_price > 0:
            filled_qty = gross_cost / avg_price
        if gross_cost <= 0 and filled_qty > 0 and avg_price > 0:
            gross_cost = filled_qty * avg_price

        return filled_qty, avg_price, gross_cost, fees, order_status

    def _split_symbol(self, symbol: str) -> tuple[str, str]:
        if "/" not in symbol:
            raise ValueError(f"Unsupported symbol format: {symbol}")
        base_asset, quote_asset = symbol.split("/", 1)
        quote_asset = quote_asset.split(":")[0]
        return base_asset.upper(), quote_asset.upper()

    def _balance_amount(self, balance: Dict[str, Any], asset: str, prefer_free: bool) -> float:
        preferred_bucket = "free" if prefer_free else "total"
        fallback_bucket = "total" if prefer_free else "free"

        preferred_map = balance.get(preferred_bucket, {})
        if isinstance(preferred_map, dict) and asset in preferred_map:
            return self._safe_float(preferred_map.get(asset), 0.0)

        asset_obj = balance.get(asset, {})
        if isinstance(asset_obj, dict):
            if asset_obj.get(preferred_bucket) is not None:
                return self._safe_float(asset_obj.get(preferred_bucket), 0.0)
            if asset_obj.get(fallback_bucket) is not None:
                return self._safe_float(asset_obj.get(fallback_bucket), 0.0)

        fallback_map = balance.get(fallback_bucket, {})
        if isinstance(fallback_map, dict) and asset in fallback_map:
            return self._safe_float(fallback_map.get(asset), 0.0)

        return 0.0

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
