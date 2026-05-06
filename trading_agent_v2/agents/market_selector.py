from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from trading_agent_v2.llm.openai_client import OpenAIJsonClient
from trading_agent_v2.tools.market_tools import MarketTools


@dataclass
class MarketSelectionResult:
    selected_symbols: list[str]
    candidate_count: int
    llm_used: bool
    regime_summary: str
    rationale_by_symbol: dict[str, str]
    candidate_pool: list[dict[str, Any]]


class MarketSelector:
    def __init__(
        self,
        market_tools: MarketTools,
        llm_client: OpenAIJsonClient | None = None,
        llm_primary: bool = True,
    ):
        self.market_tools = market_tools
        self.llm_client = llm_client
        self.llm_primary = llm_primary

    def select_symbols(
        self,
        quote_assets: list[str],
        shortlist_size: int,
        scout_limit: int,
        llm_candidate_pool_size: int,
        portfolio_snapshot: dict[str, Any] | None = None,
        force_include_symbols: list[str] | None = None,
    ) -> MarketSelectionResult:
        portfolio_snapshot = portfolio_snapshot or {}
        forced_symbols = self._normalize_symbols(force_include_symbols or [])

        candidates = self.market_tools.scan_tradeable_candidates(
            quote_assets=quote_assets,
            limit=max(1, scout_limit),
        )
        candidate_by_symbol = {
            str(item.get("symbol", "")).strip(): item
            for item in candidates
            if str(item.get("symbol", "")).strip()
        }
        if not candidates:
            raise RuntimeError("MarketSelector found no tradeable candidates from the market scan.")

        llm_pool = candidates[: max(1, llm_candidate_pool_size)]
        llm_selection = self._select_with_llm(
            candidates=llm_pool,
            portfolio_snapshot=portfolio_snapshot,
            shortlist_size=max(1, shortlist_size),
        )

        chosen = self._normalize_symbols(llm_selection.get("selected_symbols", []))

        ordered_symbols = self._merge_symbol_lists(
            forced_symbols,
            chosen,
        )[: max(1, shortlist_size + len(forced_symbols))]

        if not ordered_symbols:
            raise RuntimeError("MarketSelector LLM returned no usable symbols.")

        rationale_by_symbol = self._build_rationales(
            selected_symbols=ordered_symbols,
            forced_symbols=forced_symbols,
            llm_rationale=llm_selection.get("rationale_by_symbol", {}),
        )

        regime_summary = str(llm_selection.get("market_regime_summary", "") or "").strip()
        if not regime_summary:
            raise ValueError("MarketSelector LLM response is missing market_regime_summary.")

        selected_pool = [candidate_by_symbol[symbol] for symbol in ordered_symbols if symbol in candidate_by_symbol]

        return MarketSelectionResult(
            selected_symbols=ordered_symbols,
            candidate_count=len(candidates),
            llm_used=bool(llm_selection.get("llm_used", False)),
            regime_summary=regime_summary,
            rationale_by_symbol=rationale_by_symbol,
            candidate_pool=selected_pool,
        )

    def _select_with_llm(
        self,
        candidates: list[dict[str, Any]],
        portfolio_snapshot: dict[str, Any],
        shortlist_size: int,
    ) -> dict[str, Any]:
        if not self.llm_primary or self.llm_client is None or not candidates:
            raise RuntimeError("MarketSelector requires an LLM client and non-empty candidates.")

        payload = {
            "task": (
                "Choose symbols worth deeper analysis in the next batch. "
                "You may favor volatile altcoins and recent breakouts. "
                "Do not default to BTC unless it truly looks best."
            ),
            "constraints": {
                "max_symbols": max(1, shortlist_size),
                "must_return_symbols_from_candidates_only": True,
                "trading_style": "opportunistic spot trading from a USDT account on OKX",
                "avoid_over_diversification": True,
            },
            "portfolio": {
                "cash": portfolio_snapshot.get("cash"),
                "total_equity": portfolio_snapshot.get("total_equity"),
                "positions": list((portfolio_snapshot.get("positions") or {}).keys()),
            },
            "candidates": [
                {
                    "symbol": candidate.get("symbol"),
                    "last_price": candidate.get("last_price"),
                    "pct_change_24h": candidate.get("pct_change_24h"),
                    "range_pct_24h": candidate.get("range_pct_24h"),
                    "spread_pct": candidate.get("spread_pct"),
                    "quote_volume_24h": candidate.get("quote_volume_24h"),
                    "listing_age_hours": candidate.get("listing_age_hours"),
                    "is_recent_listing": candidate.get("is_recent_listing"),
                    "state": candidate.get("state"),
                    "scout_score": candidate.get("scout_score"),
                }
                for candidate in candidates
            ],
        }

        system_prompt = (
            "You are a crypto market selector for an aggressive spot-trading system. "
            "Pick the symbols with the best near-term tradable opportunity from the provided market-wide candidate list. "
            "You may prefer volatile altcoins, momentum names, and strong rotations. "
            "Return JSON only with keys: selected_symbols, rationale_by_symbol, market_regime_summary. "
            "selected_symbols must copy exact symbol strings from the candidate list, including the slash format like BTC/USDT. "
            "rationale_by_symbol must map each selected symbol to a short rationale."
        )
        response = self.llm_client.complete_json(system_prompt=system_prompt, payload=payload)

        raw_selected = response.get("selected_symbols", [])
        if not isinstance(raw_selected, list):
            raise ValueError("MarketSelector LLM response must include selected_symbols as a list.")
        selected_symbols = self._resolve_selected_symbols(
            raw_selected=raw_selected,
            candidates=candidates,
            shortlist_size=max(1, shortlist_size),
        )
        if not selected_symbols:
            raise ValueError(
                "MarketSelector LLM selected no symbols from the candidate list. "
                f"raw_selected={raw_selected!r}"
            )

        rationales = response.get("rationale_by_symbol", {})
        if not isinstance(rationales, dict):
            raise ValueError("MarketSelector LLM response must include rationale_by_symbol as an object.")
        normalized_rationales = self._normalize_rationale_map(rationales, candidates=candidates)
        missing_rationales = [symbol for symbol in selected_symbols if not str(normalized_rationales.get(symbol, "")).strip()]
        if missing_rationales:
            raise ValueError(f"MarketSelector LLM missing rationale(s) for: {', '.join(missing_rationales)}.")

        return {
            "selected_symbols": selected_symbols,
            "rationale_by_symbol": normalized_rationales,
            "market_regime_summary": str(response.get("market_regime_summary", "") or "").strip(),
            "llm_used": bool(selected_symbols),
        }

    def _build_rationales(
        self,
        selected_symbols: list[str],
        forced_symbols: list[str],
        llm_rationale: dict[str, Any],
    ) -> dict[str, str]:
        output: dict[str, str] = {}
        forced = set(forced_symbols)
        for symbol in selected_symbols:
            reason = str(llm_rationale.get(symbol, "") or "").strip()
            if not reason and symbol in forced:
                reason = "Included because it is an existing portfolio position that must be monitored."
            if not reason:
                raise ValueError(f"MarketSelector missing LLM rationale for {symbol}.")
            output[symbol] = reason
        return output

    def _resolve_selected_symbols(
        self,
        raw_selected: list[Any],
        candidates: list[dict[str, Any]],
        shortlist_size: int,
    ) -> list[str]:
        candidate_by_key: dict[str, str] = {}
        for candidate in candidates:
            symbol = str(candidate.get("symbol", "")).strip().upper()
            key = self._canonicalize_symbol(symbol)
            if symbol and key and key not in candidate_by_key:
                candidate_by_key[key] = symbol

        output: list[str] = []
        seen: set[str] = set()
        for item in raw_selected:
            symbol = candidate_by_key.get(self._canonicalize_symbol(item))
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            output.append(symbol)
            if len(output) >= shortlist_size:
                break
        return output

    def _normalize_rationale_map(
        self,
        rationales: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, str]:
        candidate_by_key: dict[str, str] = {}
        for candidate in candidates:
            symbol = str(candidate.get("symbol", "")).strip().upper()
            key = self._canonicalize_symbol(symbol)
            if symbol and key and key not in candidate_by_key:
                candidate_by_key[key] = symbol

        output: dict[str, str] = {}
        for raw_symbol, raw_reason in rationales.items():
            symbol = candidate_by_key.get(self._canonicalize_symbol(raw_symbol))
            reason = str(raw_reason or "").strip()
            if not symbol or not reason:
                continue
            output[symbol] = reason
        return output

    def _merge_symbol_lists(self, *groups: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for symbol in group:
                item = str(symbol).strip().upper()
                if not item or item in seen:
                    continue
                seen.add(item)
                merged.append(item)
        return merged

    def _normalize_symbols(self, symbols: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            text = str(symbol or "").strip().upper()
            if not text or "/" not in text or text in seen:
                continue
            seen.add(text)
            output.append(text)
        return output

    def _canonicalize_symbol(self, symbol: Any) -> str:
        text = str(symbol or "").strip().upper()
        if not text:
            return ""
        return re.sub(r"[^A-Z0-9]", "", text)
