from __future__ import annotations

from collections import Counter
from typing import Any

from trading_agent_v2.config import AppConfig
from trading_agent_v2.llm.openai_client import OpenAIJsonClient
from trading_agent_v2.memory.daily_review_store import DailyReviewStore
from trading_agent_v2.memory.episodic_memory import EpisodicMemoryStore
from trading_agent_v2.memory.time_utils import DEFAULT_LOCAL_TIMEZONE, to_local_date_str, utc_now_iso


DEFAULT_DAILY_REVIEW_TIMEZONE = DEFAULT_LOCAL_TIMEZONE


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _short_text(value: Any, max_len: int = 200) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


class DailyReviewEngine:
    def __init__(
        self,
        llm_client: OpenAIJsonClient | None = None,
        timezone_name: str = DEFAULT_DAILY_REVIEW_TIMEZONE,
    ):
        self.llm_client = llm_client
        self.timezone_name = timezone_name

    def generate_review(
        self,
        review_date: str,
        episodes: list[dict[str, Any]],
        previous_episode: dict[str, Any] | None = None,
        run_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cycle_records = self._build_cycle_records(episodes=episodes, run_results=run_results or [])
        performance = self._build_performance(
            cycle_records=cycle_records,
            previous_episode=previous_episode,
        )

        review = {
            "review_date": review_date,
            "timezone": self.timezone_name,
            "generated_at": utc_now_iso(),
            "symbols": performance.get("symbols", []),
            "performance": performance,
            "cycles": cycle_records,
            "headline": "",
            "summary": "",
            "market_regime": "",
            "strategy_takeaways": [],
            "mistakes_or_risks": [],
            "tomorrow_playbook": [],
            "generation": {
                "mode": "fallback",
                "llm_enabled": bool(self.llm_client and self.llm_client.enabled),
                "model": getattr(self.llm_client, "model", ""),
                "llm_error": None,
            },
        }

        llm_summary = self._generate_llm_summary(review_date=review_date, performance=performance, cycles=cycle_records)
        if llm_summary:
            review["headline"] = str(llm_summary.get("headline", "")).strip()
            review["summary"] = str(llm_summary.get("summary", "")).strip()
            review["market_regime"] = str(llm_summary.get("market_regime", "")).strip()
            review["strategy_takeaways"] = self._normalize_text_list(llm_summary.get("strategy_takeaways"))
            review["mistakes_or_risks"] = self._normalize_text_list(llm_summary.get("mistakes_or_risks"))
            review["tomorrow_playbook"] = self._normalize_text_list(llm_summary.get("tomorrow_playbook"))
            review["generation"]["mode"] = "llm"
        else:
            fallback = self._build_fallback_summary(review_date=review_date, performance=performance, cycles=cycle_records)
            review.update(fallback)
            if self.llm_client is not None:
                review["generation"]["llm_error"] = self.llm_client.last_error

        review["strategy_takeaways"] = review["strategy_takeaways"][:4]
        review["mistakes_or_risks"] = review["mistakes_or_risks"][:4]
        review["tomorrow_playbook"] = review["tomorrow_playbook"][:4]
        return review

    def _build_cycle_records(
        self,
        episodes: list[dict[str, Any]],
        run_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result_map: dict[tuple[str, str], dict[str, Any]] = {}
        for result in run_results:
            key = (
                str(result.get("symbol", "")).strip().upper(),
                str(result.get("timestamp", "")).strip(),
            )
            if key[0] and key[1]:
                result_map[key] = result

        source_items = episodes or run_results
        records: list[dict[str, Any]] = []
        for item in source_items:
            symbol = str(item.get("symbol", "")).strip().upper()
            timestamp = str(item.get("timestamp", "")).strip()
            result = result_map.get((symbol, timestamp), item)

            final_decision = (item.get("final_decision") or result.get("final_decision") or {})
            execution_result = (item.get("execution_result") or result.get("execution_result") or {})
            risk_report = (item.get("risk_report") or result.get("risk_report") or {})
            proposal = (item.get("proposal") or result.get("proposal") or {})
            snapshot = (item.get("portfolio_snapshot") or result.get("portfolio_snapshot") or {})
            raw_context = (item.get("raw_context") or result.get("raw_context") or {})
            market_data = raw_context.get("market_data", {}) if isinstance(raw_context, dict) else {}

            action = str(final_decision.get("action", execution_result.get("action", "hold")) or "hold").lower()
            status = str(execution_result.get("status", "unknown") or "unknown").lower()
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "action": action,
                    "status": status,
                    "size_pct": _safe_float(final_decision.get("size_pct"), 0.0) or 0.0,
                    "decision_reason": _short_text(final_decision.get("reason", proposal.get("thesis", "")), max_len=220),
                    "risk_summary": _short_text(risk_report.get("summary", ""), max_len=180),
                    "execution_message": _short_text(execution_result.get("message", ""), max_len=180),
                    "filled_qty": _safe_float(execution_result.get("filled_qty")),
                    "filled_price": _safe_float(execution_result.get("filled_price")),
                    "market_price": _safe_float(market_data.get("price")),
                    "portfolio_snapshot": snapshot,
                }
            )

        records.sort(key=lambda item: str(item.get("timestamp", "")))
        return records

    def _build_performance(
        self,
        cycle_records: list[dict[str, Any]],
        previous_episode: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        symbols = _dedupe_preserve_order([str(item.get("symbol", "")) for item in cycle_records])
        action_counts = Counter(str(item.get("action", "hold")).lower() for item in cycle_records)
        status_counts = Counter(str(item.get("status", "unknown")).lower() for item in cycle_records)

        previous_snapshot = {}
        if isinstance(previous_episode, dict):
            previous_snapshot = dict(previous_episode.get("portfolio_snapshot") or {})

        first_snapshot = {}
        last_snapshot = {}
        if cycle_records:
            first_snapshot = dict(cycle_records[0].get("portfolio_snapshot") or {})
            last_snapshot = dict(cycle_records[-1].get("portfolio_snapshot") or {})

        baseline_source = "previous_day_snapshot" if previous_snapshot else "first_cycle_snapshot"
        starting_equity = _safe_float(previous_snapshot.get("total_equity"))
        starting_realized_pnl = _safe_float(previous_snapshot.get("realized_pnl"))
        if starting_equity is None:
            starting_equity = _safe_float(first_snapshot.get("total_equity"))
        if starting_realized_pnl is None:
            starting_realized_pnl = _safe_float(first_snapshot.get("realized_pnl"))

        ending_equity = _safe_float(last_snapshot.get("total_equity"))
        ending_realized_pnl = _safe_float(last_snapshot.get("realized_pnl"))
        ending_cash = _safe_float(last_snapshot.get("cash"))
        ending_total_market_value = _safe_float(last_snapshot.get("total_market_value"))
        ending_position_count = int(last_snapshot.get("position_count", 0) or 0)
        ending_positions = sorted(list((last_snapshot.get("positions") or {}).keys())) if isinstance(last_snapshot, dict) else []

        equity_change = None
        if starting_equity is not None and ending_equity is not None:
            equity_change = ending_equity - starting_equity

        realized_pnl_change = None
        if starting_realized_pnl is not None and ending_realized_pnl is not None:
            realized_pnl_change = ending_realized_pnl - starting_realized_pnl

        equity_change_pct = None
        if starting_equity not in (None, 0.0) and equity_change is not None:
            equity_change_pct = equity_change / starting_equity

        return {
            "cycle_count": len(cycle_records),
            "symbol_count": len(symbols),
            "symbols": symbols,
            "action_breakdown": {
                "buy": int(action_counts.get("buy", 0)),
                "sell": int(action_counts.get("sell", 0)),
                "hold": int(action_counts.get("hold", 0)),
            },
            "execution_breakdown": {
                "filled": int(status_counts.get("filled", 0)),
                "skipped": int(status_counts.get("skipped", 0)),
                "rejected": int(status_counts.get("rejected", 0)),
                "failed": int(status_counts.get("failed", 0)),
                "unknown": int(status_counts.get("unknown", 0)),
            },
            "starting_equity": starting_equity,
            "ending_equity": ending_equity,
            "equity_change": equity_change,
            "equity_change_pct": equity_change_pct,
            "starting_realized_pnl": starting_realized_pnl,
            "ending_realized_pnl": ending_realized_pnl,
            "realized_pnl_change": realized_pnl_change,
            "ending_cash": ending_cash,
            "ending_total_market_value": ending_total_market_value,
            "ending_position_count": ending_position_count,
            "ending_positions": ending_positions,
            "filled_trade_count": int(status_counts.get("filled", 0)),
            "baseline_source": baseline_source,
        }

    def _generate_llm_summary(
        self,
        review_date: str,
        performance: dict[str, Any],
        cycles: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if self.llm_client is None:
            return None

        system_prompt = (
            "You are reviewing one day of a live crypto trading system. "
            "Return only valid JSON with keys: headline, summary, market_regime, "
            "strategy_takeaways, mistakes_or_risks, tomorrow_playbook. "
            "Each list should contain 2 to 4 concise, specific strings. "
            "Ground the review in the provided actions, execution outcomes, and equity changes. "
            "Avoid generic advice and avoid mentioning any missing data."
        )
        payload = {
            "review_date": review_date,
            "timezone": self.timezone_name,
            "performance": performance,
            "cycles": [
                {
                    "symbol": item.get("symbol"),
                    "timestamp": item.get("timestamp"),
                    "action": item.get("action"),
                    "status": item.get("status"),
                    "size_pct": item.get("size_pct"),
                    "decision_reason": item.get("decision_reason"),
                    "risk_summary": item.get("risk_summary"),
                    "execution_message": item.get("execution_message"),
                    "filled_qty": item.get("filled_qty"),
                    "filled_price": item.get("filled_price"),
                    "equity_after": (item.get("portfolio_snapshot") or {}).get("total_equity"),
                    "realized_pnl_after": (item.get("portfolio_snapshot") or {}).get("realized_pnl"),
                }
                for item in cycles
            ],
        }
        return self.llm_client.complete_json(system_prompt=system_prompt, payload=payload)

    def _build_fallback_summary(
        self,
        review_date: str,
        performance: dict[str, Any],
        cycles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        filled_trade_count = int(performance.get("filled_trade_count", 0))
        cycle_count = int(performance.get("cycle_count", 0))
        symbols = performance.get("symbols", []) or []
        action_breakdown = performance.get("action_breakdown", {}) or {}
        equity_change = _safe_float(performance.get("equity_change"))
        realized_pnl_change = _safe_float(performance.get("realized_pnl_change"))

        headline = f"{review_date}: reviewed {cycle_count} cycles across {len(symbols)} symbols."
        if equity_change is not None:
            headline = f"{review_date}: equity {'up' if equity_change >= 0 else 'down'} {abs(equity_change):.2f} across {cycle_count} cycles."

        summary_parts = [
            f"Processed {cycle_count} cycles for {', '.join(symbols) if symbols else 'the configured universe'}.",
            f"Filled trades: {filled_trade_count}.",
        ]
        if equity_change is not None:
            summary_parts.append(f"Equity change: {equity_change:+.2f}.")
        if realized_pnl_change is not None:
            summary_parts.append(f"Realized PnL change: {realized_pnl_change:+.2f}.")

        takeaways: list[str] = []
        risks: list[str] = []
        playbook: list[str] = []

        if filled_trade_count == 0:
            takeaways.append("Most cycles stayed inactive; the system did not find enough validated directional edge.")
            playbook.append("Keep waiting for cleaner setups instead of forcing turnover.")
        else:
            takeaways.append(f"The system converted {filled_trade_count} cycle(s) into executed trades.")
            playbook.append("Keep sizing discipline on setups that survive risk review and validation.")

        if (action_breakdown.get("hold", 0) or 0) >= max(2, cycle_count - 1):
            takeaways.append("Hold decisions dominated the day, which suggests signals stayed mixed or weak.")

        if equity_change is not None and equity_change < 0:
            risks.append("Daily equity closed lower; review whether position sizing was too aggressive for the observed edge.")
            playbook.append("Bias toward tighter sizing until the next batch shows a cleaner positive edge.")
        elif equity_change is not None and equity_change > 0:
            takeaways.append("Daily equity improved, which suggests the current process captured at least some valid edge.")
            playbook.append("Preserve the current discipline and avoid increasing risk just because the last batch worked.")

        if realized_pnl_change is not None and realized_pnl_change < 0:
            risks.append("Realized losses increased during the day; exits or timing likely need closer review.")

        rejected_or_failed = int((performance.get("execution_breakdown", {}) or {}).get("rejected", 0)) + int(
            (performance.get("execution_breakdown", {}) or {}).get("failed", 0)
        )
        if rejected_or_failed > 0:
            risks.append("Some decisions did not execute cleanly; double-check validation and execution friction.")

        if not risks:
            risks.append("No major execution failure stood out, but signal quality should still be monitored day by day.")

        if not playbook:
            playbook.append("Carry forward only the patterns that were supported by both validation and outcome.")

        return {
            "headline": headline,
            "summary": " ".join(summary_parts),
            "market_regime": "mixed",
            "strategy_takeaways": _dedupe_preserve_order(takeaways)[:4],
            "mistakes_or_risks": _dedupe_preserve_order(risks)[:4],
            "tomorrow_playbook": _dedupe_preserve_order(playbook)[:4],
        }

    def _normalize_text_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return _dedupe_preserve_order([_short_text(item, max_len=220) for item in value])
        if isinstance(value, str):
            text = _short_text(value, max_len=220)
            return [text] if text else []
        return []


def create_and_store_daily_review(
    config: AppConfig,
    run_results: list[dict[str, Any]] | None = None,
    timezone_name: str = DEFAULT_DAILY_REVIEW_TIMEZONE,
) -> dict[str, Any]:
    run_results = run_results or []
    reference_timestamp = None
    if run_results:
        reference_timestamp = run_results[-1].get("timestamp")
    review_date = to_local_date_str(reference_timestamp, timezone_name=timezone_name, default_now=True)
    if review_date is None:
        raise RuntimeError("Unable to determine daily review date.")

    episodic_store = EpisodicMemoryStore(str(config.episodic_memory_file))
    daily_review_store = DailyReviewStore(str(config.daily_review_file))
    llm_client = OpenAIJsonClient(
        enabled=config.llm.enabled,
        model=config.llm.model,
        base_url=config.llm.base_url,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout_sec=config.llm.timeout_sec,
        thinking_enabled=config.llm.thinking_enabled,
        reasoning_effort=config.llm.reasoning_effort,
    )

    episodes = episodic_store.load_for_local_date(review_date, timezone_name=timezone_name)
    previous_episode = episodic_store.load_latest_before_local_date(review_date, timezone_name=timezone_name)

    engine = DailyReviewEngine(
        llm_client=llm_client,
        timezone_name=timezone_name,
    )
    review = engine.generate_review(
        review_date=review_date,
        episodes=episodes,
        previous_episode=previous_episode,
        run_results=run_results,
    )
    daily_review_store.append_review(review)
    return review
