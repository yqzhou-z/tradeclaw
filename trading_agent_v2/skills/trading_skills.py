from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from trading_agent_v2.config import AppConfig
from trading_agent_v2.memory.episodic_memory import EpisodicMemoryStore
from trading_agent_v2.memory.pattern_memory import PatternMemoryStore
from trading_agent_v2.memory.reflection_engine import ReflectionEngine
from trading_agent_v2.memory.retrieval import MemoryRetriever
from trading_agent_v2.memory.strategic_memory import StrategicMemoryStore
from trading_agent_v2.portfolio.portfolio_manager import PortfolioManager
from trading_agent_v2.portfolio.trade_logger import TradeLogger
from trading_agent_v2.schemas import ExecutionResult, RawContext, build_episode_record
from trading_agent_v2.tools.feature_builder import FeatureBuilder
from trading_agent_v2.tools.market_tools import MarketTools
from trading_agent_v2.tools.news_tools import NewsTools
from trading_agent_v2.tools.onchain_tools import OnchainTools
from trading_agent_v2.tools.social_tools import SocialTools


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    return {"value": item}


def _fallback_hold_proposal(symbol: str, reason: str) -> dict[str, Any]:
    return {
        "proposal_id": "fallback",
        "symbol": symbol,
        "action": "hold",
        "size_pct": 0.0,
        "confidence": 0.0,
        "thesis": reason,
        "style": "base",
        "supporting_factors": [],
        "conflicting_factors": ["missing_proposals"],
        "reasoning_trace": {"planner_type": "fallback"},
        "metadata": {"fallback_reason": reason},
    }


@dataclass
class TradingSkills:
    market_tools: MarketTools
    news_tools: NewsTools
    onchain_tools: OnchainTools
    social_tools: SocialTools
    feature_builder: FeatureBuilder
    market_analyst: Any
    news_analyst: Any
    planner_agent: Any
    critic_agent: Any
    memory_retriever: MemoryRetriever
    risk_manager: Any
    trader_agent: Any
    position_sizer: Any
    validator: Any
    executor: Any
    portfolio_manager: PortfolioManager
    episodic_memory: EpisodicMemoryStore
    reflection_engine: ReflectionEngine
    strategic_memory_store: StrategicMemoryStore
    pattern_memory_store: PatternMemoryStore
    trade_logger: TradeLogger
    llm_client: Any

    def collect_raw_context(self, symbol: str) -> RawContext:
        market_data = self.market_tools.get_market_snapshot(symbol)
        news_data = self.news_tools.get_latest_news(symbol, limit=3)
        news_summary = self.news_tools.summarize_sentiment(news_data)
        onchain_data = self.onchain_tools.get_onchain_snapshot(symbol)
        social_data = self.social_tools.get_social_snapshot(symbol, news_data=news_data)
        social_data.update(news_summary)

        return RawContext(
            symbol=symbol,
            timestamp=utc_now_iso(),
            market_data=market_data,
            news_data=news_data,
            onchain_data=onchain_data,
            social_data=social_data,
        )

    def build_market_prices(self, raw_context: RawContext) -> dict[str, float]:
        return {raw_context.symbol: float(raw_context.market_data.get("price", 0.0))}

    def build_features(self, symbol: str, raw_context: RawContext) -> dict[str, Any]:
        return self.feature_builder.build(
            symbol=symbol,
            market_data=raw_context.market_data,
            news_data={
                "items": raw_context.news_data,
                "summary": raw_context.social_data.get("summary"),
                "sentiment": raw_context.social_data.get("sentiment_score"),
            },
            onchain_data=raw_context.onchain_data,
            social_data=raw_context.social_data,
        )

    def run_analysts(self, raw_context: RawContext) -> list[Any]:
        market_view = self.market_analyst.analyze(raw_context)
        news_view = self.news_analyst.analyze(raw_context)
        return [market_view, news_view]

    def build_retrieval_context(
        self,
        symbol: str,
        features: dict[str, Any],
        recent_memory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.memory_retriever.build_context_bundle(
            features=features,
            episodes=recent_memory,
            symbol=symbol,
            similar_k=5,
            recent_failure_k=3,
        )

    def generate_proposals(
        self,
        symbol: str,
        analyst_views: list[Any],
        features: dict[str, Any],
        portfolio: Any,
        strategy_memory: dict[str, Any],
        similar_cases: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        portfolio_payload = portfolio.to_dict() if hasattr(portfolio, "to_dict") else _as_dict(portfolio)
        return self.planner_agent.generate_proposals(
            symbol=symbol,
            analyst_views=analyst_views,
            features=features,
            portfolio=portfolio_payload,
            strategy_memory=strategy_memory,
            similar_cases=similar_cases,
        )

    def review_proposals(
        self,
        symbol: str,
        proposals: list[dict[str, Any]],
        features: dict[str, Any],
        similar_cases: list[dict[str, Any]],
        strategy_memory: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        reviewed_proposals = self.critic_agent.review(
            proposals=proposals,
            features=features,
            similar_cases=similar_cases,
            strategy_memory=strategy_memory,
        )
        if reviewed_proposals:
            return reviewed_proposals, reviewed_proposals[0]
        if proposals:
            return [proposals[0]], proposals[0]
        fallback = _fallback_hold_proposal(symbol, "No proposal generated; fallback to hold.")
        return [fallback], fallback

    def evaluate_risk(
        self,
        proposal: dict[str, Any],
        portfolio: Any,
        recent_memory: list[dict[str, Any]],
        strategy_memory: dict[str, Any],
    ) -> Any:
        return self.risk_manager.evaluate(
            proposal=proposal,
            portfolio=portfolio,
            recent_memory=recent_memory,
            strategy_memory=strategy_memory,
        )

    def make_final_decision(self, proposal: dict[str, Any], risk_report: Any) -> Any:
        final_decision = self.trader_agent.make_final_decision(
            proposal=proposal,
            risk_report=risk_report,
        )
        return self.position_sizer.apply(final_decision)

    def validate_decision(self, decision: Any, portfolio: Any, market_prices: dict[str, float]) -> Any:
        return self.validator.validate(
            decision=decision,
            portfolio=portfolio,
            market_prices=market_prices,
        )

    def execute_decision(self, decision: Any, portfolio: Any, market_prices: dict[str, float]) -> ExecutionResult:
        return self.executor.execute(
            decision=decision,
            portfolio=portfolio,
            market_prices=market_prices,
        )

    def build_rejected_execution(
        self,
        symbol: str,
        decision: Any,
        validation: Any,
        timestamp: str | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            symbol=symbol,
            timestamp=str(timestamp or getattr(decision, "timestamp", "") or utc_now_iso()),
            status="rejected",
            action=decision.action,
            message="; ".join(validation.errors),
            metadata={"validation_warnings": validation.warnings},
        )

    def apply_execution_and_snapshot(
        self,
        portfolio: Any,
        execution_result: ExecutionResult,
        market_prices: dict[str, float],
        updated_at: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        portfolio = self.portfolio_manager.apply_execution_result(
            portfolio,
            execution_result,
            updated_at=updated_at,
        )
        portfolio = self.portfolio_manager.mark_to_market(
            portfolio,
            market_prices,
            updated_at=updated_at,
        )
        self.portfolio_manager.save_portfolio(portfolio)
        snapshot = self.portfolio_manager.get_portfolio_snapshot(
            portfolio=portfolio,
        )
        return portfolio, snapshot

    def update_memory_skills(
        self,
        symbol: str,
        config: AppConfig,
        raw_context: RawContext,
        analyst_views: list[Any],
        proposal: dict[str, Any],
        risk_report: Any,
        final_decision: Any,
        execution_result: ExecutionResult,
        portfolio_snapshot: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        episode_record = build_episode_record(
            raw_context=raw_context,
            analyst_views=analyst_views,
            proposal=proposal,
            risk_report=risk_report,
            final_decision=final_decision,
            execution_result=execution_result,
            portfolio_snapshot=portfolio_snapshot,
        )
        self.episodic_memory.append_episode(episode_record)

        reflection_context = self.episodic_memory.load_recent(
            limit=config.memory.reflection_lookback_episodes,
            symbol=symbol,
        )
        reflection_note = self.reflection_engine.generate_reflection(
            episode=episode_record.to_dict(),
            recent_episodes=reflection_context,
        )
        self.reflection_engine.append_reflection(reflection_note)

        strategy_context = self.episodic_memory.load_recent(
            limit=config.memory.strategy_lookback_episodes,
            symbol=symbol,
        )
        updated_strategy_memory = self.strategic_memory_store.refresh_from_recent_episodes(strategy_context)
        updated_pattern_memory = self.pattern_memory_store.refresh_from_episodes(strategy_context)
        return reflection_note, updated_strategy_memory, updated_pattern_memory

    def assemble_cycle_result(
        self,
        symbol: str,
        config: AppConfig,
        execution_mode: str,
        portfolio_sync_error: str | None,
        raw_context: RawContext,
        analyst_views: list[Any],
        features: dict[str, Any],
        retrieval_context: dict[str, Any],
        proposals: list[dict[str, Any]],
        reviewed_proposals: list[dict[str, Any]],
        best_proposal: dict[str, Any],
        risk_report: Any,
        final_decision: Any,
        validation: Any,
        execution_result: ExecutionResult,
        portfolio_snapshot: dict[str, Any],
        reflection_note: Any,
        updated_strategy_memory: Any,
        updated_pattern_memory: dict[str, Any],
    ) -> dict[str, Any]:
        langsmith_cfg = getattr(config, "langsmith", None)
        langsmith_enabled = bool(getattr(langsmith_cfg, "enabled", False))
        langsmith_endpoint = str(getattr(langsmith_cfg, "endpoint", "") or "")
        langsmith_api_key = str(getattr(langsmith_cfg, "api_key", "") or "")
        langsmith_ready = bool(
            langsmith_enabled
            and (
                bool(langsmith_api_key.strip())
                or langsmith_endpoint.lower().startswith("http://localhost")
                or langsmith_endpoint.lower().startswith("http://127.0.0.1")
            )
        )
        result = {
            "symbol": symbol,
            "timestamp": raw_context.timestamp or utc_now_iso(),
            "llm": {
                "enabled": bool(config.llm.enabled),
                "available": bool(self.llm_client.enabled),
                "model": config.llm.model,
            },
            "observability": {
                "langsmith": {
                    "enabled": langsmith_enabled,
                    "ready": langsmith_ready,
                    "project": getattr(langsmith_cfg, "project", ""),
                    "endpoint": langsmith_endpoint,
                    "tags": list(getattr(langsmith_cfg, "tags", []) or []),
                }
            },
            "execution": {
                "mode": execution_mode,
                "okx_sandbox": bool(config.execution.okx_use_sandbox) if execution_mode == "okx" else None,
                "portfolio_sync_error": portfolio_sync_error,
            },
            "raw_context": raw_context.to_dict(),
            "analyst_views": [_as_dict(view) for view in analyst_views],
            "features": features,
            "retrieval_context": retrieval_context,
            "proposals": [_as_dict(item) for item in proposals],
            "reviewed_proposals": [_as_dict(item) for item in reviewed_proposals],
            "proposal": _as_dict(best_proposal),
            "risk_report": _as_dict(risk_report),
            "final_decision": _as_dict(final_decision),
            "validation": _as_dict(validation),
            "execution_result": _as_dict(execution_result),
            "portfolio_snapshot": portfolio_snapshot,
            "reflection_note": _as_dict(reflection_note),
            "strategy_memory": _as_dict(updated_strategy_memory),
            "pattern_memory": updated_pattern_memory,
        }
        self.trade_logger.append_cycle_summary(result)
        return result
