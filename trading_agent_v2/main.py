from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
from pprint import pprint
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from trading_agent_v2.agents.market_analyst import MarketAnalyst
from trading_agent_v2.agents.news_analyst import NewsAnalyst
from trading_agent_v2.agents.risk_manager import RiskManager
from trading_agent_v2.agents.trader_agent import TraderAgent
from trading_agent_v2.config import AppConfig, build_default_config
from trading_agent_v2.execution.order_validator import OrderValidator
from trading_agent_v2.execution.paper_executor import PaperExecutor
from trading_agent_v2.execution.okx_executor import OkxExecutor
from trading_agent_v2.execution.position_sizer import PositionSizer
from trading_agent_v2.memory.episodic_memory import EpisodicMemoryStore
from trading_agent_v2.memory.pattern_memory import PatternMemoryStore
from trading_agent_v2.memory.retrieval import MemoryRetriever
from trading_agent_v2.memory.reflection_engine import ReflectionEngine
from trading_agent_v2.memory.strategic_memory import StrategicMemoryStore
from trading_agent_v2.llm.openai_client import OpenAIJsonClient
from trading_agent_v2.portfolio.portfolio_manager import PortfolioManager
from trading_agent_v2.portfolio.trade_logger import TradeLogger
from trading_agent_v2.schemas import ExecutionResult, RawContext, build_episode_record
from trading_agent_v2.tools.market_tools import MarketTools
from trading_agent_v2.tools.news_tools import NewsTools
from trading_agent_v2.tools.onchain_tools import OnchainTools
from trading_agent_v2.tools.social_tools import SocialTools
from trading_agent_v2.tools.feature_builder import FeatureBuilder
from trading_agent_v2.agents.planner_agent import PlannerAgent
from trading_agent_v2.agents.critic_agent import CriticAgent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def collect_raw_data(
    symbol: str,
    market_tools: MarketTools,
    news_tools: NewsTools,
    onchain_tools: OnchainTools,
    social_tools: SocialTools,
) -> RawContext:
    market_data = market_tools.get_market_snapshot(symbol)
    news_data = news_tools.get_latest_news(symbol, limit=3)
    news_summary = news_tools.summarize_sentiment(news_data)
    onchain_data = onchain_tools.get_onchain_snapshot(symbol)
    social_data = social_tools.get_social_snapshot(symbol, news_data=news_data)
    social_data.update(news_summary)

    return RawContext(
        symbol=symbol,
        timestamp=utc_now_iso(),
        market_data=market_data,
        news_data=news_data,
        onchain_data=onchain_data,
        social_data=social_data,
    )


def build_market_prices(raw_context: RawContext) -> dict[str, float]:
    price = float(raw_context.market_data.get("price", 0.0))
    return {raw_context.symbol: price}


def run_cycle(symbol: str = "BTC/USDT", app_config: AppConfig | None = None) -> dict:
    config = app_config or build_default_config()
    execution_mode = str(config.execution.mode or "paper").lower().strip()

    # managers
    market_tools = MarketTools(
        exchange_id="okx" if execution_mode == "okx" else "binanceus",
        fallback_exchange_id="coinbase",
    )
    news_tools = NewsTools()
    onchain_tools = OnchainTools()
    social_tools = SocialTools()

    market_analyst = MarketAnalyst()
    news_analyst = NewsAnalyst()

    feature_builder = FeatureBuilder()
    llm_client = OpenAIJsonClient(
        enabled=config.llm.enabled,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout_sec=config.llm.timeout_sec,
    )
    planner_agent = PlannerAgent(
        action_threshold=config.planner.action_threshold,
        min_trade_confidence=config.planner.min_trade_confidence,
        min_directional_size_pct=config.planner.min_directional_size_pct,
        max_directional_size_pct=config.planner.max_directional_size_pct,
        aggressive_size_multiplier=config.planner.aggressive_size_multiplier,
        llm_client=llm_client,
        llm_primary=config.llm.enabled,
    )
    critic_agent = CriticAgent(
        llm_client=llm_client,
        llm_primary=config.llm.enabled,
    )
    memory_retriever = MemoryRetriever()

    portfolio_manager = PortfolioManager(str(config.portfolio_file))
    episodic_memory = EpisodicMemoryStore(str(config.episodic_memory_file))
    reflection_engine = ReflectionEngine(str(config.reflection_file))
    strategic_memory_store = StrategicMemoryStore(str(config.strategy_memory_file))
    pattern_memory_store = PatternMemoryStore(str(config.pattern_memory_file))
    trade_logger = TradeLogger(str(config.run_log_file))

    trader_agent = TraderAgent(
        buy_threshold=config.trader.buy_threshold,
        sell_threshold=config.trader.sell_threshold,
        min_confidence=config.trader.min_confidence,
        default_buy_size_pct=config.trader.default_buy_size_pct,
        default_sell_size_pct=config.trader.default_sell_size_pct,
        min_directional_size_pct=config.trader.min_directional_size_pct,
        default_stop_loss_pct=config.trader.default_stop_loss_pct,
        default_take_profit_pct=config.trader.default_take_profit_pct,
        analyst_weights=config.trader.analyst_weights,
        llm_client=llm_client,
        llm_primary=config.llm.enabled,
    )
    risk_manager = RiskManager(
        max_total_invested_pct=config.risk.max_total_invested_pct,
        max_single_position_pct=config.risk.max_single_position_pct,
        min_proposal_confidence=config.risk.min_proposal_confidence,
        max_conflicting_factors=config.risk.max_conflicting_factors,
        max_loss_streak=config.risk.max_loss_streak,
        volatility_size_cut_ratio=config.risk.volatility_size_cut_ratio,
        high_risk_score_threshold=config.risk.high_risk_score_threshold,
        medium_risk_score_threshold=config.risk.medium_risk_score_threshold,
        elevated_risk_size_multiplier=config.risk.elevated_risk_size_multiplier,
        conflict_size_cut_ratio=config.risk.conflict_size_cut_ratio,
        volatility_stop_loss_tighten_ratio=config.risk.volatility_stop_loss_tighten_ratio,
    )
    validator = OrderValidator(
        min_size_pct=config.validation.min_size_pct,
        max_size_pct=config.validation.max_size_pct,
        allow_short=config.validation.allow_short,
    )
    position_sizer = PositionSizer(
        min_size_pct=config.validation.min_size_pct,
        max_size_pct=config.validation.max_size_pct,
        enforce_min_size=config.execution.enforce_min_size,
    )
    if execution_mode == "okx":
        executor = OkxExecutor(
            api_key=config.execution.okx_api_key,
            secret=config.execution.okx_secret,
            passphrase=config.execution.okx_passphrase,
            use_sandbox=config.execution.okx_use_sandbox,
            timeout_ms=config.execution.okx_timeout_ms,
            enable_rate_limit=config.execution.okx_enable_rate_limit,
        )
    elif execution_mode == "paper":
        executor = PaperExecutor(
            trading_fee_rate=config.execution.trading_fee_rate,
            slippage_rate=config.execution.slippage_rate,
        )
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    # state
    portfolio_manager.ensure_portfolio_exists(initial_cash=config.initial_cash)
    portfolio = portfolio_manager.load_portfolio()

    recent_memory = episodic_memory.load_recent(
        limit=config.memory.recent_episodes,
        symbol=symbol,
    )
    strategy_memory_obj = strategic_memory_store.load()
    strategy_memory = strategy_memory_obj.to_dict()
    pattern_memory = pattern_memory_store.load()
    strategy_memory["pattern_insights"] = (pattern_memory.get("metadata") or {}).get("insights", [])
    strategy_memory["pattern_stats"] = pattern_memory.get("patterns", {})

    # data collection
    raw_context = collect_raw_data(
        symbol=symbol,
        market_tools=market_tools,
        news_tools=news_tools,
        onchain_tools=onchain_tools,
        social_tools=social_tools,
    )
    market_prices = build_market_prices(raw_context)

    portfolio_sync_error = None
    if execution_mode == "okx":
        try:
            portfolio = executor.sync_portfolio_state(
                portfolio=portfolio,
                symbols=config.symbols or [symbol],
                market_prices=market_prices,
            )
            portfolio_manager.save_portfolio(portfolio)
        except Exception as exc:
            portfolio_sync_error = str(exc)

    # mark-to-market
    portfolio = portfolio_manager.mark_to_market(portfolio, market_prices)

    # unified features
    features = feature_builder.build(
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

    # analyst layer
    market_view = market_analyst.analyze(raw_context)
    news_view = news_analyst.analyze(raw_context)
    analyst_views = [market_view, news_view]

    # planner -> critic
    retrieval_context = memory_retriever.build_context_bundle(
        features=features,
        episodes=recent_memory,
        symbol=symbol,
        similar_k=5,
        recent_failure_k=3,
    )
    similar_cases = retrieval_context.get("similar_cases", [])

    proposals = planner_agent.generate_proposals(
        symbol=symbol,
        analyst_views=analyst_views,
        features=features,
        portfolio=portfolio.to_dict(),
        strategy_memory=strategy_memory,
        similar_cases=similar_cases,
    )

    reviewed_proposals = critic_agent.review(
        proposals=proposals,
        features=features,
        similar_cases=similar_cases,
        strategy_memory=strategy_memory,
    )

    best_proposal = reviewed_proposals[0]

    # risk
    risk_report = risk_manager.evaluate(
        proposal=best_proposal,
        portfolio=portfolio,
        recent_memory=recent_memory,
        strategy_memory=strategy_memory,
    )

    # final decision
    final_decision = trader_agent.make_final_decision(
        proposal=best_proposal,
        risk_report=risk_report,
    )
    final_decision = position_sizer.apply(final_decision)

    # validation
    validation = validator.validate(
        decision=final_decision,
        portfolio=portfolio,
        market_prices=market_prices,
    )

    # execution
    if validation.valid:
        execution_result = executor.execute(
            decision=final_decision,
            portfolio=portfolio,
            market_prices=market_prices,
        )
    else:
        execution_result = ExecutionResult(
            symbol=symbol,
            timestamp=utc_now_iso(),
            status="rejected",
            action=final_decision.action,
            message="; ".join(validation.errors),
            metadata={
                "validation_warnings": validation.warnings,
            },
        )

    # portfolio updates
    portfolio = portfolio_manager.apply_execution_result(portfolio, execution_result)
    portfolio = portfolio_manager.mark_to_market(portfolio, market_prices)
    portfolio_manager.save_portfolio(portfolio)

    snapshot = portfolio_manager.get_portfolio_snapshot(
        portfolio=portfolio,
        market_prices=market_prices,
    )

    # memory / reflection
    episode_record = build_episode_record(
        raw_context=raw_context,
        analyst_views=analyst_views,
        proposal=best_proposal,
        risk_report=risk_report,
        final_decision=final_decision,
        execution_result=execution_result,
        portfolio_snapshot=snapshot,
    )
    episodic_memory.append_episode(episode_record)

    reflection_context = episodic_memory.load_recent(
        limit=config.memory.reflection_lookback_episodes,
        symbol=symbol,
    )
    reflection_note = reflection_engine.generate_reflection(
        episode=episode_record.to_dict(),
        recent_episodes=reflection_context,
    )
    reflection_engine.append_reflection(reflection_note)

    strategy_context = episodic_memory.load_recent(
        limit=config.memory.strategy_lookback_episodes,
        symbol=symbol,
    )
    updated_strategy_memory = strategic_memory_store.refresh_from_recent_episodes(strategy_context)
    updated_pattern_memory = pattern_memory_store.refresh_from_episodes(strategy_context)

    # result bundle
    result = {
        "symbol": symbol,
        "timestamp": utc_now_iso(),
        "llm": {
            "enabled": bool(config.llm.enabled),
            "available": bool(llm_client.enabled),
            "model": config.llm.model,
        },
        "execution": {
            "mode": execution_mode,
            "okx_sandbox": bool(config.execution.okx_use_sandbox) if execution_mode == "okx" else None,
            "portfolio_sync_error": portfolio_sync_error,
        },
        "raw_context": raw_context.to_dict(),
        "analyst_views": [view.to_dict() for view in analyst_views],
        "features": features,
        "retrieval_context": retrieval_context,
        "proposals": [p.to_dict() if hasattr(p, "to_dict") else p for p in proposals],
        "reviewed_proposals": [p.to_dict() if hasattr(p, "to_dict") else p for p in reviewed_proposals],
        "proposal": best_proposal.to_dict() if hasattr(best_proposal, "to_dict") else best_proposal,
        "risk_report": risk_report.to_dict(),
        "final_decision": final_decision.to_dict(),
        "validation": validation.to_dict(),
        "execution_result": execution_result.to_dict(),
        "portfolio_snapshot": snapshot,
        "reflection_note": reflection_note.to_dict(),
        "strategy_memory": updated_strategy_memory.to_dict(),
        "pattern_memory": updated_pattern_memory,
    }
    trade_logger.append_cycle_summary(result)

    return result


def run_batch(symbols: list[str] | None = None, app_config: AppConfig | None = None) -> list[dict]:
    config = app_config or build_default_config()
    target_symbols = symbols or config.symbols

    results: list[dict] = []
    for symbol in target_symbols:
        results.append(run_cycle(symbol=symbol, app_config=config))
    return results


def main() -> None:
    config = build_default_config()
    results = run_batch(app_config=config)

    print("\n" + "=" * 88)
    print("TRADE SYSTEM RUN BATCH RESULT")
    print("=" * 88)

    for idx, result in enumerate(results, start=1):
        print(f"\n[{idx}] Symbol: {result['symbol']}")
        print("-" * 88)

        print("\n[Features]")
        pprint(result["features"])

        print("\n[All Proposals]")
        pprint(result["proposals"])

        print("\n[Reviewed Proposals]")
        pprint(result["reviewed_proposals"])

        print("\n[Risk Report]")
        pprint(result["risk_report"])

        print("\n[Final Decision]")
        pprint(result["final_decision"])

        print("\n[Validation]")
        pprint(result["validation"])

        print("\n[Execution Result]")
        pprint(result["execution_result"])

        print("\n[Reflection Note]")
        pprint(result["reflection_note"])

        print("\n[Portfolio Snapshot]")
        pprint(result["portfolio_snapshot"])

    print("\nBatch done.")


if __name__ == "__main__":
    main()
