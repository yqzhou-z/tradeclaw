from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

from trading_agent_v2.config import AppConfig, build_default_config
from trading_agent_v2.tools.feature_builder import FeatureBuilder
from trading_agent_v2.tools.market_tools import MarketTools
from trading_agent_v2.tools.news_tools import NewsTools
from trading_agent_v2.tools.onchain_tools import OnchainTools
from trading_agent_v2.tools.social_tools import SocialTools


@dataclass
class MCPToolRuntime:
    market_tools: MarketTools
    news_tools: NewsTools
    onchain_tools: OnchainTools
    social_tools: SocialTools
    feature_builder: FeatureBuilder


def _build_runtime(config: AppConfig) -> MCPToolRuntime:
    execution_mode = str(config.execution.mode or "paper").lower().strip()
    market_tools = MarketTools(
        exchange_id="okx" if execution_mode == "okx" else "binanceus",
        fallback_exchange_id="coinbase",
    )
    news_tools = NewsTools()
    onchain_tools = OnchainTools()
    social_tools = SocialTools()
    feature_builder = FeatureBuilder()

    return MCPToolRuntime(
        market_tools=market_tools,
        news_tools=news_tools,
        onchain_tools=onchain_tools,
        social_tools=social_tools,
        feature_builder=feature_builder,
    )


def _build_raw_context(runtime: MCPToolRuntime, symbol: str) -> dict[str, Any]:
    market_data = runtime.market_tools.get_market_snapshot(symbol)
    news_data = runtime.news_tools.get_latest_news(symbol, limit=3)
    news_summary = runtime.news_tools.summarize_sentiment(news_data)
    onchain_data = runtime.onchain_tools.get_onchain_snapshot(symbol)
    social_data = runtime.social_tools.get_social_snapshot(symbol=symbol, news_data=news_data)
    social_data.update(news_summary)
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "market_data": market_data,
        "news_data": news_data,
        "onchain_data": onchain_data,
        "social_data": social_data,
    }


def create_mcp_server(app_config: AppConfig | None = None) -> FastMCP:
    config = app_config or build_default_config()
    runtime = _build_runtime(config)

    mcp = FastMCP(
        name="trading-agent-v2-tools",
        instructions=(
            "MCP tools for trading-agent-v2 market/news/on-chain/social feature building. "
            "Use these tools to fetch snapshots or construct a unified feature bundle."
        ),
    )

    @mcp.tool(
        name="market_snapshot",
        description="Fetch latest market snapshot for a symbol, e.g. BTC/USDT.",
    )
    def market_snapshot(symbol: str) -> dict[str, Any]:
        return runtime.market_tools.get_market_snapshot(symbol)

    @mcp.tool(
        name="latest_news",
        description="Fetch latest symbol-related news list.",
    )
    def latest_news(symbol: str, limit: int = 3) -> list[dict[str, Any]]:
        safe_limit = max(1, min(20, int(limit)))
        return runtime.news_tools.get_latest_news(symbol=symbol, limit=safe_limit)

    @mcp.tool(
        name="news_sentiment_summary",
        description="Build sentiment summary from latest news.",
    )
    def news_sentiment_summary(symbol: str, limit: int = 3) -> dict[str, Any]:
        safe_limit = max(1, min(20, int(limit)))
        news_items = runtime.news_tools.get_latest_news(symbol=symbol, limit=safe_limit)
        return runtime.news_tools.summarize_sentiment(news_items)

    @mcp.tool(
        name="onchain_snapshot",
        description="Fetch on-chain proxy snapshot for a symbol.",
    )
    def onchain_snapshot(symbol: str) -> dict[str, Any]:
        return runtime.onchain_tools.get_onchain_snapshot(symbol)

    @mcp.tool(
        name="social_snapshot",
        description="Fetch social/community snapshot for a symbol.",
    )
    def social_snapshot(symbol: str, limit: int = 3) -> dict[str, Any]:
        safe_limit = max(1, min(20, int(limit)))
        news_items = runtime.news_tools.get_latest_news(symbol=symbol, limit=safe_limit)
        return runtime.social_tools.get_social_snapshot(symbol=symbol, news_data=news_items)

    @mcp.tool(
        name="raw_context_bundle",
        description="Fetch unified raw_context bundle (market/news/onchain/social).",
    )
    def raw_context_bundle(symbol: str) -> dict[str, Any]:
        return _build_raw_context(runtime=runtime, symbol=symbol)

    @mcp.tool(
        name="feature_bundle",
        description="Fetch raw_context and build normalized features for downstream agents.",
    )
    def feature_bundle(symbol: str) -> dict[str, Any]:
        raw_context = _build_raw_context(runtime=runtime, symbol=symbol)
        features = runtime.feature_builder.build(
            symbol=symbol,
            market_data=raw_context["market_data"],
            news_data={
                "items": raw_context["news_data"],
                "summary": raw_context["social_data"].get("summary"),
                "sentiment": raw_context["social_data"].get("sentiment_score"),
            },
            onchain_data=raw_context["onchain_data"],
            social_data=raw_context["social_data"],
        )
        return {
            "symbol": symbol,
            "raw_context": raw_context,
            "features": features,
        }

    return mcp


def run_mcp_server(transport: str = "stdio", app_config: AppConfig | None = None) -> None:
    normalized_transport = str(transport).lower().strip()
    if normalized_transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(f"Unsupported MCP transport: {transport}")
    server = create_mcp_server(app_config=app_config)
    server.run(transport=normalized_transport)
