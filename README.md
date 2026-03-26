# TradeSystem

`trading_agent_v2` is a multi-agent crypto trading system built around LangGraph. It combines market data, news, on-chain proxy signals, social/community signals, portfolio state, memory, and LLM reasoning to generate trade proposals, review them, apply risk constraints, and execute on OKX.

This repository currently contains:

- `trading_agent_v2`: the current LangGraph-based workflow

This README focuses on `trading_agent_v2`.


## Workflow Overview

For each symbol, the system runs a full cycle:

1. Build runtime and load portfolio/memory
2. Collect market, news, on-chain, and social raw context
3. Sync or mark-to-market the portfolio
4. Build normalized features
5. Run analyst agents
6. Generate proposals
7. Review proposals
8. Evaluate risk
9. Make final decision
10. Validate order
11. Execute or reject
12. Update portfolio and memories
13. Persist run summary

The main workflow lives in `trading_agent_v2/workflows/langgraph_workflow.py`.

## Data Sources

`trading_agent_v2` uses public and exchange-backed data with fallback behavior:

- Market data:
  - primary via `ccxt`
  - fallback via `yfinance`
  - final fallback via deterministic synthetic market snapshot
- News:
  - primary via CryptoCompare
  - fallback via CoinDesk / CoinTelegraph RSS
- On-chain proxy signals:
  - CoinGecko market endpoints
- Social/community signals:
  - CoinGecko community endpoints
  - fallback derived from news sentiment/activity

## Project Structure

```text
tradesystem/
|-- trading_agent_v1/
|-- trading_agent_v2/
|   |-- agents/          # analyst, planner, critic, risk, trader agents
|   |-- data/            # portfolio, memories, run logs, ablation outputs
|   |-- evaluation/      # backtest and ablation utilities
|   |-- execution/       # paper and OKX executors
|   |-- llm/             # OpenAI client wrapper
|   |-- mcp/             # MCP server entrypoints and tools
|   |-- memory/          # episodic, strategic, reflection, retrieval
|   |-- portfolio/       # portfolio state and logging
|   |-- prompts/         # prompt templates
|   |-- skills/          # workflow integration layer
|   |-- tools/           # market/news/on-chain/social feature tools
|   |-- workflows/       # LangGraph workflow
|   |-- config.py        # central configuration
|   `-- main.py          # main run entrypoint
|-- requirements.txt
`-- README.md
```

## Installation

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

## Quick Start

### Minimal `.env` example

Create a root-level `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key

TRADING_EXECUTION_MODE=paper
TRADING_OKX_USE_SANDBOX=true

TRADING_LLM_ENABLED=true
TRADING_LLM_MODEL=o3
TRADING_LLM_TEMPERATURE=0.2
TRADING_LLM_MAX_TOKENS=1200
TRADING_LLM_TIMEOUT_SEC=30

TRADING_LANGSMITH_ENABLED=false
TRADING_LANGSMITH_PROJECT=trading-agent-v2
TRADING_LANGSMITH_TAGS=local,v2
TRADING_LANGSMITH_ENDPOINT=https://api.smith.langchain.com
TRADING_LANGSMITH_API_KEY=

# Optional for live OKX mode
OKX_API_KEY=
OKX_SECRET_KEY=
OKX_PASSPHRASE=

# Optional news provider enhancement
CRYPTOCOMPARE_API_KEY=

# Optional for MCP server
TRADING_MCP_TRANSPORT=stdio
```

## Running the Main Workflow

### Run the default batch

This uses `config.symbols`, which currently defaults to `["BTC/USDT"]`.

```powershell
python trading_agent_v2/main.py
```

## Configuration

The main config builder is `trading_agent_v2/config.py::build_default_config`.

### Core environment variables

- `OPENAI_API_KEY`: required when `TRADING_LLM_ENABLED=true`
- `TRADING_EXECUTION_MODE`: `paper` or `okx`
- `TRADING_OKX_USE_SANDBOX`: `true` / `false`
- `OKX_API_KEY`
- `OKX_SECRET_KEY`
- `OKX_PASSPHRASE`
- `TRADING_LLM_ENABLED`
- `TRADING_LLM_MODEL`
- `TRADING_LLM_TEMPERATURE`
- `TRADING_LLM_MAX_TOKENS`
- `TRADING_LLM_TIMEOUT_SEC`

### Planner tuning

- `TRADING_PLANNER_ACTION_THRESHOLD`
- `TRADING_PLANNER_MIN_TRADE_CONFIDENCE`
- `TRADING_PLANNER_MIN_DIRECTIONAL_SIZE_PCT`
- `TRADING_PLANNER_MAX_DIRECTIONAL_SIZE_PCT`
- `TRADING_PLANNER_AGGRESSIVE_SIZE_MULTIPLIER`

### LangSmith

- `TRADING_LANGSMITH_ENABLED`
- `TRADING_LANGSMITH_PROJECT`
- `TRADING_LANGSMITH_ENDPOINT`
- `TRADING_LANGSMITH_API_KEY`
- `TRADING_LANGSMITH_TAGS`

### MCP

- `TRADING_MCP_TRANSPORT`: `stdio`, `sse`, or `streamable-http`

## LangSmith Tracing

This version supports LangSmith tracing for the LangGraph workflow and OpenAI client calls.

To enable it:

```env
TRADING_LANGSMITH_ENABLED=true
TRADING_LANGSMITH_API_KEY=your_langsmith_api_key
TRADING_LANGSMITH_PROJECT=trading-agent-v2
TRADING_LANGSMITH_TAGS=local,v2
```

Notes:

- LangSmith environment variables are bootstrapped from the `TRADING_LANGSMITH_*` namespace at startup
- Traces are tagged with the project tags plus a symbol tag like `symbol:BTC_USDT`
- Sensitive runtime objects are not passed into traced graph state in the current version

## Outputs and Persistence

By default, `trading_agent_v2` writes state into `trading_agent_v2/data/`:

- `paper_portfolio.json`: current paper portfolio
- `trade_history.jsonl`: episodic memory
- `reflections.jsonl`: reflection notes
- `strategy_memory.json`: strategic memory
- `pattern_memory.json`: pattern memory
- `run_log.jsonl`: cycle summaries
- `ablation/`: backtest/ablation outputs

## MCP Server

Run the MCP server:

```powershell
python -m trading_agent_v2.mcp
```

Available tool families include:

- `market_snapshot`
- `latest_news`
- `news_sentiment_summary`
- `onchain_snapshot`
- `social_snapshot`
- `raw_context_bundle`
- `feature_bundle`

## Backtesting

Example:

```python
from trading_agent_v2.evaluation.backtest_engine import BacktestEngine
from trading_agent_v2.config import build_default_config

config = build_default_config()
config.execution.mode = "paper"

engine = BacktestEngine()
result = engine.run(
    cycles=10,
    symbols=["BTC/USDT"],
    app_config=config,
    data_dir="trading_agent_v2/data/backtest_baseline",
    config_name="baseline",
)

print(result.to_dict())
```

Tracked metrics include:

- cycle count
- fill rate
- win rate
- average risk score
- average size
- realized PnL change
- equity change
- max drawdown

## Ablation Studies

Example:

```python
from trading_agent_v2.evaluation.ablation_runner import AblationRunner
from trading_agent_v2.config import build_default_config

config = build_default_config()
config.execution.mode = "paper"

runner = AblationRunner()
summary = runner.run(
    cycles=10,
    symbols=["BTC/USDT"],
    base_config=config,
    output_root="trading_agent_v2/data/ablation",
)

print(summary["ranking"])
```

Current ablation variants:

- `baseline`
- `no_news_weight`
- `no_market_weight`
- `tight_risk`
- `loose_risk`

## Known Limitations

- Backtest here is a repeated live-cycle replay wrapper, not a historical candle simulator
- Public data APIs can rate-limit or fail, causing fallback behavior
- The system currently focuses on spot-style flows and does not enable shorting by default
- Real execution quality depends on exchange connectivity, symbol availability, balances, and minimum order constraints

## Recommended First Validation Flow

1. Create `.env` with `TRADING_EXECUTION_MODE=paper`
2. Set `OPENAI_API_KEY`
3. Optionally enable LangSmith
4. Run `python trading_agent_v2/main.py`
5. Inspect the printed result and `trading_agent_v2/data/run_log.jsonl`
6. Only after validation, configure OKX credentials and switch to `okx`
