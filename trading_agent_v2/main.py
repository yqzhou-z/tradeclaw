from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys

from dotenv import load_dotenv

load_dotenv()

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from trading_agent_v2.config import AppConfig, bootstrap_langsmith_env, build_default_config

bootstrap_langsmith_env()

from trading_agent_v2.workflows.langgraph_workflow import (
    run_batch_with_langgraph,
    run_cycle_with_langgraph,
)


def run_cycle(symbol: str = "BTC/USDT", app_config: AppConfig | None = None) -> dict:
    return run_cycle_with_langgraph(symbol=symbol, app_config=app_config)


def run_batch(symbols: list[str] | None = None, app_config: AppConfig | None = None) -> list[dict]:
    return run_batch_with_langgraph(symbols=symbols, app_config=app_config)


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
