from __future__ import annotations

import os

from trading_agent_v2.mcp.server import run_mcp_server


def main() -> None:
    transport = os.getenv("TRADING_MCP_TRANSPORT", "stdio").strip().lower()
    run_mcp_server(transport=transport)


if __name__ == "__main__":
    main()
