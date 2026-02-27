import json
import os
import requests
import ccxt
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORTFOLIO_FILE = "paper_portfolio.json"


# ==========================================
# 0. Virtual Portfolio Management
# ==========================================
def init_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        initial_state = {
            "USDT": 10000.0,  # Starting virtual balance: $10,000
            "holdings": {}
        }
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(initial_state, f, indent=4)
        print(f"[*] Initialized new paper trading portfolio with $10,000 USDT.")


def load_portfolio():
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)


def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ==========================================
# 1. Telegram Push Module
# ==========================================
def send_telegram_message(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[!] Telegram keys missing. Skipping push.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
        print("[+] Telegram push successful!")
    except Exception as e:
        print(f"[-] Telegram push failed: {e}")


# ==========================================
# 2. System Prompt (JSON Execution Mode)
# ==========================================
TRADER_SYSTEM_PROMPT = """
You are a ruthless Crypto Quantitative Trading Bot. Your sole purpose is to maximize profit and rigorously protect capital. You are NOT a passive long-term investor. You are an agile swing trader.

RULES:
1. You MUST use 'search_crypto_news' and 'get_crypto_price' before making a decision.
2. You MUST output your final decision ONLY as a valid JSON object. No markdown formatting, no explanations outside the JSON.
3. The JSON must exactly match this structure:
{
    "symbol": "BTC/USDT",
    "action": "BUY", // strictly "BUY", "SELL", or "HOLD"
    "amount_usdt": 500, // You MUST mathematically decide this number. Set to 0 if HOLD. If SELL, this is the USD value of the coin to dump.
    "reason": "Short explanation based on news and price data."
}

CRITICAL TRADING LOGIC:
- [TAKE PROFIT / STOP LOSS]: If you currently HOLD the asset and observe any of the following: 1) Price facing heavy resistance, 2) Bearish news, 3) Downward momentum in the last 5 hours, you MUST output 'SELL' to lock in profits or cut losses immediately.
- [AGGRESSIVE BUYING]: If you observe 1) Price bouncing strongly off a 3-Day Support level, 2) Major bullish news catalysts, OR 3) Strong upward volume and momentum breaking resistance, OR 4) Any other reason you think it is a good time to buy in the bottom, you MUST output 'BUY'. Do not be paralyzed by fear. Allocate a reasonable amount of your available USDT (e.g., 10% to 20%) to capture the trend.
- [DO NOT BE GREEDY]: It is better to SELL and hold USDT than to watch your portfolio bleed.
- [SCALING OUT]: You don't have to sell everything. You can SELL a portion (e.g., 30% or 50% of your holdings' USDT value) to manage risk.
"""


# ==========================================
# 3. Knowledge Base & Tools
# ==========================================
def init_knowledge_base():
    try:
        chroma_client = chromadb.PersistentClient(path="./news_db")
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
        return chroma_client.get_collection(name="crypto_news", embedding_function=embedding_fn)
    except Exception as e:
        print(f"[-] ChromaDB Error: {e}")
        return None


db_collection = init_knowledge_base()


def search_crypto_news(query: str, top_k: int = 10) -> str:
    print(f"[*] Tool executing: Searching news for '{query}'...")
    if not db_collection: return "DB not ready."

    # Increased top_k to 5 for better fundamental context
    results = db_collection.query(query_texts=[query], n_results=top_k)
    if results['documents'] and len(results['documents'][0]) > 0:
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])
    return "No recent news found."


def get_crypto_price(symbol: str) -> str:
    print(f"[*] Tool executing: Fetching 3-day 15m K-line data for {symbol}...")
    try:
        exchange = ccxt.binanceus({'enableRateLimit': True})

        # 1. Fetch 24H Ticker snapshot
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        pct_change = ticker['percentage']

        # 2. Fetch OHLCV: 3 days * 24 hours * 4 (15m intervals) = 288 candles
        # Format: [timestamp, open, high, low, close, volume]
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=288)

        if not ohlcv:
            return f"Current Price: {current_price}, 24H Change: {pct_change}%"

        # 3. Calculate 3-day Support and Resistance
        highs = [candle[2] for candle in ohlcv]
        lows = [candle[3] for candle in ohlcv]
        highest_3d = max(highs)
        lowest_3d = min(lows)

        # 4. Extract the most recent 5 candles to show immediate momentum
        recent_candles = ohlcv[-20:]
        recent_trend = "\n".join([
            f"  - Close: {c[4]}, Vol: {c[5]}" for c in recent_candles
        ])

        # 5. Fetch 1-Year Macro OHLCV (Daily candles)
        daily_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=365)
        if daily_ohlcv:
            yearly_high = max([candle[2] for candle in daily_ohlcv])
            yearly_low = min([candle[3] for candle in daily_ohlcv])
        else:
            yearly_high, yearly_low = "N/A", "N/A"

        report = (
            f"[{symbol} Market Data]\n"
            f"Current Price: {current_price}\n"
            f"24H Change: {pct_change}%\n"
            f"Macro 1-Year High: {yearly_high}\n"      
            f"Macro 1-Year Low: {yearly_low}\n"
            f"3-Day High (Resistance): {highest_3d}\n"
            f"3-Day Low (Support): {lowest_3d}\n"
            f"Recent 15m Candles (Last 5 hours):\n{recent_trend}"
        )
        return report

    except Exception as e:
        return f"Error fetching price: {e}"


def execute_paper_trade(decision_json: dict):
    print("\n[*] Processing Paper Trade Execution...")
    try:
        symbol = decision_json['symbol']
        action = decision_json['action'].upper()
        amount_usdt = float(decision_json['amount_usdt'])
        reason = decision_json['reason']

        if action == "HOLD" or amount_usdt <= 0:
            msg = f"⏸️ **ACTION: HOLD**\nSymbol: {symbol}\nReason: {reason}"
            print(msg)
            send_telegram_message(msg)
            return

        # Fetch actual current price to calculate coin amount
        exchange = ccxt.binanceus({'enableRateLimit': True})
        current_price = exchange.fetch_ticker(symbol)['last']
        coin_amount = amount_usdt / current_price

        portfolio = load_portfolio()
        base_coin = symbol.split('/')[0]  # e.g., 'BTC'

        msg = ""
        if action == "BUY":
            if portfolio["USDT"] >= amount_usdt:
                portfolio["USDT"] -= amount_usdt
                portfolio["holdings"][base_coin] = portfolio["holdings"].get(base_coin, 0) + coin_amount
                msg = f"🟢 **PAPER TRADE: BUY**\nSymbol: {symbol}\nSpent: ${amount_usdt}\nGot: {coin_amount:.6f} {base_coin}\nPrice: ${current_price}\nReason: {reason}\n💰 Remaining USDT: ${portfolio['USDT']:.2f}"
            else:
                msg = f"❌ **PAPER TRADE FAILED**\nInsufficient USDT balance. Needed: ${amount_usdt}, Have: ${portfolio['USDT']:.2f}"

        elif action == "SELL":
            current_holdings = portfolio["holdings"].get(base_coin, 0)

            # [HOTFIX] Slippage & Precision Tolerance
            # If the calculated coin_amount exceeds our holdings slightly,
            # it means the Agent wants to liquidate everything but got hit by a price drop.
            # We automatically adjust the sell amount to our maximum holdings.
            if coin_amount > current_holdings:
                print(
                    f"[*] Adjusting sell amount due to slippage. Original needed: {coin_amount:.6f}, Adjusted to max holding: {current_holdings:.6f}")
                coin_amount = current_holdings
                amount_usdt = coin_amount * current_price  # Recalculate the USD value obtained

            if current_holdings >= coin_amount and current_holdings > 0:
                portfolio["holdings"][base_coin] -= coin_amount
                portfolio["USDT"] += amount_usdt
                msg = f"🔴 **PAPER TRADE: SELL**\nSymbol: {symbol}\nSold: {coin_amount:.6f} {base_coin}\nGot: ${amount_usdt:.2f}\nPrice: ${current_price}\nReason: {reason}\n💰 Current USDT: ${portfolio['USDT']:.2f}"
            else:
                msg = f"❌ **PAPER TRADE FAILED**\nInsufficient {base_coin} balance. Needed: {coin_amount:.6f}, Have: {current_holdings:.6f}"

        save_portfolio(portfolio)
        print(msg)
        send_telegram_message(msg)

    except Exception as e:
        error_msg = f"[-] Error executing paper trade: {e}\nRaw JSON: {decision_json}"
        print(error_msg)
        send_telegram_message(error_msg)


# ==========================================
# 4. Agent Core Engine
# ==========================================
def run_trading_agent(symbol_input: str):
    init_portfolio()

    # 1. Read the current portfolio state BEFORE asking GPT
    portfolio = load_portfolio()
    base_coin = symbol_input.split('/')[0]  # e.g., 'BTC'
    current_holdings = portfolio["holdings"].get(base_coin, 0.0)
    current_usdt = portfolio["USDT"]

    # 2. Inject the portfolio data into the user trigger
    system_trigger = (
        f"Evaluate {symbol_input} and decide whether to BUY, SELL, or HOLD based on current news and price. Return strictly JSON.\n\n"
        f"--- CURRENT PORTFOLIO STATUS ---\n"
        f"Available USDT Balance: ${current_usdt:.2f}\n"
        f"Current {base_coin} Holdings: {current_holdings:.6f} {base_coin}\n"
        f"--------------------------------\n\n"
        f"CRITICAL RULE: You CANNOT SELL if your Current Holdings are 0. If you have holdings and the trend is bearish, you MUST consider SELL to stop loss or take profit."
    )

    print(f"\n[User] Command received with portfolio context...")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_crypto_news",
                "description": "Search local DB for crypto news.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_crypto_price",
                "description": "Fetch latest price for symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": TRADER_SYSTEM_PROMPT},
        {"role": "user", "content": system_trigger}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "search_crypto_news":
                result = search_crypto_news(query=args.get("query"))
            elif tool_call.function.name == "get_crypto_price":
                result = get_crypto_price(symbol=args.get("symbol"))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": result
            })

        print("[*] Agent thinking and formatting JSON decision...")
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}  # Force strict JSON output
        )
        final_text = final_response.choices[0].message.content
    else:
        final_text = response_message.content

    # Try to parse the final JSON and execute
    try:
        decision_json = json.loads(final_text)
        execute_paper_trade(decision_json)
    except json.JSONDecodeError:
        print(f"[-] Model failed to return valid JSON. Output was:\n{final_text}")


if __name__ == "__main__":
    print("=== Auto Paper Trading Agent Started ===")

    # List of coins you want the bot to trade automatically
    target_coins = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    for coin in target_coins:
        print(f"\n[*] Initiating automated analysis for {coin}...")
        try:
            run_trading_agent(coin)
        except Exception as e:
            print(f"[-] Critical error during {coin} analysis: {e}")

    print("\n=== Automated Trading Cycle Complete ===")