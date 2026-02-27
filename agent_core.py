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
You are an autonomous Crypto Quantitative Trading Bot.
Your goal is to analyze the provided cryptocurrency, fetch its latest news and price, and make a trading decision.

RULES:
1. You MUST use 'search_crypto_news' and 'get_crypto_price' before making a decision.
2. You MUST output your final decision ONLY as a valid JSON object. No markdown formatting, no explanations outside the JSON.
3. The JSON must exactly match this structure:
{
    "symbol": "BTC/USDT",
    "action": "BUY", // strictly "BUY", "SELL", or "HOLD"
    "amount_usdt": 500, // Amount of USDT to spend (if BUY) or value to sell (if SELL). Set to 0 if HOLD.
    "reason": "Short explanation based on news and price data."
}
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


def search_crypto_news(query: str, top_k: int = 3) -> str:
    print(f"[*] Tool executing: Searching news for '{query}'...")
    if not db_collection: return "DB not ready."
    results = db_collection.query(query_texts=[query], n_results=top_k)
    if results['documents'] and len(results['documents'][0]) > 0:
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])
    return "No recent news found."


def get_crypto_price(symbol: str) -> str:
    print(f"[*] Tool executing: Fetching price for {symbol}...")
    try:
        exchange = ccxt.binanceus({'enableRateLimit': True})
        ticker = exchange.fetch_ticker(symbol)
        return f"Current Price: {ticker['last']}, 24H Change: {ticker['percentage']}%"
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
            if current_holdings >= coin_amount:
                portfolio["holdings"][base_coin] -= coin_amount
                portfolio["USDT"] += amount_usdt
                msg = f"🔴 **PAPER TRADE: SELL**\nSymbol: {symbol}\nSold: {coin_amount:.6f} {base_coin}\nGot: ${amount_usdt}\nPrice: ${current_price}\nReason: {reason}\n💰 Current USDT: ${portfolio['USDT']:.2f}"
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

    # Force the user input to be a trading request
    system_trigger = f"Evaluate {symbol_input} and decide whether to BUY, SELL, or HOLD based on current news and price. Return strictly JSON."
    print(f"\n[User] Command received: {system_trigger}")

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