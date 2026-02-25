import json
import os
import requests
import ccxt
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI()

# Load Telegram keys
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ==========================================
# 0. Telegram Push Module
# ==========================================
def send_telegram_message(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("️ Telegram keys not configured. Skipping push notification.")
        return

    print(" Pushing report to Telegram...")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
        print(" Push successful!")
    except Exception as e:
        print(f" Push failed: {e}")


# ==========================================
# 1. System Prompt (Crypto Quant Analyst)
# ==========================================
TRADER_SYSTEM_PROMPT = """
You are a top-tier Crypto Quantitative Analyst and Trading Strategy Expert.
Your task is to combine the latest market news (fundamentals) and real-time crypto prices (technicals) to provide a detailed, in-depth investment analysis report.

Core principles:
1. Dual Verification: Always call both `search_crypto_news` and `get_crypto_price`. Do not fabricate data.
2. Structured Output (Use Markdown for readability):
   -  **[Core Conclusion]**: One sentence summarizing the bullish/bearish stance.
   -  **[Market Data Analysis]**: List current price & 24H change, assess short-term momentum.
   -  **[News & Fundamentals]**: Deep dive into the retrieved news and its impact on the asset.
   -  **[Strategy & Outlook]**: Project potential market moves and specific price action levels to watch.
3. Professional & Objective: Use a professional, cold tone. Refuse emotional vocabulary.
4. Compliance: Always end with: "*Crypto markets are highly volatile. This analysis is for paper-trading reference only and does not constitute financial advice.*"
"""


# ==========================================
# 2. Knowledge Base Connection
# ==========================================
def init_knowledge_base():
    try:
        chroma_client = chromadb.PersistentClient(path="./news_db")
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
        collection = chroma_client.get_collection(name="crypto_news", embedding_function=embedding_fn)
        return collection
    except Exception as e:
        print(f"Failed to connect to Knowledge Base. Error: {e}")
        return None


db_collection = init_knowledge_base()


# ==========================================
# 3. Agent Skills (Tools)
# ==========================================
def search_crypto_news(query: str, top_k: int = 3) -> str:
    print(f"\n[Agent Thinking] Querying local knowledge base for '{query}'...")
    if not db_collection:
        return "Local knowledge base is not ready."
    results = db_collection.query(query_texts=[query], n_results=top_k)
    if results['documents'] and len(results['documents'][0]) > 0:
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])
    return f"No recent news found for '{query}' in the current database."


def get_crypto_price(symbol: str) -> str:
    print(f"[Agent Thinking] Fetching live market data for {symbol} via Binance.US...")
    try:
        # Using binanceus to bypass geo-restrictions
        exchange = ccxt.binanceus({'enableRateLimit': True})
        ticker = exchange.fetch_ticker(symbol)

        current_price = ticker['last']
        pct_change = ticker['percentage']

        return f"{symbol} Current Price: $ {current_price}, 24H Change: {pct_change:.2f}%"
    except Exception as e:
        return f"Failed to fetch price for {symbol}: {str(e)}"


# ==========================================
# 4. Agent Core Engine
# ==========================================
def run_trading_agent(user_message: str):
    print(f"\n[User Command] {user_message}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_crypto_news",
                "description": "Search the local vector database for the latest crypto market news.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Search keyword (e.g., 'Bitcoin', 'Ethereum', 'SEC')"},
                        "top_k": {"type": "integer", "description": "Number of results to return, default 3"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_crypto_price",
                "description": "Fetch the latest price and 24H change for a given cryptocurrency trading pair.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string",
                                   "description": "Trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDT')"}
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": TRADER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message

    final_text = ""
    if response_message.tool_calls:
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "search_crypto_news":
                result = search_crypto_news(query=args.get("query"), top_k=args.get("top_k", 3))
            elif tool_call.function.name == "get_crypto_price":
                result = get_crypto_price(symbol=args.get("symbol"))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": result
            })

        print("[Agent Thinking] Analysis complete. Generating final report...")
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        final_text = final_response.choices[0].message.content
    else:
        final_text = response_message.content

    print(f"\n[GPT Quant Analyst]\n{final_text}")
    send_telegram_message(final_text)


if __name__ == "__main__":
    print("=== Trading System Started (Crypto Mode + Telegram Push) ===")
    while True:
        user_input = input("\nWhich crypto pair would you like to analyze? (Enter 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        if user_input.strip():
            run_trading_agent(user_input)