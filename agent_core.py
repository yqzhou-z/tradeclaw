import json
import os
import yfinance as yf
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
client = OpenAI()

# ==========================================
# 1. 赋予灵魂：System Prompt (全面升级版)
# ==========================================
TRADER_SYSTEM_PROMPT = """
你现在是一位顶级的A股量化分析师和交易策略专家。
你的任务是结合最新的市场快讯（消息面）和股票的实时盘面价格（技术面），为用户出具一份详尽、深度的投资分析研报。

你的核心工作原则：
1. 数据双重验证：分析任何个股时，必须同时调用 `search_market_news` 和 `get_stock_price` 获取最新数据。绝不凭空捏造。
2. 结构化深度输出：你的分析报告必须详细且具有逻辑性，请严格按照以下结构进行输出：
   - 【核心结论】：用一句话总结当前该标的的多空态势。
   - 【盘面数据解析】：客观列出当前价格和涨跌幅，并简要评估短期的资金情绪。
   - 【消息面深度推演】：不仅要复述检索到的核心新闻，更要深度剖析该事件对该公司的基本面、产业链或所属板块的实质性影响（利好/利空级别，以及发酵周期）。
   - 【后市策略展望】：结合盘面价格和消息面，推演主力可能的资金意图，并给出具体的观察节点。
3. 专业冷酷：语言风格要专业、客观，拒绝使用情绪化词汇。逻辑推演过程必须完整详实，不要为了简短而牺牲分析深度。
4. 合规底线：永远在回答的最后附上一句免责声明：“股市有风险，投资需谨慎，本建议仅供策略回测参考，不构成实盘交易建议。”
"""


# ==========================================
# 2. 初始化知识库连接
# ==========================================
def init_knowledge_base():
    try:
        chroma_client = chromadb.PersistentClient(path="./news_db")
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
        collection = chroma_client.get_collection(name="a_share_news", embedding_function=embedding_fn)
        return collection
    except Exception as e:
        print(f"知识库连接失败，请确保 news_db 存在。报错: {e}")
        return None


db_collection = init_knowledge_base()


# ==========================================
# 3. 打造武器库：Agent 可调用的 Skills
# ==========================================
def search_market_news(query: str, top_k: int = 3) -> str:
    """工具1：检索本地新闻库"""
    print(f"\n[Agent 思考中] 正在查阅知识库中关于 '{query}' 的最新情报...")
    if not db_collection:
        return "本地知识库未准备好，无法检索。"
    results = db_collection.query(query_texts=[query], n_results=top_k)
    if results['documents'] and len(results['documents'][0]) > 0:
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])
    return f"当前数据库中没有找到与 '{query}' 相关的最新新闻。"


def get_stock_price(ticker: str) -> str:
    """
    【工具说明书】
    功能：获取指定股票的最新盘面价格和今日涨跌幅。
    参数:
    - ticker (str): 股票代码。A股请务必带上后缀，例如沪市用 "600519.SS" (贵州茅台)，深市用 "002594.SZ" (比亚迪)。
    """
    print(f"[Agent 思考中] 正在拉取终端实时盘面数据: {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return f"无法获取 {ticker} 的价格数据，请检查代码是否正确（A股需加 .SS 或 .SZ 后缀）。"

        close_price = hist['Close'].iloc[-1]
        open_price = hist['Open'].iloc[-1]
        pct_change = ((close_price - open_price) / open_price) * 100

        return f"{ticker} 最新收盘/当前价格: {close_price:.2f}，今日涨跌幅: {pct_change:.2f}%"
    except Exception as e:
        return f"获取股价失败: {str(e)}"


# ==========================================
# 4. 接入 GPT 大脑 (核心执行引擎)
# ==========================================
def run_trading_agent(user_message: str):
    print(f"\n[老板指令] {user_message}")

    # 注册所有的工具说明书
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_market_news",
                "description": "检索A股市场新闻和快讯。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词（如 '新能源', '降准'）"},
                        "top_k": {"type": "integer", "description": "返回条数，默认3"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "获取指定股票的最新价格和今日涨跌幅。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "股票代码，A股务必带后缀（如 '600519.SS', '002594.SZ'）"}
                    },
                    "required": ["ticker"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": TRADER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # 第 1 轮对话：询问 GPT 是否需要调用工具
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message

    if response_message.tool_calls:
        messages.append(response_message)

        # 依次执行大模型要求调用的所有工具（它可能会同时查新闻和查股价）
        for tool_call in response_message.tool_calls:
            args = json.loads(tool_call.function.arguments)

            if tool_call.function.name == "search_market_news":
                result = search_market_news(query=args.get("query"), top_k=args.get("top_k", 3))
            elif tool_call.function.name == "get_stock_price":
                result = get_stock_price(ticker=args.get("ticker"))

            # 把结果喂回给大模型
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": result
            })

        print("[Agent 思考完毕，正在生成综合分析报告...]")

        # 第 2 轮对话：拿着所有数据写研报
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        print(f"\n[GPT 量化分析师]\n{final_response.choices[0].message.content}")
    else:
        print(f"\n[GPT 量化分析师]\n{response_message.content}")


if __name__ == "__main__":
    print("=== OpenClaw 交易系统已启动 (支持量价+新闻双模分析) ===")
    while True:
        user_input = input("\n你想分析哪只股票？(输入 'q' 退出): ")
        if user_input.lower() == 'q':
            break
        if user_input.strip():
            run_trading_agent(user_input)