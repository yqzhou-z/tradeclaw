import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import json
import os
from openai import OpenAI
# ==========================================
# 1. 赋予灵魂：System Prompt (系统提示词)
# ==========================================
TRADER_SYSTEM_PROMPT = """
你现在是一位顶级的A股量化分析师和交易策略专家。
你的任务是根据最新的市场快讯、政策动态和资金流向，为用户提供极度客观、理性的投资和交易分析。

你的核心工作原则：
1. 事实至上：在回答用户的任何市场问题前，必须优先调用你的 `search_market_news` 工具，从本地知识库中获取最新资讯。绝不能凭空捏造（幻觉）新闻。
2. 深度推演：不仅要复述新闻，还要分析该新闻对特定板块、产业链或个股的利好/利空影响，并给出清晰的逻辑推演。
3. 专业冷酷：语言风格要精炼、专业、一针见血，直接给出结论，拒绝使用情绪化或含糊的词汇。
4. 合规底线：永远在回答的最后附上一句免责声明：“股市有风险，投资需谨慎，本建议仅供策略回测参考，不构成实盘交易建议。”
"""


# ==========================================
# 2. 初始化知识库连接
# ==========================================
def init_knowledge_base():
    try:
        client = chromadb.PersistentClient(path="./news_db")
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
        collection = client.get_collection(name="a_share_news", embedding_function=embedding_fn)
        return collection
    except Exception as e:
        print(f"知识库连接失败，请确保 news_db 存在。报错: {e}")
        return None


db_collection = init_knowledge_base()


# ==========================================
# 3. 打造武器：Agent 可调用的 Tool / Skill
# ==========================================
def search_market_news(query: str, top_k: int = 3) -> str:
    """
    【工具说明书】
    功能：从本地向量数据库中检索最新的A股市场新闻和快讯。
    触发时机：当你需要回答关于某只股票、某个行业板块或宏观经济的最新动态时，必须调用此工具获取上下文。

    参数:
    - query (str): 搜索关键词，必须是精炼的实体词（例如 "新能源"、"降准"、"贵州茅台"）。
    - top_k (int): 返回最相关的新闻条数，默认返回3条。

    返回:
    - str: 包含最新相关新闻文本的字符串，若无结果则返回提示。
    """
    print(f"\n[Agent 思考中] 正在调用工具检索本地知识库: {query}...")

    if not db_collection:
        return "本地知识库未准备好，无法检索。"

    results = db_collection.query(
        query_texts=[query],
        n_results=top_k
    )

    if results['documents'] and len(results['documents'][0]) > 0:
        # 将检索到的新闻拼接成一个长字符串，喂给大模型
        news_list = [f"- {doc}" for doc in results['documents'][0]]
        return "\n".join(news_list)
    else:
        return f"当前数据库中没有找到与 '{query}' 相关的最新新闻。"




# ==========================================
# 4. 接入 GPT-5.2 大脑 (OpenAI API)
# ==========================================
# 初始化 OpenAI 客户端 (系统会自动读取环境里的 OPENAI_API_KEY)
load_dotenv()
client = OpenAI()


def run_trading_agent(user_message: str):
    print(f"\n[用户] {user_message}")

    # 1. 向大模型注册我们的工具库
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_market_news",
                "description": "从本地向量数据库中检索最新的A股市场新闻和快讯。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词，必须是精炼的实体词（如 '新能源', '降准', '贵州茅台'）",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回最相关的新闻条数，默认返回 3 条",
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # 初始化对话上下文，注入你的 System Prompt
    messages = [
        {"role": "system", "content": TRADER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # 2. 第一次调用：让 GPT-5.2 决定是否需要动用工具
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # 3. 检查 GPT-5.2 是否发起了工具调用请求
    if response_message.tool_calls:
        messages.append(response_message)  # 把它的调用请求记入历史

        # 执行所有的调用请求
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "search_market_news":
                # 解析大模型聪明地提取出来的参数
                args = json.loads(tool_call.function.arguments)
                query = args.get("query")
                top_k = args.get("top_k", 3)

                # 真正去本地执行你的 Python 函数！
                tool_result = search_market_news(query=query, top_k=top_k)

                # 把知识库查回来的新闻喂回给 GPT-5.2
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "search_market_news",
                    "content": tool_result
                })

        print("\n[Agent 思考完毕，正在生成分析报告...]")

        # 4. 第二次调用：GPT-5.2 拿着新闻开始写分析报告
        final_response = client.chat.completions.create(
            model="gpt-5.2",
            messages=messages
        )
        print(f"\n[GPT-5.2 量化分析师]\n{final_response.choices[0].message.content}")
    else:
        # 如果大模型觉得不需要查新闻，直接输出（比如你只是对它说了句“你好”）
        print(f"\n[GPT-5.2 量化分析师]\n{response_message.content}")


if __name__ == "__main__":
    print("=== OpenClaw 交易系统已启动 (Core: GPT-5.2) ===")
    while True:
        user_input = input("\n你想分析什么板块或股票？(输入 'q' 退出): ")
        if user_input.lower() == 'q':
            print("系统关闭。")
            break
        if user_input.strip():
            run_trading_agent(user_input)