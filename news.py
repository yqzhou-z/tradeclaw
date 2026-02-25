import requests
import json
from datetime import datetime


def fetch_latest_a_share_news(limit: int = 10) -> str:
    """
    抓取最新的A股财经滚动新闻。

    Args:
        limit (int): 需要抓取的新闻条数，默认 10 条。

    Returns:
        str: 格式化后的新闻文本，可以直接喂给大模型或存入知识库。
    """
    # 这是一个示例接口（实际操作中建议通过浏览器 F12 开发者工具抓取最新的财联社/新浪财经 API URL）
    # 很多财经网站的快讯接口会返回类似如下结构的 JSON 数据
    url = f"https://zhibo.sina.com.cn/api/zhibo/feed?page=1&page_size={limit}&zhibo_id=152"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 解析返回的 JSON 数据 (需要根据实际接口结构调整)
        news_list = data.get('result', {}).get('data', {}).get('feed', {}).get('list', [])

        formatted_news = []
        for item in news_list:
            content = item.get('rich_text', '')
            create_time = item.get('create_time', '')

            if content:
                formatted_news.append(f"[{create_time}] {content}")

        # 将列表拼接成一段清晰的文本
        return "\n\n".join(formatted_news)

    except Exception as e:
        return f"获取新闻失败: {str(e)}"

# 本地测试一下
print(fetch_latest_a_share_news(3))