import requests
import time
import hashlib
import logging
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 0. 配置日志 (专为后台定时任务设计)
# 会同时把信息打印到屏幕，并保存到 scraper.log 文件中
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# ==========================================
# 1. 数据库初始化配置
# ==========================================
def init_db():
    logging.info("初始化 ChromaDB 向量数据库")
    # 自动在当前运行目录下创建或连接 news_db 文件夹
    chroma_client = chromadb.PersistentClient(path="./news_db")

    logging.info("加载中文 Embedding 模型")
    chinese_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")

    # 创建或获取集合
    collection = chroma_client.get_or_create_collection(
        name="a_share_news",
        embedding_function=chinese_embedding
    )
    return collection


# ==========================================
# 2. 数据抓取逻辑
# ==========================================
def fetch_latest_news(limit=15):
    logging.info(f"开始抓取最新 {limit} 条A股市场快讯...")
    # 这是一个公开的新浪财经 7x24 滚动快讯接口
    url = f"https://zhibo.sina.com.cn/api/zhibo/feed?page=1&page_size={limit}&zhibo_id=152"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    news_items = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 解析 JSON 拿到新闻列表
        news_list = data.get('result', {}).get('data', {}).get('feed', {}).get('list', [])

        for item in news_list:
            content = item.get('rich_text', '')
            create_time = item.get('create_time', '')

            if content:
                # 拼装带时间戳的完整文本
                full_text = f"[{create_time}] {content}"
                news_items.append(full_text)

        logging.info(f"成功抓取 {len(news_items)} 条快讯。")
        return news_items

    except Exception as e:
        logging.error(f"获取新闻失败: {str(e)}")
        return []


# ==========================================
# 3. 数据去重与入库逻辑
# ==========================================
def save_to_chroma(collection, news_items):
    if not news_items:
        logging.warning("没有可存入的数据。")
        return

    documents = []
    ids = []

    for text in news_items:
        # 核心防重机制：使用新闻文本生成 MD5 唯一 ID
        # 这样即使定时任务反复抓到同一条新闻，数据库里也永远只有一条
        doc_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        documents.append(text)
        ids.append(doc_id)

    try:
        # 使用 upsert (插入或更新)，完美处理重复 ID 报错问题
        collection.upsert(
            documents=documents,
            ids=ids
        )
        # 打印当前数据库里的总新闻条数
        logging.info(f"数据入库完毕！当前知识库总条数: {collection.count()}")
    except Exception as e:
        logging.error(f"存入数据库时发生错误: {str(e)}")


# ==========================================
# 主程序执行入口
# ==========================================
if __name__ == "__main__":
    logging.info("=== 自动抓取任务开始 ===")
    try:
        # 1. 初始化数据库
        db_collection = init_db()
        # 2. 抓取新闻
        latest_news = fetch_latest_news(limit=20)
        # 3. 存入知识库
        save_to_chroma(db_collection, latest_news)
    except Exception as e:
        logging.critical(f"程序运行发生致命错误: {str(e)}")
    logging.info("=== 自动抓取任务结束 ===\n")