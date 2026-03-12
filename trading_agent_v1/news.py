import os
import time
import hashlib
import requests
import chromadb
from chromadb.utils import embedding_functions


def fetch_crypto_news(limit: int = 20):
    print(f"Fetching latest {limit} crypto news from CryptoCompare API...")
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        news_list = data.get("Data", [])[:limit]
        formatted_news = []

        for item in news_list:
            published_on = item.get("published_on")
            pub_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(published_on)) if published_on else "Unknown Time"
            title = item.get("title", "")
            body = item.get("body", "")
            categories = item.get("categories", "")
            source = item.get("source_info", {}).get("name", "cryptocompare")

            content = f"[{pub_time}] {title} - {body}"

            formatted_news.append({
                "content": content,
                "metadata": {
                    "source": source,
                    "published_on": published_on if published_on else None,
                    "categories": categories
                }
            })

        return formatted_news

    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def get_md5_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def update_vector_db():
    print("=== Crypto Auto-Scraper Task Started ===")
    print("Initializing ChromaDB vector database...")

    chroma_client = chromadb.PersistentClient(path="./news_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    collection = chroma_client.get_or_create_collection(
        name="crypto_news",
        embedding_function=embedding_fn
    )

    latest_news = fetch_crypto_news(limit=20)

    if not latest_news:
        print("No news fetched. Exiting.")
        return

    new_count = 0
    for item in latest_news:
        news_text = item["content"]
        metadata = item["metadata"]
        news_id = get_md5_hash(news_text)

        existing = collection.get(ids=[news_id])
        if not existing["ids"]:
            collection.add(
                documents=[news_text],
                metadatas=[metadata],
                ids=[news_id]
            )
            new_count += 1
            print(f"[NEW] Added: {news_text[:80]}...")

    print(f"Database update complete! Added {new_count} new items.")
    print(f"Total items in 'crypto_news' collection: {collection.count()}")
    print("=== Crypto Auto-Scraper Task Finished ===\n")


if __name__ == "__main__":
    update_vector_db()