import time
import hashlib
import feedparser
import chromadb
from chromadb.utils import embedding_functions


def get_md5_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def fetch_rss_alpha():
    print("[*] Initiating Free RSS Alpha Radar...")

    rss_urls = [
        "https://cointelegraph.com/rss",
        "https://rsshub.app/telegram/channel/Tree_News"
    ]

    formatted_news = []

    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "")
                published = entry.get("published", "")

                summary = entry.get("summary", "")
                if "<" in summary:
                    summary = summary.split("<")[0]

                published_ts = None
                if entry.get("published_parsed"):
                    published_ts = int(time.mktime(entry.published_parsed))

                content = f"[Alpha Radar | {published}] {title} - {summary}"

                formatted_news.append({
                    "content": content,
                    "metadata": {
                        "source": url,
                        "published_on": published_ts
                    }
                })
        except Exception as e:
            print(f"[-] Error fetching {url}: {e}")

    return formatted_news


def update_vector_db_with_rss():
    print("\n=== RSS Alpha Feeder Started ===")

    chroma_client = chromadb.PersistentClient(path="./news_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    collection = chroma_client.get_or_create_collection(
        name="crypto_news",
        embedding_function=embedding_fn
    )

    latest_alpha = fetch_rss_alpha()

    if not latest_alpha:
        print("[-] No alpha fetched. Exiting.")
        return

    new_count = 0
    for item in latest_alpha:
        alpha_text = item["content"]
        metadata = item["metadata"]
        alpha_id = get_md5_hash(alpha_text)

        existing = collection.get(ids=[alpha_id])
        if not existing["ids"]:
            collection.add(
                documents=[alpha_text],
                metadatas=[metadata],
                ids=[alpha_id]
            )
            new_count += 1
            print(f"[NEW ALPHA] Stored: {alpha_text[:80]}...")

    print(f"[*] Task complete! Added {new_count} new high-signal alpha to ChromaDB.")
    print("=== RSS Alpha Feeder Finished ===\n")


if __name__ == "__main__":
    update_vector_db_with_rss()