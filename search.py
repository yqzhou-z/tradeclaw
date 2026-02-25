import chromadb
from chromadb.utils import embedding_functions


def init_retriever():
    print("Connecting to ChromaDB vector knowledge base...")
    # Connect to the local database folder we generated
    chroma_client = chromadb.PersistentClient(path="./news_db")

    # Must use the EXACT SAME embedding model as used for data ingestion
    chinese_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")

    try:
        # Get the existing collection
        collection = chroma_client.get_collection(
            name="a_share_news",
            embedding_function=chinese_embedding
        )
        print(f"Connection successful! The knowledge base currently contains {collection.count()} news items.\n")
        return collection
    except Exception as e:
        print(f"Connection failed. Please ensure 'news.py' has been run successfully to generate data. Error: {e}")
        return None


def search_news(collection, query, top_k=3):
    print(f"\nRetrieving news most relevant to '{query}'...\n" + "=" * 50)

    # Execute vector similarity search
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    # Extract and print the results
    if results['documents'] and len(results['documents'][0]) > 0:
        for i, doc in enumerate(results['documents'][0]):
            print(f"[{i + 1}] {doc}")
            print("-" * 50)
    else:
        print("No highly relevant matching results found in the database.")


if __name__ == "__main__":
    db_collection = init_retriever()

    if db_collection:
        print(
            "Testing Guide: You can enter specific stocks (e.g., Kweichow Moutai, BYD) or macroeconomic concepts (e.g., rate cut, new energy, semiconductors).")
        while True:
            user_input = input("\nPlease enter a search keyword (enter 'q' to quit): ")
            if user_input.lower() == 'q':
                print("Exiting the search test.")
                break
            if user_input.strip():
                search_news(db_collection, user_input)