import logging

from config.settings import settings
from src.retrieval.vector_store import ESGVectorStore
from src.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def main():
    vector_store = ESGVectorStore()
    vector_store.load(settings.indexes_dir)

    query = input("Enter your ESG question: ").strip()

    results = vector_store.search(query, top_k=5)

    if not results:
        print("No matching results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\n{'=' * 70}")
        print(f"Result {rank}")
        print(f"Source: {result.get('source_file', 'Unknown')}")
        print(f"Page: {result.get('page', 'Unknown')}")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Text:\n{result['text'][:1000]}")


if __name__ == "__main__":
    main()