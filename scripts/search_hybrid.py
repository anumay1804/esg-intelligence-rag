import logging

from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def main():
    retriever = HybridRetriever()

    print("\nESG Hybrid Search")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your ESG question: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results = retriever.search(
            query=query,
            top_k=5,
            candidate_k=20,
        )

        if not results:
            print("No matching results found.")
            continue

        for rank, result in enumerate(results, start=1):
            print("\n" + "=" * 75)
            print(f"Rank: {rank}")
            print(f"Source: {result.get('source_file', 'Unknown')}")
            print(f"Page: {result.get('page', 'Unknown')}")
            print(f"RRF score: {result['rrf_score']:.6f}")
            print(f"FAISS rank: {result.get('vector_rank')}")
            print(f"BM25 rank: {result.get('bm25_rank')}")

            similarity = result.get("similarity_score")
            if similarity is not None:
                print(f"FAISS similarity: {similarity:.4f}")

            bm25_score = result.get("bm25_score")
            if bm25_score is not None:
                print(f"BM25 score: {bm25_score:.4f}")

            print("\nRetrieved text:")
            print(result.get("text", "")[:900])


if __name__ == "__main__":
    main()