from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ESGReranker


def main():
    print("Loading hybrid retriever...")
    retriever = HybridRetriever()

    print("Loading cross-encoder reranker...")
    reranker = ESGReranker()

    print("\nESG Hybrid Search + Reranking")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your ESG question: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            print("Please enter a question.")
            continue

        # Retrieve a broader set of candidates first.
        candidates = retriever.search(
            query,
            top_k=20,
            candidate_k=30,
        )

        if not candidates:
            print("No relevant passages found.")
            continue

        # Rerank the hybrid retrieval results.
        results = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=5,
        )

        print("\n" + "=" * 75)
        print("TOP RESULTS AFTER CROSS-ENCODER RERANKING")
        print("=" * 75)

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Source: {result.get('source_file', result.get('source', 'Unknown'))}")

            # Support either common page-number field.
            page = result.get("page", result.get("page_number", "Unknown"))
            print(f"Page: {page}")

            print(
                "Reranker score: "
                f"{result.get('reranker_score', 0.0):.4f}"
            )
            print(
                "RRF score: "
                f"{result.get('rrf_score', 0.0):.6f}"
            )

            print("\nRetrieved text:")
            print(result.get("text", "")[:1800])
            print("-" * 75)


if __name__ == "__main__":
    main()