import time

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ESGReranker
from src.generation.gemini_generator import GeminiAnswerGenerator


def main():
    print("Initializing ESG Intelligence RAG...")
    print("Loading hybrid retriever...")

    retriever = HybridRetriever()

    print("Loading cross-encoder...")
    reranker = ESGReranker()

    print("Initializing Gemini...")
    generator = GeminiAnswerGenerator()

    print("\n" + "=" * 65)
    print("ESG INTELLIGENCE ASSISTANT")
    print("Hybrid Retrieval + Cross-Encoder + Gemini")
    print("Type 'exit' to quit.")
    print("=" * 65)

    while True:
        question = input("\nEnter your ESG question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            print("Please enter a question.")
            continue

        start_time = time.perf_counter()

        try:
            # Stage 1: Hybrid retrieval
            candidates = retriever.search(
                question,
                top_k=20,
                candidate_k=40,
            )

            if not candidates:
                print("No relevant passages were retrieved.")
                continue

            # Stage 2: Cross-encoder reranking
            passages = reranker.rerank(
                query=question,
                candidates=candidates,
                top_k=8,
            )

            if not passages:
                print("No usable passages were found.")
                continue

            retrieval_time = time.perf_counter() - start_time

            # Stage 3: Gemini answer generation
            generation_start = time.perf_counter()

            result = generator.generate_answer(
                question=question,
                passages=passages,
            )

            generation_time = time.perf_counter() - generation_start
            total_time = time.perf_counter() - start_time

            print("\n" + "=" * 65)
            print("ANSWER")
            print("=" * 65)
            print(result["answer"])

            print("\n" + "-" * 65)
            print("SOURCES")
            print("-" * 65)

            for citation in result["citations"]:
                print(
                    f"[{citation['id']}] "
                    f"{citation['source_file']} "
                    f"(Page {citation['page']})"
                )

            print("\n" + "-" * 65)
            print("PERFORMANCE")
            print("-" * 65)
            print(f"Model: {result['model']}")
            print(f"Retrieval + reranking: {retrieval_time:.2f} seconds")
            print(f"Gemini generation: {generation_time:.2f} seconds")
            print(f"Total time: {total_time:.2f} seconds")

        except Exception as exc:
            print(f"\nAn error occurred: {exc}")
            print(
                "Check your API key, model availability, network "
                "connection, and terminal traceback."
            )


if __name__ == "__main__":
    main()