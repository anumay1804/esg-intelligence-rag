import json
import re
import time
from pathlib import Path

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ESGReranker


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE = BASE_DIR / "evaluation" / "test_questions.json"

CANDIDATE_K = 40
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 8
EVALUATION_K = 5

# A passage is provisionally relevant if it contains
# at least two expected keywords.
MIN_MATCHED_KEYWORDS = 2


# --------------------------------------------------
# TEXT UTILITIES
# --------------------------------------------------

def normalize_text(text):
    """Normalize text for case-insensitive keyword matching."""
    return re.sub(r"\s+", " ", str(text)).casefold()


def get_passage_text(passage):
    """Safely extract passage text from a retrieved result."""
    return passage.get("text", "") or ""


def get_source(passage):
    """Extract source filename and page information."""
    source = passage.get(
        "source_file",
        passage.get("source", "Unknown source"),
    )

    page = passage.get(
        "page",
        passage.get("page_number", "Unknown"),
    )

    return source, page


def matched_keywords(text, expected_keywords):
    """Return expected keywords present in the text."""
    normalized = normalize_text(text)

    return [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in normalized
    ]


def is_relevant(passage, expected_keywords):
    """
    Preliminary relevance heuristic.

    A passage is relevant when it contains at least
    MIN_MATCHED_KEYWORDS expected keywords.

    This is not a substitute for human-verified relevance labels.
    """
    text = get_passage_text(passage)

    matches = matched_keywords(text, expected_keywords)

    return len(matches) >= MIN_MATCHED_KEYWORDS


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_hit_at_k(passages, expected_keywords, k=5):
    """
    Hit@K = 1 if at least one relevant passage occurs
    in the first K results; otherwise 0.
    """
    top_passages = passages[:k]

    return int(
        any(
            is_relevant(passage, expected_keywords)
            for passage in top_passages
        )
    )


def calculate_reciprocal_rank(passages, expected_keywords):
    """
    Reciprocal rank = 1 / rank of the first relevant passage.
    Returns 0 if no relevant passage is found.
    """
    for rank, passage in enumerate(passages, start=1):
        if is_relevant(passage, expected_keywords):
            return 1.0 / rank

    return 0.0


def calculate_keyword_coverage(passages, expected_keywords, k=5):
    """
    Fraction of distinct expected keywords found across
    the top K passages.

    This measures keyword coverage, not factual correctness.
    """
    if not expected_keywords:
        return 0.0

    top_passages = passages[:k]

    combined_text = " ".join(
        get_passage_text(passage)
        for passage in top_passages
    )

    found = matched_keywords(combined_text, expected_keywords)

    return len(found) / len(expected_keywords)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_test_questions():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {TEST_FILE}"
        )

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError(
            "The test dataset must contain a non-empty JSON list."
        )

    for index, item in enumerate(questions, start=1):
        if not item.get("question"):
            raise ValueError(
                f"Test case {index} is missing its question."
            )

        if not item.get("expected_keywords"):
            raise ValueError(
                f"Test case {index} has no expected_keywords."
            )

    return questions


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():
    print("=" * 75)
    print("ESG RAG RETRIEVAL EVALUATION")
    print("=" * 75)

    test_questions = load_test_questions()

    print(f"\nTest questions: {len(test_questions)}")
    print("\nInitializing hybrid retriever...")
    retriever = HybridRetriever()

    print("Initializing cross-encoder reranker...")
    reranker = ESGReranker()

    results = []

    for index, test_case in enumerate(test_questions, start=1):
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        expected_answerable = test_case.get("expected_answerable", True)

        print("\n" + "-" * 75)
        print(f"Test {index}/{len(test_questions)}")
        print(f"Question: {question}")

        start_time = time.perf_counter()

        try:
            # Stage 1: Hybrid retrieval
            candidates = retriever.search(
                question,
                top_k=RETRIEVAL_TOP_K,
                candidate_k=CANDIDATE_K,
            )

            retrieval_elapsed = time.perf_counter() - start_time

            if not candidates:
                print("No candidate passages retrieved.")

                results.append({
                    "question": question,
                    "expected_answerable": expected_answerable,
                    "evaluation_type": (
                        "retrieval_quality"
                        if expected_answerable
                        else "unanswerable_query"
                    ),
                    "hit_at_5": 0,
                    "mrr": 0.0,
                    "keyword_coverage": 0.0,
                    "retrieval_seconds": retrieval_elapsed,
                    "reranking_seconds": 0.0,
                    "total_seconds": retrieval_elapsed,
                    "status": "no_candidates",
                })

                continue

            # Stage 2: Cross-encoder reranking
            rerank_start = time.perf_counter()

            passages = reranker.rerank(
                query=question,
                candidates=candidates,
                top_k=RERANK_TOP_K,
            )

            reranking_elapsed = time.perf_counter() - rerank_start
            total_elapsed = time.perf_counter() - start_time

            if not passages:
                print("No passages returned after reranking.")

                results.append({
                    "question": question,
                    "expected_answerable": expected_answerable,
                    "evaluation_type": (
                        "retrieval_quality"
                        if expected_answerable
                        else "unanswerable_query"
                    ),
                    "hit_at_5": 0,
                    "mrr": 0.0,
                    "keyword_coverage": 0.0,
                    "retrieval_seconds": retrieval_elapsed,
                    "reranking_seconds": reranking_elapsed,
                    "total_seconds": total_elapsed,
                    "status": "no_reranked_passages",
                })

                continue

            # Stage 3: Calculate metrics
            hit_at_5 = calculate_hit_at_k(
                passages,
                expected_keywords,
                k=EVALUATION_K,
            )

            mrr = calculate_reciprocal_rank(
                passages,
                expected_keywords,
            )

            coverage = calculate_keyword_coverage(
                passages,
                expected_keywords,
                k=EVALUATION_K,
            )

            print(f"Candidate passages: {len(candidates)}")
            print(f"Reranked passages: {len(passages)}")
            print(f"Hit@5: {hit_at_5}")
            print(f"Reciprocal rank: {mrr:.3f}")
            print(f"Keyword coverage: {coverage:.1%}")
            print(f"Retrieval time: {retrieval_elapsed:.2f}s")
            print(f"Reranking time: {reranking_elapsed:.2f}s")
            print(f"Total time: {total_elapsed:.2f}s")

            print("\nTop 3 passages:")

            for rank, passage in enumerate(passages[:3], start=1):
                source, page = get_source(passage)
                matches = matched_keywords(
                    get_passage_text(passage),
                    expected_keywords,
                )

                print(
                    f"\n  Rank {rank}: {source}, Page {page}"
                )
                print(f"  Matched keywords: {matches}")
                print(
                    "  Text: "
                    + get_passage_text(passage)[:250].replace(
                        "\n", " "
                    )
                )

            results.append({
                "question": question,
                "expected_answerable": expected_answerable,
                "evaluation_type": (
                    "retrieval_quality"
                    if expected_answerable
                    else "unanswerable_query"
                ),
                "hit_at_5": hit_at_5,
                "mrr": mrr,
                "keyword_coverage": coverage,
                "retrieval_seconds": retrieval_elapsed,
                "reranking_seconds": reranking_elapsed,
                "total_seconds": total_elapsed,
                "status": "success",
            })

        except Exception as exc:
            print(f"Evaluation failed: {exc}")

            results.append({
                "question": question,
                "expected_answerable": expected_answerable,
                "evaluation_type": (
                    "retrieval_quality"
                    if expected_answerable
                    else "unanswerable_query"
                ),
                "hit_at_5": 0,
                "mrr": 0.0,
                "keyword_coverage": 0.0,
                "retrieval_seconds": 0.0,
                "reranking_seconds": 0.0,
                "total_seconds": 0.0,
                "status": f"error: {exc}",
            })

    # --------------------------------------------------
    # AGGREGATE RESULTS
    # --------------------------------------------------

    # These statuses mean the question completed evaluation, including
    # valid zero-result retrievals. Exceptions are counted as failed runs.
    completed_statuses = {
        "success",
        "no_candidates",
        "no_reranked_passages",
    }

    completed = [
        result
        for result in results
        if result["status"] in completed_statuses
    ]

    failed = [
        result
        for result in results
        if result["status"] not in completed_statuses
    ]

    # Retrieval-quality metrics are calculated ONLY for questions marked
    # answerable in test_questions.json. Unanswerable questions are kept
    # separate and should be evaluated for answer abstention at generation
    # time, not by searching source passages for the words "insufficient evidence".
    retrieval_results = [
        result
        for result in completed
        if result.get("expected_answerable", True)
    ]

    unanswerable_results = [
        result
        for result in completed
        if not result.get("expected_answerable", True)
    ]

    print("\\n" + "=" * 75)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 75)

    print(f"Total test cases: {len(results)}")
    print(f"Completed cases: {len(completed)}")
    print(f"Failed cases (exceptions): {len(failed)}")
    print(f"Answerable cases included in retrieval metrics: {len(retrieval_results)}")
    print(f"Unanswerable cases evaluated separately: {len(unanswerable_results)}")

    if retrieval_results:
        count = len(retrieval_results)

        mean_hit_at_5 = sum(
            item["hit_at_5"] for item in retrieval_results
        ) / count

        mean_mrr = sum(
            item["mrr"] for item in retrieval_results
        ) / count

        mean_coverage = sum(
            item["keyword_coverage"] for item in retrieval_results
        ) / count

        mean_retrieval = sum(
            item["retrieval_seconds"] for item in retrieval_results
        ) / count

        mean_reranking = sum(
            item["reranking_seconds"] for item in retrieval_results
        ) / count

        mean_total = sum(
            item["total_seconds"] for item in retrieval_results
        ) / count

        print("\\nANSWERABLE-QUESTION RETRIEVAL METRICS")
        print(f"Mean Hit@{EVALUATION_K}: {mean_hit_at_5:.1%}")
        print(f"Mean MRR: {mean_mrr:.3f}")
        print(f"Mean keyword coverage: {mean_coverage:.1%}")
        print(f"Mean retrieval time: {mean_retrieval:.2f}s")
        print(f"Mean reranking time: {mean_reranking:.2f}s")
        print(f"Mean retrieval pipeline time: {mean_total:.2f}s")
    else:
        print("\\nNo answerable questions were available for retrieval metrics.")

    if unanswerable_results:
        print("\\nUNANSWERABLE-QUESTION SUMMARY")
        print(
            "These cases are excluded from retrieval-quality averages. "
            "Review the generated answers separately to measure whether "
            "the application abstains when evidence is missing."
        )

        for item in unanswerable_results:
            print(f"- {item['question']} (status: {item['status']})")

    # Save results for later comparison and reporting.
    output_file = BASE_DIR / "evaluation" / "evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")
    print("\nNote: These metrics use keyword-based relevance heuristics.")
    print("Validate against manually labelled relevant passages before")
    print("reporting them as formal retrieval-quality measurements.")


if __name__ == "__main__":
    main()