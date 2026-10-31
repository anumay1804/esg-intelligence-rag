"""Compare hybrid retrieval against graph-assisted retrieval.

Run: python -m scripts.run_evaluation
This is a keyword-overlap smoke test, not a human-labelled accuracy benchmark.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.graph_retriever import ESGGraphRetriever
from src.retrieval.reranker import ESGReranker

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "evaluation" / "test_questions.json"
OUTPUT = ROOT / "evaluation" / "graph_comparison_results.json"


def _keywords(text: str, expected: list[str]) -> list[str]:
    normalized = re.sub(r"\s+", " ", (text or "").casefold())
    return [str(word) for word in expected if str(word).casefold() in normalized]


def _evaluate(results: list[dict[str, Any]], expected: list[str], k: int = 5) -> dict[str, Any]:
    top = results[:k]
    joined = " ".join(str(x.get("text", "")) for x in top)
    found = _keywords(joined, expected)
    # A hit is deliberately a lenient heuristic: at least two expected
    # keywords occur somewhere in the top-k passages.
    hit = int(len(found) >= min(2, len(expected))) if expected else 0
    first_rank = None
    for rank, item in enumerate(results, start=1):
        if len(_keywords(str(item.get("text", "")), expected)) >= min(2, len(expected)) and expected:
            first_rank = rank
            break
    return {
        "hit_at_5": hit,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "keyword_coverage": len(found) / len(expected) if expected else 0.0,
        "top_passages": [
            {
                "source_file": item.get("source_file", item.get("source", "Unknown")),
                "page": item.get("page", item.get("page_number", "Unknown")),
                "chunk_id": item.get("chunk_id"),
                "text_excerpt": str(item.get("text", ""))[:350],
            }
            for item in top[:3]
        ],
    }


def main() -> int:
    if not QUESTIONS.exists():
        raise FileNotFoundError(f"Evaluation questions missing: {QUESTIONS}")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    graph_retriever = ESGGraphRetriever(hybrid=retriever)
    reranker = ESGReranker()
    records = []
    for idx, item in enumerate(questions, start=1):
        question = str(item["question"]).strip()
        expected = item.get("expected_keywords", [])
        print(f"[{idx}/{len(questions)}] {question}")
        row = {"question": question, "expected_answerable": item.get("expected_answerable")}
        for label, engine in (("hybrid", retriever), ("graph_assisted", graph_retriever)):
            start = time.perf_counter()
            candidates = engine.search(question, top_k=20, candidate_k=40)
            passages = reranker.rerank(question, candidates, top_k=8) if candidates else []
            elapsed = time.perf_counter() - start
            metrics = _evaluate(passages, expected)
            row[label] = {**metrics, "elapsed_seconds": round(elapsed, 4), "candidate_count": len(candidates)}
        records.append(row)
        OUTPUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved comparison results: {OUTPUT}")
    for label in ("hybrid", "graph_assisted"):
        values = [r[label] for r in records]
        if values:
            print(f"{label}: Hit@5={sum(x['hit_at_5'] for x in values)/len(values):.1%}, "
                  f"MRR={sum(x['mrr'] for x in values)/len(values):.3f}, "
                  f"keyword coverage={sum(x['keyword_coverage'] for x in values)/len(values):.1%}")
    print("These metrics are heuristic. Use manually reviewed relevance labels for formal comparison.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
