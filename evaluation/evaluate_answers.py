"""Evaluate answer generation, citation integrity, and evidence overlap.

Run from the project root: python -m evaluation.evaluate_answers
This makes Gemini API calls (one per test question) and saves a JSON audit.
The numeric/lexical checks are heuristics; they do not prove factual correctness.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from config.settings import settings
from src.generation.gemini_generator import GeminiAnswerGenerator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ESGReranker
from evaluation.citation_metrics import calculate_citation_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE = BASE_DIR / "evaluation" / "test_questions.json"
OUTPUT_FILE = BASE_DIR / "evaluation" / "answer_evaluation_results.json"
logger = logging.getLogger(__name__)


def _load_questions() -> list[dict[str, Any]]:
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test question file not found: {TEST_FILE}")
    questions = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("test_questions.json must contain a non-empty JSON list.")
    for i, item in enumerate(questions, start=1):
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            raise ValueError(f"Test question #{i} is missing its question text.")
        if not isinstance(item.get("expected_keywords", []), list):
            raise ValueError(f"Test question #{i} expected_keywords must be a list.")
    return questions


def _abstained(answer: str) -> bool:
    text = re.sub(r"\s+", " ", (answer or "").casefold()).strip()

    # An answer that supplies a substantive response should not be
    # classified as an abstention merely because it notes a missing detail.
    partial_answer_patterns = (
        r"\btarget\b.{0,100}\b(?:reduction|reduce|by 20\d{2}|net.zero)\b",
        r"\b\d+(?:\.\d+)?\s*%\s+(?:reduction|decrease|cut)\b",
    )
    if any(re.search(pattern, text) for pattern in partial_answer_patterns):
        return False

    patterns = (
        r"\bcould(?: not|n't) find (?:sufficient|relevant|enough) evidence\b",
        r"\binsufficient evidence\b",
        r"\bnot (?:stated|provided|available|disclosed|specified) in the (?:provided )?(?:documents|reports|evidence|sources)\b",
        r"\bthe (?:provided )?(?:evidence|documents|reports|sources) do not\b",
        r"\bthe (?:provided )?(?:evidence|documents|reports|sources) does not contain\b",
        r"\bthe (?:provided )?(?:evidence|documents|reports|sources) does not (?:state|mention|specify|provide|include|identify|indicate|show|establish)\b",
        r"\bunable to (?:determine|confirm|verify|identify|calculate)\b",
        r"\bcannot (?:determine|confirm|verify|identify|calculate)\b",
        r"\bremains unknown\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)

def _expected_keywords_found(answer: str, keywords: list[str]) -> list[str]:
    normalized = re.sub(r"\s+", " ", (answer or "").casefold())
    return [str(k) for k in keywords if str(k).casefold() in normalized]


def _empty_metrics() -> dict[str, Any]:
    return {
        "used_citation_ids": [],
        "invalid_citation_ids": [],
        "citation_validity": 0.0,
        "citation_coverage": 0.0,
        "mean_evidence_keyword_overlap": 0.0,
        "factual_sentence_count": 0,
        "cited_factual_sentence_count": 0,
        "claim_support_audit": [],
        "claims_flagged_for_review": 0,
        "support_metric_note": "No answer was available for diagnostics.",
    }


def main() -> int:
    questions = _load_questions()
    print("=" * 72)
    print("ESG RAG ANSWER + CITATION EVALUATION")
    print(f"Questions: {len(questions)}")
    print("Note: this calls Gemini and consumes API quota.")
    print("=" * 72)

    print("Loading hybrid retriever and reranker...")
    retriever = HybridRetriever()
    reranker = ESGReranker()
    generator = GeminiAnswerGenerator()
    # Resume from existing results, preserving successful answers.
    existing_by_id: dict[str, dict[str, Any]] = {}
    if OUTPUT_FILE.exists():
        try:
            previous_results = json.loads(
                OUTPUT_FILE.read_text(encoding="utf-8")
            )
            if isinstance(previous_results, list):
                existing_by_id = {
                    str(record.get("question_id")): record
                    for record in previous_results
                    if isinstance(record, dict) and record.get("question_id")
                }
                print(
                    f"Loaded {len(existing_by_id)} existing evaluation records."
                )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Could not load prior results; starting fresh: {exc}")

    results: list[dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        question_id = str(item.get("id", f"Q{index:02d}"))
        previous = existing_by_id.get(question_id)

        if previous and previous.get("status") == "success":
            results.append(previous)
            print(
                f"[{index}/{len(questions)}] SKIP | existing successful result | "
                f"{item['question'].strip()}"
            )
            continue

        question = item["question"].strip()
        expected_answerable = item.get("expected_answerable")
        started = time.perf_counter()
        record: dict[str, Any] = {
            "question_id": question_id,
            "question": question,
            "expected_answerable": expected_answerable,
            "expected_keywords": item.get("expected_keywords", []),
        }
        try:
            candidates = retriever.search(question, top_k=20, candidate_k=40)
            if candidates:
                passages = reranker.rerank(question, candidates, top_k=8)
            else:
                passages = []
            record["candidate_count"] = len(candidates)
            record["reranked_passage_count"] = len(passages)
            record["retrieval_seconds"] = round(time.perf_counter() - started, 4)

            # Generator handles empty passages by abstaining without an API call.
            generation_started = time.perf_counter()
            generated = generator.generate_answer(question, passages)
            answer = str(generated.get("answer", "") or "")
            citations = generated.get("citations", []) or []
            generation_seconds = time.perf_counter() - generation_started

            metrics = calculate_citation_metrics(answer, citations, passages)
            found_keywords = _expected_keywords_found(answer, item.get("expected_keywords", []))
            abstained = _abstained(answer)
            answerability_correct = (
                None if expected_answerable is None
                else (not abstained if expected_answerable else abstained)
            )

            record.update({
                "status": "success",
                "answer": answer,
                "citations": citations,
                "model": generated.get("model", settings.gemini_model),
                "abstained": abstained,
                "answerability_classification_correct": answerability_correct,
                "expected_keywords_found": found_keywords,
                "expected_keyword_coverage": (
                    round(len(found_keywords) / len(item.get("expected_keywords", [])), 4)
                    if item.get("expected_keywords") else None
                ),
                "generation_seconds": round(generation_seconds, 4),
                "total_seconds": round(time.perf_counter() - started, 4),
                **metrics,
            })
            print(f"[{index}/{len(questions)}] OK | abstained={abstained} | "
                  f"citation coverage={metrics['citation_coverage']:.0%} | {question}")
        except Exception as exc:
            logger.exception("Question %d failed: %s", index, question)
            record.update({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "answer": "",
                "citations": [],
                "abstained": None,
                "answerability_classification_correct": None,
                "retrieval_seconds": round(time.perf_counter() - started, 4),
                "generation_seconds": 0.0,
                "total_seconds": round(time.perf_counter() - started, 4),
                **_empty_metrics(),
            })
            print(f"[{index}/{len(questions)}] ERROR | {question} | {exc}")
        results.append(record)
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    successful = [r for r in results if r.get("status") == "success"]
    print("\n" + "=" * 72)
    print(f"Completed: {len(successful)}/{len(results)}")
    if successful:
        print(f"Mean citation validity: {sum(r['citation_validity'] for r in successful)/len(successful):.1%}")
        print(f"Mean factual-sentence citation coverage: {sum(r['citation_coverage'] for r in successful)/len(successful):.1%}")
        print(f"Mean evidence keyword overlap: {sum(r['mean_evidence_keyword_overlap'] for r in successful)/len(successful):.1%}")
        labeled = [r for r in successful if r.get("answerability_classification_correct") is not None]
        if labeled:
            print(f"Answerability heuristic agreement: {sum(bool(r['answerability_classification_correct']) for r in labeled)}/{len(labeled)}")
    print(f"Detailed audit saved to: {OUTPUT_FILE}")
    print("Reminder: citation-ID validity and lexical overlap do not prove semantic support.")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
