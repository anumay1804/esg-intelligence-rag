"""Validate the gold answer key and its evidence references.

Gold answers and retrieval test questions are separate datasets: a gold answer
may paraphrase a retrieval question or cover a related but non-identical task.
If a retrieval question supplies `gold_answer_id`, linkage is validated.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = BASE_DIR / "evaluation" / "test_questions.json"
GOLD_PATH = BASE_DIR / "evaluation" / "gold_answers.json"


def validate_datasets(questions: list[dict], gold_answers: list[dict]) -> list[str]:
    errors: list[str] = []
    gold_by_id = {item.get("id"): item for item in gold_answers}
    if len(gold_by_id) != len(gold_answers):
        errors.append("Gold answer IDs must be unique.")
    for index, question in enumerate(questions, start=1):
        question_text = str(question.get("question", "")).strip()
        if not question_text:
            errors.append(f"Question {index} has no question text.")
        if "expected_answerable" in question and not isinstance(question["expected_answerable"], bool):
            errors.append(f"Question {index} expected_answerable must be a boolean.")
        gold_id = question.get("gold_answer_id")
        if gold_id:
            if gold_id not in gold_by_id:
                errors.append(f"Question {index} references missing gold answer: {gold_id!r}.")
            elif gold_by_id[gold_id].get("question") != question_text:
                errors.append(f"Question text does not match gold answer {gold_id}.")
    for item in gold_answers:
        gold_id = item.get("id", "<missing>")
        if not item.get("id"):
            errors.append("Each gold answer must have an ID.")
        if not str(item.get("question", "")).strip():
            errors.append(f"Gold answer {gold_id} has no question.")
        if not str(item.get("expected_answer", "")).strip():
            errors.append(f"Gold answer {gold_id} has no expected answer.")
        evidence = item.get("evidence", [])
        if not evidence:
            errors.append(f"Gold answer {gold_id} has no evidence references.")
        for source in evidence:
            if not source.get("document") or not isinstance(source.get("pdf_page"), int):
                errors.append(f"Gold answer {gold_id} has incomplete evidence metadata.")
            if not source.get("supports"):
                errors.append(f"Gold answer {gold_id} has no support checklist.")
    return errors


def validate_pdf_references(gold_answers: list[dict], raw_dir: Path) -> list[str]:
    """Check that referenced report files and 1-based PDF pages exist."""
    try:
        import fitz
    except ImportError:
        return ["PyMuPDF is required to validate referenced PDF page numbers."]
    errors: list[str] = []
    page_counts: dict[str, int] = {}
    for item in gold_answers:
        for evidence in item.get("evidence", []):
            filename = evidence.get("document", "")
            page_number = evidence.get("pdf_page")
            source_path = raw_dir / filename
            if not source_path.exists():
                errors.append(f"Gold answer {item.get('id')} references missing PDF: {filename}.")
                continue
            if filename not in page_counts:
                try:
                    with fitz.open(source_path) as document:
                        page_counts[filename] = len(document)
                except Exception as exc:
                    errors.append(f"Could not open referenced PDF {filename}: {exc}")
                    continue
            if not isinstance(page_number, int) or page_number < 1 or page_number > page_counts[filename]:
                errors.append(f"Gold answer {item.get('id')} references invalid PDF page {page_number} in {filename}.")
    return errors


def main() -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    gold_answers = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    errors = validate_datasets(questions, gold_answers)
    errors.extend(validate_pdf_references(gold_answers, BASE_DIR / "data" / "raw"))
    if errors:
        print("Gold-answer dataset validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Dataset validation passed: {len(questions)} retrieval questions, {len(gold_answers)} gold answers.")
    print("All gold-answer PDF files and page references exist.")
    print("Reminder: valid page references do not prove that each passage supports the expected answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
