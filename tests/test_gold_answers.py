import json
from pathlib import Path
from evaluation.validate_gold_answers import validate_datasets

BASE_DIR = Path(__file__).resolve().parent.parent

def test_gold_answer_key_and_retrieval_questions_are_valid():
    questions = json.loads((BASE_DIR / "evaluation" / "test_questions.json").read_text(encoding="utf-8"))
    gold = json.loads((BASE_DIR / "evaluation" / "gold_answers.json").read_text(encoding="utf-8"))
    assert validate_datasets(questions, gold) == []
    assert len(questions) >= 10
    assert len(gold) >= 10

def test_gold_answer_validator_detects_missing_evidence():
    questions = [{"question": "What is the target?"}]
    gold = [{"id": "Q1", "question": "What is the target?", "expected_answer": "25%", "evidence": []}]
    errors = validate_datasets(questions, gold)
    assert any("no evidence references" in error for error in errors)

def test_gold_answer_validator_detects_broken_optional_link():
    questions = [{"question": "What is the target?", "gold_answer_id": "missing"}]
    errors = validate_datasets(questions, [])
    assert any("references missing gold answer" in error for error in errors)
