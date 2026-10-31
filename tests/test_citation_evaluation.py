from evaluation.citation_metrics import extract_citation_ids, calculate_citation_metrics
from evaluation.validate_citation_support import collect_claims, extract_json


def test_extracts_single_and_grouped_citations():
    text = "Emissions fell [S1]. The target is 25% [S2, S3]."
    assert extract_citation_ids(text) == {"S1", "S2", "S3"}


def test_citation_metrics_include_evidence_for_manual_review():
    answer = "Emissions were 17 million tonnes in 2024 [S1]."
    citations = [{"id": "S1", "source_file": "report.pdf", "page": 12}]
    passages = [{"chunk_id": "c1", "text": "In 2024 emissions were 17 million tonnes.",
                 "source_file": "report.pdf", "page": 12}]
    result = calculate_citation_metrics(answer, citations, passages)
    assert result["citation_validity"] == 1.0
    assert result["citation_coverage"] == 1.0
    audit = result["claim_support_audit"][0]
    assert audit["citation_ids"] == ["S1"]
    assert audit["cited_evidence"][0]["page"] == 12
    assert "17 million tonnes" in audit["cited_evidence"][0]["excerpt"]


def test_invalid_citation_is_flagged():
    result = calculate_citation_metrics(
        "Emissions were 17 million tonnes [S9].",
        [{"id": "S1"}],
        [{"text": "Emissions were 17 million tonnes."}],
    )
    assert result["invalid_citation_ids"] == ["S9"]
    assert result["claim_support_audit"][0]["review_flag"] == "invalid_citation_id"


def test_collect_claims_for_judge():
    records = [{
        "question": "What were emissions?",
        "claim_support_audit": [{
            "sentence": "Emissions were 17 tonnes [S1].",
            "citation_ids": ["S1"],
            "has_citation": True,
            "cited_evidence": [{"citation_id": "S1", "excerpt": "Emissions were 17 tonnes."}],
        }],
    }]
    claims = collect_claims(records)
    assert claims[0]["claim_id"] == "Q1-C1"
    assert claims[0]["evidence"][0]["citation_id"] == "S1"


def test_extract_json_handles_code_fence():
    parsed = extract_json('```json\n{"results": []}\n```')
    assert parsed == {"results": []}
