from src.processing.metadata_extractor import (
    classify_statement,
    extract_metadata,
    extract_scopes,
    extract_years,
    extract_metric_categories,
)


def test_extract_years():
    text = "The report covers 2023 and 2024. It also mentions 2024 again."

    assert extract_years(text) == [2023, 2024]


def test_extract_scopes():
    text = "Scope 1 emissions decreased, while Scope 2 emissions increased."

    assert extract_scopes(text) == ["Scope 1", "Scope 2"]


def test_extract_metric_categories():
    text = "We reduced greenhouse gas emissions and increased renewable energy use."

    categories = extract_metric_categories(text)

    assert "greenhouse_gas_emissions" in categories
    assert "energy" in categories


def test_extract_metadata_preserves_original_chunk():
    chunk = {
        "chunk_id": "report.pdf:page_5:chunk_1",
        "source_file": "BASF Sustainability Report 2024.pdf",
        "page": 5,
        "text": (
            "In 2024, Scope 1 emissions were 120,000 tonnes. "
            "The company also reported greenhouse gas emissions."
        ),
        "metadata": {"title": "Sustainability Report"},
    }

    result = extract_metadata(chunk)

    assert result["chunk_id"] == chunk["chunk_id"]
    assert result["page"] == 5
    assert result["metadata"] == chunk["metadata"]
    assert result["esg_metadata"]["company"] == "BASF"
    assert 2024 in result["esg_metadata"]["years_mentioned"]
    assert "Scope 1" in result["esg_metadata"]["scopes_mentioned"]
    assert result["esg_metadata"]["value_mentions"]

def test_extract_scope_3_1():
    text = (
        "Scope 3.1 emissions intensity decreased from "
        "1.64 kg CO2 per kg of raw materials in 2022 "
        "to a target of 1.39 kg in 2030."
    )

    scopes = extract_scopes(text)

    assert "Scope 3.1" in scopes
    assert "Scope 3" not in scopes


def test_classify_toc_text_as_unclassified():
    text = (
        "Contents: Environmental performance, "
        "Financial results, and Sustainability strategy."
    )

    assert classify_statement(text) == "unclassified"


def test_classify_target_language():
    text = (
        "We aim to reduce greenhouse gas emissions by 25% by 2030."
    )

    assert classify_statement(text) == "potential_target"
def test_scope_3_subcategory_matching():
    text = (
        "Scope 3.1 emissions are included. "
        "Scope 3.11 and Scope 3.12 are not supported categories."
    )
    scopes = extract_scopes(text)
    assert "Scope 3.1" in scopes
    assert "Scope 3.11" not in scopes
    assert "Scope 3.12" not in scopes
    assert "Scope 3" not in scopes
