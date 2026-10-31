from src.processing.chunker import chunk_page, chunk_document


def test_chunk_page_respects_size_limit():
    text = (
        "Scope 1 emissions were 100 tonnes.\n\n"
        "Scope 2 emissions were 200 tonnes.\n\n"
        "Scope 3 emissions were 500 tonnes."
    )

    chunks = chunk_page(
        text,
        max_chars=60,
        overlap_chars=10,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)


def test_chunk_document_preserves_page_metadata():
    document = {
        "source_file": "report.pdf",
        "source_path": "data/raw/report.pdf",
        "metadata": {"title": "ESG Report"},
        "pages": [
            {
                "page": 12,
                "text": "The company reduced its carbon emissions.",
            }
        ],
    }

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0]["source_file"] == "report.pdf"
    assert chunks[0]["page"] == 12
    assert chunks[0]["metadata"]["title"] == "ESG Report"
    assert chunks[0]["text"]