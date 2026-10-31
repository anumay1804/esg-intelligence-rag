import fitz
import pytest

from src.ingestion.pdf_loader import load_pdf


def test_load_pdf_preserves_page_numbers(tmp_path):

    pdf_path = tmp_path / "sample.pdf"

    with fitz.open() as document:

        page = document.new_page()

        page.insert_text(
            (72, 72),
            "Scope 1 emissions were 100 tonnes."
        )

        document.save(pdf_path)

    result = load_pdf(pdf_path)

    assert result["page_count"] == 1

    assert result["pages"][0]["page"] == 1

    assert (
        "Scope 1 emissions"
        in result["pages"][0]["text"]
    )


def test_load_pdf_rejects_non_pdf(tmp_path):

    text_path = tmp_path / "notes.txt"

    text_path.write_text(
        "This is not a PDF.",
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_pdf(text_path)