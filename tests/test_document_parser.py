from pathlib import Path
import fitz
from src.ingestion.document_parser import parse_pdf_directory


def test_parser_skips_malformed_pdf_and_keeps_valid_pdf(tmp_path):
    valid = tmp_path / "valid.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "BASF renewable electricity was 26% in 2024.")
        doc.save(valid)
    malformed = tmp_path / "broken.pdf"
    malformed.write_bytes(b"this is not a real PDF file")

    parsed = parse_pdf_directory(tmp_path)

    assert len(parsed) == 1
    assert parsed[0]["source_file"] == "valid.pdf"
    assert "renewable electricity" in parsed[0]["pages"][0]["text"]


def test_parser_returns_empty_list_for_directory_without_pdfs(tmp_path):
    (tmp_path / "notes.txt").write_text("not a PDF", encoding="utf-8")
    assert parse_pdf_directory(tmp_path) == []
