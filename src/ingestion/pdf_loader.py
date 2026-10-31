"""
Extract text from PDF reports while preserving page numbers.
"""

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str | Path) -> dict[str, Any]:
    """
    Extract text and basic metadata from a PDF.

    Page numbers are 1-based, matching the numbers
    users normally see in a PDF reader.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF does not exist: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {path.name}"
        )

    pages = []

    try:
        with fitz.open(path) as pdf:

            pdf_metadata = pdf.metadata or {}

            for page_index, page in enumerate(pdf):

                text = page.get_text(
                    "text",
                    sort=True
                ).strip()

                pages.append({
                    "page": page_index + 1,
                    "text": text,
                    "char_count": len(text),
                    "is_empty": not bool(text)
                })

            result = {
                "source_file": path.name,
                "source_path": str(path),
                "page_count": len(pages),

                "pdf_metadata": {
                    "title": pdf_metadata.get("title") or "",
                    "author": pdf_metadata.get("author") or "",
                    "subject": pdf_metadata.get("subject") or "",
                    "creation_date": (
                        pdf_metadata.get("creationDate") or ""
                    )
                },

                "pages": pages
            }

    except Exception:
        logger.exception(
            "Failed to parse PDF: %s",
            path
        )
        raise

    empty_pages = sum(
        1 for page in pages
        if page["is_empty"]
    )

    logger.info(
        "Parsed %s: %d pages, %d empty-text pages",
        path.name,
        len(pages),
        empty_pages
    )

    if empty_pages:
        logger.warning(
            "%s has %d pages without extractable text. "
            "The PDF may require OCR.",
            path.name,
            empty_pages
        )

    return result