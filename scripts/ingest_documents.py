"""
Parse ESG reports and save the extracted content as JSON.
"""

import json
import logging
from pathlib import Path

from config.settings import settings
from src.ingestion.document_parser import (
    parse_pdf_directory
)
from src.utils.logging_config import configure_logging


def main() -> None:

    configure_logging()

    output_dir = Path(
        settings.processed_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Parse all PDFs from data/raw/
    documents = parse_pdf_directory(
        settings.raw_dir
    )

    # Save extracted documents
    output_path = (
        output_dir / "documents.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    # Calculate processing statistics
    total_pages = sum(
        document["page_count"]
        for document in documents
    )

    total_characters = sum(
        page["char_count"]
        for document in documents
        for page in document["pages"]
    )

    logging.info(
        "Saved parsed documents to %s",
        output_path.resolve()
    )

    logging.info(
        "Processed %d documents, %d pages, "
        "and %d extracted characters",
        len(documents),
        total_pages,
        total_characters
    )

    if not documents:
        logging.warning(
            "No documents were processed. "
            "Add PDFs to data/raw/ and run again."
        )


if __name__ == "__main__":
    main()