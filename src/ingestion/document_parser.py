"""
Parse all PDF reports from a directory.
"""

import logging
from pathlib import Path
from typing import Any

from src.ingestion.pdf_loader import load_pdf

logger = logging.getLogger(__name__)


def parse_pdf_directory(
    input_dir: str | Path
) -> list[dict[str, Any]]:
    """
    Parse every PDF in the specified directory.

    If one PDF fails, continue processing the others.
    """

    directory = Path(input_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".pdf"
    )

    if not pdf_paths:
        logger.warning(
            "No PDF reports found in %s",
            directory.resolve()
        )
        return []

    documents = []

    for pdf_path in pdf_paths:

        try:
            document = load_pdf(pdf_path)
            documents.append(document)

        except Exception as exc:
            logger.error(
                "Skipping %s because parsing failed: %s",
                pdf_path.name,
                exc
            )

    logger.info(
        "Successfully parsed %d out of %d PDF files",
        len(documents),
        len(pdf_paths)
    )

    return documents