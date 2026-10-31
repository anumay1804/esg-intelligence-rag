import json
import logging
from src.utils.logging_config import configure_logging
from pathlib import Path
from config.settings import settings
from src.processing.chunker import chunk_document



configure_logging()
logger = logging.getLogger(__name__)


def main():
    input_path = Path(settings.processed_dir) / "documents.json"
    output_path = Path(settings.processed_dir) / "chunks.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Parsed documents not found: {input_path}. "
            "Run the ingestion script first."
        )

    with open(input_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    all_chunks = []

    for document in documents:
        chunks = chunk_document(
            document,
            max_chars=1800,
            overlap_chars=250,
        )

        all_chunks.extend(chunks)

        logger.info(
            "Processed %s: generated %d chunks",
            document.get("source_file", "unknown.pdf"),
            len(chunks),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, ensure_ascii=False, indent=2)

    logger.info("Total chunks: %d", len(all_chunks))
    logger.info("Saved chunks to: %s", output_path)


if __name__ == "__main__":
    main()