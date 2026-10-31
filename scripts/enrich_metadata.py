import json
import logging
from pathlib import Path

from config.settings import settings
from src.processing.metadata_extractor import extract_metadata
from src.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def main():
    processed_dir = Path(settings.processed_dir)
    input_path = processed_dir / "chunks.json"
    output_path = processed_dir / "enriched_chunks.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {input_path}. "
            "Run 'python -m scripts.build_chunks' first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    enriched_chunks = [
        extract_metadata(chunk)
        for chunk in chunks
    ]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            enriched_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    companies = {
        chunk["esg_metadata"]["company"]
        for chunk in enriched_chunks
        if chunk["esg_metadata"]["company"]
    }

    logger.info("Input chunks: %d", len(chunks))
    logger.info("Enriched chunks: %d", len(enriched_chunks))
    logger.info("Recognized companies: %s", sorted(companies))
    logger.info("Saved enriched data to: %s", output_path)


if __name__ == "__main__":
    main()