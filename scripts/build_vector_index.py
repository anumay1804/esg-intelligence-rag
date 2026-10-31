import json
import logging
from pathlib import Path

from config.settings import settings
from src.retrieval.vector_store import ESGVectorStore
from src.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def main():
    processed_dir = Path(settings.processed_dir)
    index_dir = Path(settings.indexes_dir)

    input_path = processed_dir / "enriched_chunks.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Enriched chunks not found: {input_path}. "
            "Run 'python -m scripts.enrich_metadata' first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    vector_store = ESGVectorStore()
    vector_store.build(chunks)
    vector_store.save(index_dir)

    logger.info("Vector index created successfully.")


if __name__ == "__main__":
    main()