import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings


logger = logging.getLogger(__name__)


class ESGVectorStore:
    """Local embedding and FAISS search for ESG document chunks."""

    def __init__(self, model_name: str | None = None):
        self.model_name = (
            model_name
            or settings.embedding_model
        )
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.chunks: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Create a FAISS index from the supplied chunks."""
        if not chunks:
            raise ValueError("Cannot build an index from empty chunks.")

        texts = [
            chunk.get("text", "").strip()
            for chunk in chunks
        ]

        valid_positions = [
            i for i, text in enumerate(texts) if text
        ]

        if not valid_positions:
            raise ValueError("No non-empty chunk text was found.")

        self.chunks = [chunks[i] for i in valid_positions]
        valid_texts = [texts[i] for i in valid_positions]

        logger.info("Generating embeddings for %d chunks", len(valid_texts))

        embeddings = self.model.encode(
            valid_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Inner product over normalized vectors is cosine similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        logger.info(
            "FAISS index built: %d vectors, %d dimensions",
            self.index.ntotal,
            embeddings.shape[1],
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most semantically similar chunks."""
        if self.index is None:
            raise RuntimeError("Build or load the FAISS index first.")

        if not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue

            chunk = self.chunks[int(index_position)]

            results.append({
                **chunk,
                "similarity_score": float(score),
            })

        return results

    def save(self, index_dir: str | Path) -> None:
        """Save the FAISS index and its associated chunks."""
        if self.index is None:
            raise RuntimeError("Build the index before saving it.")

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(index_dir / "esg.index"),
        )

        with (index_dir / "indexed_chunks.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("Saved vector index to %s", index_dir)

    def load(self, index_dir: str | Path) -> None:
        """Load a saved FAISS index and its associated chunks."""
        index_dir = Path(index_dir)

        index_path = index_dir / "esg.index"
        chunks_path = index_dir / "indexed_chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"FAISS index or chunk metadata not found in {index_dir}"
            )

        self.index = faiss.read_index(str(index_path))

        with chunks_path.open("r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "The number of FAISS vectors does not match indexed chunks."
            )

        logger.info("Loaded %d indexed chunks", len(self.chunks))