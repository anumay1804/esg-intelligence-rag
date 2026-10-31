import logging
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from config.settings import settings
from src.retrieval.vector_store import ESGVectorStore


logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Normalize text for keyword-based retrieval."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """Combine FAISS semantic search and BM25 keyword search using RRF."""

    def __init__(
        self,
        index_dir: str | Path | None = None,
        rrf_k: int = 60,
    ):
        self.index_dir = Path(index_dir or settings.indexes_dir)
        self.rrf_k = rrf_k

        self.vector_store = ESGVectorStore()
        self.vector_store.load(self.index_dir)

        self.chunks = self.vector_store.chunks

        tokenized_chunks = [
            tokenize(chunk.get("text", ""))
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

        logger.info(
            "Hybrid retriever initialized with %d chunks",
            len(self.chunks),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Retrieve candidates from FAISS and BM25, then fuse their rankings.

        RRF score = sum(1 / (rrf_k + rank)) across retrieval systems.
        Ranks are 1-based.
        """
        if not query or not query.strip():
            return []

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive.")

        tokens = tokenize(query)

        if not tokens:
            return []

        candidate_k = min(candidate_k, len(self.chunks))

        # 1. Semantic retrieval.
        vector_results = self.vector_store.search(
            query,
            top_k=candidate_k,
        )

        # 2. Keyword retrieval.
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_positions = sorted(
            range(len(bm25_scores)),
            key=lambda i: float(bm25_scores[i]),
            reverse=True,
        )[:candidate_k]

        # Use chunk IDs to merge the two rankings.
        fused: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                }

            fused[chunk_id]["vector_rank"] = rank
            fused[chunk_id]["rrf_score"] += 1 / (self.rrf_k + rank)

        for rank, position in enumerate(bm25_positions, start=1):
            chunk = self.chunks[position]
            chunk_id = chunk["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **chunk,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "bm25_rank": None,
                    "similarity_score": None,
                    "bm25_score": None,
                }

            fused[chunk_id]["bm25_rank"] = rank
            fused[chunk_id]["bm25_score"] = float(
                bm25_scores[position]
            )
            fused[chunk_id]["rrf_score"] += 1 / (self.rrf_k + rank)

        # 3. Sort by the combined ranking score.
        results = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return results[:top_k]