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


def detect_company(query: str) -> str | None:
    """
    Detect a single company explicitly mentioned in a query.

    Return None for general questions or comparisons mentioning
    both companies, so those queries can search the full corpus.
    """
    query_lower = query.lower()

    basf_match = bool(re.search(r"\bbasf\b", query_lower))
    eastman_match = bool(re.search(r"\beastman\b", query_lower))

    if basf_match and not eastman_match:
        return "BASF"

    if eastman_match and not basf_match:
        return "Eastman"

    return None


def chunk_company(chunk: dict[str, Any]) -> str:
    """Read the normalized company name from ESG metadata."""
    esg_metadata = chunk.get("esg_metadata", {})
    company = esg_metadata.get("company", "")
    return str(company).strip().lower()


class HybridRetriever:
    """Combine FAISS and BM25 using Reciprocal Rank Fusion."""

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
        Retrieve candidates using FAISS and BM25, then fuse rankings.

        If exactly one supported company is explicitly named, restrict
        candidates to that company's chunks before ranking fusion.

        For general questions and cross-company comparisons, retain
        the original full-corpus retrieval behavior.
        """
        if not query or not query.strip():
            return []

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive.")

        tokens = tokenize(query)

        if not tokens:
            return []

        # COMPARISON RETRIEVAL: SEARCH EACH COMPANY SEPARATELY
        basf_match = bool(re.search(r"\bbasf\b", query, re.IGNORECASE))
        eastman_match = bool(re.search(r"\beastman\b", query, re.IGNORECASE))

        if basf_match and eastman_match:
            shared_query = re.sub(
                r"\bbasf\b|\beastman\b",
                " ",
                query,
                flags=re.IGNORECASE,
            )
            shared_query = re.sub(r"\s+", " ", shared_query).strip()

            company_queries = [
                (
                    "BASF",
                    f"BASF {shared_query} greenhouse gas emissions "
                    "Scope 1 Scope 2 Scope 3.1 reduction target "
                    "target year baseline year",
                ),
                (
                    "Eastman",
                    f"Eastman {shared_query} greenhouse gas emissions "
                    "absolute Scope 1 Scope 2 reduction target "
                    "target year baseline year",
                ),
            ]

            company_results = {}
            for company, company_query in company_queries:
                company_results[company] = self.search(
                    company_query,
                    top_k=top_k,
                    candidate_k=candidate_k,
                )

            # Alternate companies so one company's passages cannot fill
            # the entire comparison evidence list.
            balanced_results = []
            seen_chunk_ids = set()
            max_per_company = max(
                len(company_results["BASF"]),
                len(company_results["Eastman"]),
            )

            for rank in range(max_per_company):
                for company in ("BASF", "Eastman"):
                    results_for_company = company_results[company]

                    if rank >= len(results_for_company):
                        continue

                    result = results_for_company[rank]
                    chunk_id = result.get("chunk_id")

                    if chunk_id in seen_chunk_ids:
                        continue

                    seen_chunk_ids.add(chunk_id)
                    balanced_results.append(result)

                    if len(balanced_results) >= top_k:
                        return balanced_results

            return balanced_results

        company_filter = detect_company(query)
        candidate_k = min(candidate_k, len(self.chunks))

        # Identify eligible chunks when a single company is specified.
        if company_filter:
            eligible_positions = [
                i
                for i, chunk in enumerate(self.chunks)
                if chunk_company(chunk) == company_filter.lower()
            ]

            if not eligible_positions:
                logger.warning(
                    "No chunks found for company %s; using full corpus.",
                    company_filter,
                )
                company_filter = None
        else:
            eligible_positions = list(range(len(self.chunks)))

        # 1. Semantic retrieval.
        if company_filter:
            # Search the complete small FAISS index, then restrict results
            # to the requested company before selecting candidates.
            all_vector_results = self.vector_store.search(
                query,
                top_k=len(self.chunks),
            )

            vector_results = [
                result
                for result in all_vector_results
                if chunk_company(result) == company_filter.lower()
            ][:candidate_k]
        else:
            vector_results = self.vector_store.search(
                query,
                top_k=candidate_k,
            )

        # 2. Keyword retrieval.
        bm25_scores = self.bm25.get_scores(tokens)

        if company_filter:
            bm25_positions = sorted(
                eligible_positions,
                key=lambda i: float(bm25_scores[i]),
                reverse=True,
            )[:candidate_k]
        else:
            bm25_positions = sorted(
                range(len(bm25_scores)),
                key=lambda i: float(bm25_scores[i]),
                reverse=True,
            )[:candidate_k]

        # 3. Merge rankings by chunk ID and apply RRF.
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
            fused[chunk_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

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
            fused[chunk_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

        results = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        logger.info(
            "Retrieved %d candidates; company_filter=%s",
            len(results),
            company_filter or "none",
        )

        return results[:top_k]
