from sentence_transformers import CrossEncoder


class ESGReranker:
    """Rerank retrieved ESG passages using a cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        Reorder candidate passages according to their relevance to the query.

        Args:
            query: User's ESG question.
            candidates: Passages returned by hybrid retrieval.
            top_k: Number of reranked passages to return.

        Returns:
            Passages sorted by cross-encoder relevance score.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        # Pair the same query with each candidate passage.
        pairs = [
            (query, candidate.get("text", ""))
            for candidate in candidates
        ]

        # Calculate a relevance score for each query-passage pair.
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Copy candidates so that the original results remain unchanged.
        reranked = []

        for candidate, score in zip(candidates, scores):
            if not candidate.get("text", "").strip():
                continue

            result = candidate.copy()
            result["reranker_score"] = float(score)
            reranked.append(result)

        # Highest cross-encoder score first.
        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return reranked[:max(1, top_k)]