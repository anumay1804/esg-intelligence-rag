"""Lightweight graph-assisted candidate expansion for ESG RAG."""
import json
import re
from pathlib import Path
from typing import Any
from config.settings import settings

CATEGORY_ALIASES = {
    "greenhouse_gas_emissions": ("emission", "emissions", "ghg", "greenhouse", "carbon", "co2"),
    "energy": ("energy", "electricity", "power", "renewable", "electric"),
    "water": ("water", "withdrawal", "discharge"),
    "waste": ("waste", "recycling", "recycled"),
    "safety": ("safety", "injury", "injuries", "ltir"),
    "targets": ("target", "targets", "net zero", "decarbonization", "decarbonisation"),
}

class ESGGraphRetriever:
    """Use graph-connected chunk metadata to expand a hybrid retrieval result.

    This is graph-assisted retrieval, not an LLM-built ontology. The graph is
    only used to discover related source chunks; final answers still cite text.
    """
    def __init__(self, hybrid=None, graph_dir: str | Path | None = None):
        if hybrid is None:
            from src.retrieval.hybrid_retriever import HybridRetriever
            hybrid = HybridRetriever()
        self.hybrid = hybrid
        self.graph_dir = Path(graph_dir or (Path(settings.artifacts_dir) / "knowledge_graph"))
        manifest_path = self.graph_dir / "graph_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Graph manifest not found: {manifest_path}. Run python -m scripts.build_knowledge_graph first.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.graph_chunks = self.manifest.get("chunks", {})
        self.entities = self.manifest.get("entities", {})
        self.entity_to_chunks = {key: set(item.get("chunk_ids", [])) for key, item in self.entities.items()}

    @staticmethod
    def _query_features(query: str) -> dict[str, set[str]]:
        q = query.casefold()
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", q))
        scopes = set()
        for match in re.findall(r"scope\s*([123](?:\.1)?)", q):
            scopes.add("Scope " + match)
            # Scope 3.1 is a Scope 3 category, not a separate GHG Protocol scope.
            # Match both the explicit query label and its parent Scope 3 entity.
            if "." in match:
                scopes.add("Scope " + match.split(".", 1)[0])
        companies = {name for name in ("basf", "dow", "covestro", "sabic", "lyondellbasell") if name in q}
        categories = {category for category, aliases in CATEGORY_ALIASES.items() if any(alias in q for alias in aliases)}
        return {"years": years, "scopes": scopes, "companies": companies, "categories": categories}

    def search(self, query: str, top_k: int = 8, candidate_k: int = 40) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        base = self.hybrid.search(query, top_k=top_k, candidate_k=candidate_k)
        scores: dict[str, float] = {}
        records: dict[str, dict] = {}
        # Baseline hybrid candidates remain in the candidate set.
        for rank, item in enumerate(base, start=1):
            cid = str(item.get("chunk_id", ""))
            if not cid: continue
            records[cid] = dict(item)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
        features = self._query_features(query)
        seed_nodes = []
        for node_id, entity in self.entities.items():
            kind = entity.get("type")
            label = str(entity.get("label", ""))
            normalized = label.casefold()
            matched = False
            if kind == "year" and normalized in features["years"]: matched = True
            elif kind == "scope" and any(s.casefold() == normalized for s in features["scopes"]): matched = True
            elif kind == "company" and normalized in features["companies"]: matched = True
            elif kind == "metric_category" and normalized in features["categories"]: matched = True
            if matched: seed_nodes.append(node_id)
        # Match metadata nodes, then expand to their source chunks. Rank graph
        # candidates by how many query entities they share, not by chunk ID.
        graph_candidate_scores: dict[str, float] = {}
        type_weights = {"company": 0.30, "year": 0.20, "scope": 0.30, "metric_category": 0.30}
        for node_id in seed_nodes:
            entity = self.entities.get(node_id, {})
            weight = type_weights.get(entity.get("type"), 0.15)
            for cid in self.entity_to_chunks.get(node_id, set()):
                graph_candidate_scores[cid] = graph_candidate_scores.get(cid, 0.0) + weight
        # Add lexical overlap candidates so queries like "renewable electricity"
        # still work when extracted metadata labels are broad or missing.
        q_tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))
        lexical_scored = []
        for cid, chunk in self.graph_chunks.items():
            text_tokens = set(re.findall(r"[a-z0-9]+", chunk.get("text", "").casefold()))
            overlap = len(q_tokens & text_tokens) / max(1, len(q_tokens))
            if overlap >= 0.25:
                lexical_scored.append((overlap, cid))
        lexical_scored.sort(reverse=True)
        for overlap, cid in lexical_scored[:candidate_k]:
            graph_candidate_scores[cid] = graph_candidate_scores.get(cid, 0.0) + overlap
        ranked_graph_candidates = sorted(
            graph_candidate_scores.items(), key=lambda item: item[1], reverse=True
        )[:candidate_k]
        baseline_ids = {str(x.get("chunk_id", "")) for x in base}
        # Add a modest RRF contribution for graph-derived candidates.
        for rank, (cid, graph_score) in enumerate(ranked_graph_candidates, start=1):
            if cid not in records:
                chunk = self.graph_chunks.get(cid)
                if not chunk: continue
                records[cid] = dict(chunk)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
            records[cid]["graph_augmented"] = cid not in baseline_ids
            records[cid]["graph_match_score"] = round(graph_score, 4)
        ordered = sorted(records.values(), key=lambda x: scores.get(str(x.get("chunk_id", "")), 0.0), reverse=True)
        for item in ordered:
            item["graph_rrf_score"] = scores.get(str(item.get("chunk_id", "")), 0.0)
        return ordered[:max(1, top_k)]
