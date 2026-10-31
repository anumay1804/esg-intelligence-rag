"""Build a provenance-preserving ESG knowledge graph from enriched chunks.

Usage: python -m scripts.build_knowledge_graph
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import networkx as nx

from config.settings import settings
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

CATEGORY_ALIASES = {
    "greenhouse_gas_emissions": ("emission", "emissions", "ghg", "greenhouse", "carbon", "co2"),
    "energy": ("energy", "electricity", "power", "renewable", "electric"),
    "water": ("water", "withdrawal", "discharge"),
    "waste": ("waste", "recycling", "recycled"),
    "safety": ("safety", "injury", "injuries", "ltir"),
    "targets": ("target", "targets", "net zero", "decarbonization", "decarbonisation"),
}


def _add_entity(entities: dict[str, dict[str, Any]], entity_id: str,
                label: str, kind: str, chunk_id: str) -> None:
    entity = entities.setdefault(entity_id, {
        "label": label, "type": kind, "chunk_ids": []
    })
    if chunk_id not in entity["chunk_ids"]:
        entity["chunk_ids"].append(chunk_id)


def build_graph(chunks: list[dict[str, Any]]) -> tuple[nx.Graph, dict[str, Any]]:
    """Create a graph and JSON manifest; each entity links back to source chunks."""
    graph = nx.Graph()
    entities: dict[str, dict[str, Any]] = {}
    chunk_map: dict[str, dict[str, Any]] = {}

    for index, raw in enumerate(chunks, start=1):
        chunk = dict(raw)
        text = str(chunk.get("text", "") or "").strip()
        if not text:
            continue
        chunk_id = str(chunk.get("chunk_id") or f"chunk-{index}")
        chunk["chunk_id"] = chunk_id
        chunk_map[chunk_id] = chunk
        graph.add_node(chunk_id, node_type="chunk",
                       source_file=str(chunk.get("source_file", "")),
                       page=str(chunk.get("page", "")),
                       text=text[:500])

        meta = chunk.get("esg_metadata", {}) or {}
        company = meta.get("company")
        years = meta.get("years_mentioned", [])
        scopes = meta.get("scopes_mentioned", [])
        categories = meta.get("metric_categories", [])

        source_file = str(chunk.get("source_file", "Unknown source"))
        specs = [(f"document:{source_file}", source_file, "document")]
        if company:
            specs.append((f"company:{str(company).casefold()}", str(company), "company"))
        specs.extend((f"year:{year}", str(year), "year") for year in years)
        specs.extend((f"scope:{str(scope).casefold().replace(' ', '_')}", str(scope), "scope")
                     for scope in scopes)

        for category in categories:
            label = str(category)
            specs.append((f"category:{label}", label, "metric_category"))

        # Link repeated numeric mentions by normalized value + unit. This is a
        # discovery aid only: the nearby context remains attached to the chunk.
        for mention in meta.get("value_mentions", []) or []:
            value = str(mention.get("value", "")).strip()
            unit = re.sub(r"\s+", " ", str(mention.get("unit", "")).strip())
            if value and unit:
                label = f"{value} {unit}"
                specs.append((f"value:{label.casefold()}", label, "value_mention"))

        # Fallback heuristic category links ensure basic graph recall when
        # metadata extraction has not assigned a category.
        lowered = text.casefold()
        known = set(categories)
        for category, aliases in CATEGORY_ALIASES.items():
            if category not in known and any(alias in lowered for alias in aliases):
                specs.append((f"category:{category}", category, "metric_category"))

        for entity_id, label, kind in specs:
            _add_entity(entities, entity_id, label, kind, chunk_id)
            graph.add_node(entity_id, node_type=kind, label=label)
            graph.add_edge(chunk_id, entity_id, relation="mentions")

    manifest = {
        "schema_version": 1,
        "chunk_count": len(chunk_map),
        "entity_count": len(entities),
        "chunks": chunk_map,
        "entities": entities,
    }
    return graph, manifest


def main() -> int:
    configure_logging()
    input_path = Path(settings.processed_dir) / "enriched_chunks.json"
    output_dir = Path(settings.artifacts_dir) / "knowledge_graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Enriched chunks not found: {input_path}. Run "
            "'python -m scripts.ingest_documents', 'python -m scripts.build_chunks', "
            "and 'python -m scripts.enrich_metadata' first."
        )
    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    graph, manifest = build_graph(chunks)
    manifest_path = output_dir / "graph_manifest.json"
    graph_path = output_dir / "esg_knowledge_graph.graphml"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    nx.write_graphml(graph, graph_path)
    logger.info("Knowledge graph built: %d chunks, %d entities, %d edges",
                manifest["chunk_count"], manifest["entity_count"], graph.number_of_edges())
    logger.info("Saved manifest: %s", manifest_path)
    logger.info("Saved GraphML: %s", graph_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
