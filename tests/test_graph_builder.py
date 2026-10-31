from scripts.build_knowledge_graph import build_graph

def test_graph_builder_preserves_provenance_and_entities():
    chunks = [{
        "chunk_id": "basf:page_4:chunk_1",
        "source_file": "basf_report.pdf",
        "page": 4,
        "text": "BASF reduced Scope 1 emissions in 2024 and targets renewable energy by 2030.",
        "esg_metadata": {
            "company": "BASF",
            "years_mentioned": [2024, 2030],
            "scopes_mentioned": ["Scope 1"],
            "metric_categories": ["greenhouse_gas_emissions", "energy", "targets"],
        },
    }]
    graph, manifest = build_graph(chunks)
    assert manifest["chunk_count"] == 1
    assert "basf:page_4:chunk_1" in manifest["chunks"]
    assert manifest["entities"]["company:basf"]["chunk_ids"] == ["basf:page_4:chunk_1"]
    assert graph.has_edge("basf:page_4:chunk_1", "year:2030")
