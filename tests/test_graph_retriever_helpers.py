from src.retrieval.graph_retriever import ESGGraphRetriever

def test_query_features_detect_year_scope_company_and_energy():
    features = ESGGraphRetriever._query_features(
        "For BASF, what is the Scope 3.1 emissions target for renewable energy in 2030?"
    )
    assert "2030" in features["years"]
    assert "Scope 3.1" in features["scopes"]
    assert "Scope 3" in features["scopes"]  # Scope 3.1 maps to parent Scope 3 graph entity
    assert "basf" in features["companies"]
    assert "energy" in features["categories"]
    assert "greenhouse_gas_emissions" in features["categories"]

def test_graph_retriever_expands_to_connected_chunks(tmp_path):
    import json
    from src.retrieval.graph_retriever import ESGGraphRetriever

    graph_dir = tmp_path / "knowledge_graph"
    graph_dir.mkdir()
    manifest = {
        "chunks": {
            "chunk-a": {"chunk_id": "chunk-a", "source_file": "basf.pdf", "page": 10,
                        "text": "BASF emissions target for 2030", "esg_metadata": {}},
            "chunk-b": {"chunk_id": "chunk-b", "source_file": "basf.pdf", "page": 11,
                        "text": "Other report content about energy", "esg_metadata": {}},
        },
        "entities": {
            "company:basf": {"label": "BASF", "type": "company", "chunk_ids": ["chunk-a", "chunk-b"]},
            "year:2030": {"label": "2030", "type": "year", "chunk_ids": ["chunk-a"]},
            "category:greenhouse_gas_emissions": {"label": "greenhouse_gas_emissions", "type": "metric_category", "chunk_ids": ["chunk-a"]},
        },
    }
    (graph_dir / "graph_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class FakeHybrid:
        def search(self, query, top_k=5, candidate_k=20):
            return [{"chunk_id": "chunk-b", "source_file": "basf.pdf", "page": 11,
                     "text": "Other report content about energy", "rrf_score": 0.01}]

    retriever = ESGGraphRetriever(hybrid=FakeHybrid(), graph_dir=graph_dir)
    results = retriever.search("What is BASF emissions target for 2030?", top_k=5)
    ids = {item["chunk_id"] for item in results}
    assert "chunk-a" in ids
    assert "chunk-b" in ids
