# Final QA Report — ESG Intelligence Assistant

Package validation date: 2026-10-30

## Included functionality

- PDF text extraction with page-level provenance.
- Paragraph-aware chunking and heuristic ESG metadata enrichment.
- Local Sentence Transformers embeddings and FAISS index.
- BM25 keyword retrieval with Reciprocal Rank Fusion.
- Cross-encoder reranking.
- Gemini grounded-answer generation with inline citation IDs and actionable API error messages.
- Streamlit interface with hybrid and graph-assisted retrieval modes.
- Graph builder that creates a provenance-preserving JSON manifest and GraphML file.
- Retrieval evaluation, graph comparison, answer/citation diagnostics, optional Gemini claim-support audit, and gold-answer validation.

## Checks performed for this package

- Python syntax compilation: all 48 Python files passed.
- Dependency-light unit tests: 22 passed.
- Gemini generator edge-case tests: 7 passed using a small local SDK stub because the packaging environment does not have `google-genai` installed.
- Gold-answer data validation: 20 retrieval questions, 12 gold answers, and all referenced PDF files/page numbers validated.
- Data consistency: 1,500 processed chunks, 1,500 indexed chunk records, and 1,500 graph-manifest chunks. Indexed chunk IDs match the processed chunk IDs. The rebuilt graph has 2,036 nodes (1,500 source chunks plus 536 metadata/value entities) and 7,167 edges.
- Secret hygiene: local `.env` file removed from the distributable package; `.env.example` contains only a placeholder.

## Run this in the target Windows environment

Use Python 3.11, install `requirements.txt`, add a valid Gemini key to a local `.env`, then run:

```powershell
python -m pytest -q
python -m evaluation.validate_gold_answers
streamlit run app.py
```

The packaging environment did not include the native retrieval/UI dependencies (`faiss-cpu`, `sentence-transformers`, `rank-bm25`, `streamlit`) or the Google Gemini SDK, so the complete end-to-end UI/API flow could not be executed here. Install the listed requirements and verify the first live Gemini request in the user's environment before the interview.

## Important limitations

This is an interview proof of concept, not a production-certified ESG system. Keyword overlap and citation-ID validity are diagnostics, not proof of semantic correctness. Manually inspect important numeric claims, units, scope boundaries, baselines, and report pages before presenting them as validated.
