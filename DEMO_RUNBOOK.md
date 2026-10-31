# ESG Intelligence RAG — Interview Demo Runbook

## 1. What to demonstrate (5–7 minutes)

1. Explain the problem: search multiple chemical-industry sustainability reports and answer questions with page-level provenance.
2. Show the pipeline: PDF ingestion → paragraph-aware chunking → ESG metadata → local embeddings/FAISS + BM25 → Reciprocal Rank Fusion → cross-encoder reranking → Gemini response with citations.
3. Ask a numeric target question, for example: `What is BASF's greenhouse gas emissions reduction target for 2030?`
4. Click/open the displayed source references and the retrieved-evidence expander. Verify the exact figure, scope, year, baseline, and unit against the cited report page.
5. Toggle **Use graph-assisted retrieval**, repeat the same query, and explain that graph links expand candidate chunks using company/year/scope/category mentions. Do not claim it improves quality unless the measured comparison supports that.
6. Show `evaluation/gold_answers.json` and explain that the 12 expected answers are linked to report PDF pages. Open the cited pages for at least two numeric claims (Scope 1+2 target on PDF page 195; renewable electricity share on PDF page 189).
7. Show the retrieval evaluation and graph comparison output; describe metrics as keyword-based heuristics, not formal accuracy. Run `python -m evaluation.validate_gold_answers` and `python -m pytest -q` to show dataset integrity and edge-case tests.
8. Demonstrate an unsupported question unrelated to the BASF reports and explain that the model should abstain when evidence does not answer it. Do not claim this is guaranteed solely by the prompt; review the result live.
9. Finish with limitations: heuristic metadata/graph extraction, PDF table extraction, LLM-as-judge is not ground truth, small benchmark, manual citation review, and no cloud deployment/load test yet.

## 2. Windows PowerShell commands

Run these from the project root with the virtual environment active.

```powershell
# Install dependencies if needed
python -m pip install -r requirements.txt

# Check project tests (does not call Gemini)
python -m pytest -q

# Rebuild only if source PDFs or processing logic changed
python -m scripts.ingest_documents
python -m scripts.build_chunks
python -m scripts.enrich_metadata
python -m scripts.build_vector_index
python -m scripts.build_knowledge_graph

# Retrieval-only evaluation (no Gemini generation calls)
python -m evaluation.evaluate_retrieval

# Compare baseline hybrid vs graph-assisted retrieval (no Gemini generation calls)
python -m scripts.run_evaluation

# Validate gold answer dataset and report-page references (no Gemini call)
python -m evaluation.validate_gold_answers

# Answer/citation diagnostics (calls Gemini once per question)
python -m evaluation.evaluate_answers

# Optional citation-support judge (additional Gemini calls; not ground truth)
python -m evaluation.validate_citation_support

# Launch the demo
streamlit run app.py
```

## 3. Environment and secrets

- Copy `.env.example` to `.env` and set `GEMINI_API_KEY` locally.
- `GEMINI_MODEL=gemini-3.1-flash-lite` is the current project default; use a model enabled for your Google AI Studio account.
- Never commit `.env`, API keys, or private documents.
- Local parsing, embeddings, FAISS, BM25, graph construction, and retrieval are local. Gemini answer generation and optional LLM-as-judge calls use API quota.

## 4. How to report metrics honestly

- **Citation coverage** means the heuristic detected a citation in every factual sentence it identified; it does not prove that the source entails the claim.
- **Citation ID validity** means each cited ID exists in the answer's retrieved source list.
- **Keyword overlap / Hit@5 / MRR** use heuristic labels and can overestimate relevance.
- Report graph-vs-hybrid numbers only after running `python -m scripts.run_evaluation` locally on the same index and machine.
- Do not claim production readiness or a guaranteed monthly cost from a local POC. Measure API usage and hosting costs for the expected workload.
