# ESG Intelligence Assistant — Hybrid RAG + Graph-assisted Retrieval

An interview-ready local proof of concept that answers questions over the bundled BASF 2024 sustainability/annual reports. The pipeline preserves report page provenance and combines local semantic search, BM25 keyword search, Reciprocal Rank Fusion, cross-encoder reranking, optional graph-assisted candidate expansion, and Google Gemini answer generation.

## What is included

- Two public BASF 2024 PDF reports in `data/raw/`.
- Preprocessed documents/chunks and ESG metadata in `data/processed/`.
- A prebuilt FAISS index and matching chunk metadata in `data/indexes/`.
- A graph manifest and GraphML file in `artifacts/knowledge_graph/`.
- Streamlit UI (`app.py`) with hybrid and graph-assisted retrieval modes.
- CLI for asking questions and inspecting retrieval.
- Retrieval, answer/citation diagnostics, gold-answer data, and unit tests.

**The bundled index is ready to use.** You do not need to rebuild the PDFs/index before the first demo. The first run may download the embedding and cross-encoder model files from Hugging Face; internet access is needed for that and for Gemini API calls.

## Recommended environment

- Windows 10/11, 64-bit
- **Python 3.11 (recommended)**. Native ML libraries such as FAISS can lag behind the newest Python release.
- Internet access for initial dependency/model downloads and Gemini API requests
- A Google AI Studio API key with access to the configured Gemini model

## Windows PowerShell setup

Open PowerShell in the extracted project folder (the folder containing `app.py`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

In `.env`, replace `your_gemini_api_key_here` with your own key. Keep this file private. Never paste your key into a chat or commit it to source control.

If `py -3.11` is not available, install Python 3.11 first. If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## Run the app

From the project root, with the virtual environment activated:

```powershell
python -m pytest -q
python -m evaluation.validate_gold_answers
streamlit run app.py
```

Streamlit prints a local address, usually `http://localhost:8501`. Open it in your browser. Ask an ESG question, then inspect the citations and retrieved evidence. Try the graph-assisted retrieval toggle and compare the same question in both modes.

Example:
> What is BASF's greenhouse gas emissions reduction target for 2030?

If the app cannot start, first confirm you are in the folder containing `app.py`, your virtual environment is active, and `pip install -r requirements.txt` completed successfully.

## CLI usage

```powershell
python -m scripts.ask_esg
python -m scripts.search_hybrid
python -m scripts.search_reranked
```

## Rebuild the local index (only if PDFs or processing logic changed)

The following pipeline overwrites the processed JSON and FAISS index. Keep a backup if you need the current bundled data.

```powershell
python -m scripts.ingest_documents
python -m scripts.build_chunks
python -m scripts.enrich_metadata
python -m scripts.build_vector_index
python -m scripts.build_knowledge_graph
```

## Evaluation

```powershell
# Retrieval keyword diagnostics (local; no Gemini generation)
python -m evaluation.evaluate_retrieval

# Compare hybrid vs graph-assisted retrieval (local; no Gemini generation)
python -m scripts.run_evaluation

# Answer generation + citation-ID/coverage/lexical-overlap diagnostics.
# Calls Gemini for each question that has evidence.
python -m evaluation.evaluate_answers

# Optional Gemini-as-judge citation audit. This consumes additional API quota.
python -m evaluation.validate_citation_support
```

Results are written under `evaluation/`. Evaluation metrics are diagnostics:
- Keyword overlap does not prove a passage is relevant.
- Citation-ID validity does not prove that a cited passage supports a claim.
- Lexical overlap is not semantic entailment.
- Gemini-as-judge output is not ground truth.
- Manually verify important figures, units, scope boundaries, reporting years, and baseline years in the cited PDF pages before presenting a claim as validated.

## Architecture

1. **PDF ingestion:** PyMuPDF extracts page text and document metadata.
2. **Chunking:** Paragraph-aware chunks preserve source file and 1-based PDF page number.
3. **ESG metadata:** Heuristics identify company, years, scopes, candidate metric categories, and numeric mentions.
4. **Dense retrieval:** Sentence Transformers (`all-MiniLM-L6-v2`) creates normalized embeddings; FAISS inner-product search approximates cosine similarity.
5. **Keyword retrieval:** BM25 finds exact terms/numbers and complements semantic search.
6. **Fusion:** Reciprocal Rank Fusion combines the two ranked lists.
7. **Reranking:** A cross-encoder reranks candidate passages for the query.
8. **Optional graph-assisted retrieval:** A NetworkX graph links chunks to extracted company/year/scope/category metadata and expands candidate retrieval. Final answers still cite original report passages.
9. **Generation:** Gemini is prompted to answer only from supplied evidence, cite factual claims, preserve units and boundaries, and abstain when evidence is insufficient.
10. **UI:** Streamlit displays the answer, source file/page references, timing, and retrieved passages.

## Cost and privacy notes

PDF parsing, chunking, metadata extraction, embeddings, FAISS, BM25, graph construction, and retrieval run locally. Gemini answer generation and the optional claim-support judge use API calls and quota. Initial model downloads require internet access. Do not add confidential reports unless you are authorized to process them.

## Known limitations

- PDF table extraction and multi-column reading order can be imperfect; scanned PDFs may need OCR.
- ESG metadata and graph edges are heuristic and should be validated.
- The graph is a candidate-expansion aid, not a verified ESG ontology.
- The dataset contains only two BASF reports; it is not a multi-company production system.
- Local models and Streamlit are suitable for a single-user demo, not a production concurrency or security guarantee.
- This is an interview POC, not a certified ESG reporting platform.
