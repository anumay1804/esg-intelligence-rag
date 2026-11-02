# ESG Intelligence Assistant
### Hybrid Retrieval-Augmented Generation (RAG) with Graph-Assisted Retrieval

An AI-powered application for querying corporate sustainability reports using hybrid information retrieval, semantic search, keyword matching, cross-encoder reranking, and Google Gemini. The system retrieves relevant evidence from ESG documents and generates answers with source citations and page-level provenance.

## Project Overview

The ESG Intelligence Assistant enables users to explore sustainability disclosures and retrieve information about greenhouse gas emissions, climate targets, renewable energy, environmental performance, and corporate ESG commitments.

The retrieval pipeline combines dense vector search, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), and cross-encoder reranking. An optional graph-assisted retrieval mode expands candidate passages using relationships between document chunks and extracted ESG metadata.

The application uses a prebuilt FAISS index and processed document data to support an initial demonstration without rebuilding the document corpus.

## Key Features

- **Hybrid Retrieval:** Combines semantic vector search with BM25 keyword retrieval.
- **Reciprocal Rank Fusion:** Merges results from multiple retrieval methods.
- **Cross-Encoder Reranking:** Reorders retrieved passages according to query relevance.
- **Graph-Assisted Retrieval:** Uses a NetworkX knowledge graph to support candidate expansion.
- **LLM-Based Answer Generation:** Uses Google Gemini to generate responses grounded in retrieved evidence.
- **Source Citations:** Displays source document names and PDF page references.
- **Evidence Inspection:** Allows users to inspect retrieved passages supporting generated responses.
- **Evaluation Tools:** Includes retrieval diagnostics, gold-answer datasets, citation checks, and automated tests.
- **Command-Line Utilities:** Provides scripts for querying documents and inspecting retrieval results.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Language Model | Google Gemini |
| Embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Ranking Fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Knowledge Graph | NetworkX |
| PDF Processing | PyMuPDF |
| Data Processing | JSON and Python-based metadata extraction |
| Testing | pytest |

## Project Structure

```text
esg-intelligence-rag/
├── app.py
├── config/
├── data/
│   ├── indexes/
│   ├── processed/
│   └── raw/                 # Original source documents, kept locally
├── artifacts/
│   └── knowledge_graph/
├── evaluation/
│   ├── evaluate_answers.py
│   ├── gold_answers.json
│   └── test_questions.json
├── scripts/
│   ├── ask_esg.py
│   ├── search_hybrid.py
│   └── search_reranked.py
├── src/
│   ├── generation/
│   ├── processing/
│   └── retrieval/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── START_DEMO.bat
```

*The tree highlights the main project components; additional files may be present in individual directories.*

## Data Sources

The local document corpus contains two BASF 2024 reports:

- BASF 2024 sustainability report
- BASF 2024 annual/reporting document

The project includes preprocessed document records, text chunks, ESG metadata, a FAISS index, and knowledge-graph artifacts generated from the source corpus.

The original PDF files remain local and are not included in the Git repository.

## System Architecture

The application follows a retrieval-augmented generation architecture.

1. **Document Ingestion:** Extracts text and page-level information from PDF documents using PyMuPDF.
2. **Chunking:** Splits document content into manageable passages while retaining source and page metadata.
3. **Metadata Extraction:** Identifies candidate company names, reporting years, emissions scopes, ESG categories, and numeric mentions using heuristic processing.
4. **Dense Retrieval:** Generates sentence embeddings and uses FAISS similarity search to identify semantically relevant passages.
5. **Keyword Retrieval:** Uses BM25 to retrieve passages containing relevant terminology, metrics, and keywords.
6. **Result Fusion:** Applies Reciprocal Rank Fusion to combine the dense and keyword retrieval rankings.
7. **Reranking:** Uses a cross-encoder to refine the relevance ordering of candidate passages.
8. **Optional Graph-Assisted Retrieval:** Uses relationships between document chunks and extracted metadata to support candidate expansion.
9. **Answer Generation:** Passes retrieved evidence to Gemini to generate a response grounded in the available context.
10. **Evidence Presentation:** Displays the answer, source references, page numbers, and retrieved passages through Streamlit.

The system is designed to make generated responses easier to inspect by presenting the document evidence used during retrieval and answer generation.

## Requirements

- Windows 10/11 or a compatible Python environment
- Python 3.11 recommended
- Internet access for initial model downloads and Gemini API requests
- A Google AI Studio API key with access to the configured Gemini model

Native machine-learning dependencies may require additional setup depending on the operating system and Python environment.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/anumay1804/esg-intelligence-rag.git
cd esg-intelligence-rag
```

### 2. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a local environment file from the provided example:

```powershell
Copy-Item .env.example .env
notepad .env
```

Add your own Gemini API key using the variable name expected by the application:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your actual key.

**Security:** Never commit `.env` to Git, publish API keys, or include secrets in project documentation. The repository's `.gitignore` is configured to exclude `.env`.

### 5. Run the application

From the project root:

```powershell
streamlit run app.py
```

Open the local address displayed in the terminal, typically `http://localhost:8501`.

Enter a question about the available ESG reports and inspect the generated answer, citations, and supporting passages.

## Example Query

> What is BASF's greenhouse gas emissions reduction target for 2030?

The application searches the indexed corpus, reranks relevant passages, and generates an answer using Gemini when supporting evidence is available.

## Command-Line Utilities

The project includes command-line tools for querying documents and inspecting retrieval results.

```powershell
python -m scripts.ask_esg
python -m scripts.search_hybrid
python -m scripts.search_reranked
```

## Index and Data Processing

A prebuilt FAISS index and processed data files are included in the repository. These artifacts allow the application to use the existing indexed corpus without immediately rerunning the ingestion pipeline.

If the source documents or processing logic change, regenerate the processed data and indexes using the project scripts:

```powershell
python -m scripts.ingest_documents
python -m scripts.build_chunks
python -m scripts.enrich_metadata
python -m scripts.build_vector_index
python -m scripts.build_knowledge_graph
```

Regenerating these artifacts may overwrite existing processed data and indexes. Back up any files that need to be retained before running the pipeline.

## Evaluation

The repository contains tools for evaluating retrieval behavior, answer generation, and citation diagnostics.

### Retrieval evaluation

```powershell
python -m evaluation.evaluate_retrieval
```

### Hybrid versus graph-assisted retrieval

```powershell
python -m scripts.run_evaluation
```

### Answer and citation diagnostics

```powershell
python -m evaluation.evaluate_answers
```

### Citation-support audit

```powershell
python -m evaluation.validate_citation_support
```

The citation-support audit may consume additional Gemini API quota.

### Interpreting evaluation results

Evaluation metrics should be interpreted according to what they measure:

- Keyword overlap measures lexical similarity, not semantic correctness.
- Citation-ID validity confirms that references exist, not that they support every claim.
- Citation coverage does not independently establish factual accuracy.
- LLM-based judging can
