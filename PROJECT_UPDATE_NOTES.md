# Project update notes — 2026-10-30

## Final package updates
- Removed the truncated `evaluation/evaluate_answers.py` and replaced it with a complete answer-generation/citation diagnostics runner.
- Implemented the knowledge-graph build command and hybrid-vs-graph comparison command (both were previously placeholders).
- Added a graph-assisted retrieval toggle in the Streamlit sidebar.
- Added actionable Gemini API failure messages and stronger instructions to abstain when retrieved evidence does not directly answer the question.
- Fixed gold-answer validation so the 20-question retrieval benchmark and 12-answer gold reference set can be maintained separately; optional explicit links are still validated.
- Added a Windows launcher (`START_DEMO.bat`) and updated setup instructions for Python 3.11.
- Removed the local `.env` from the distributable package. `.env.example` contains only a placeholder.
- Removed Python and pytest cache directories from the ZIP.

## Validation performed in the packaging environment
- Python syntax compilation: 48 Python files passed.
- Dependency-light unit tests: 22 passed.
- Gemini generator edge-case tests: 7 passed using a local SDK stub because the packaging environment does not include `google-genai`.
- Gold-answer data validation: 20 retrieval questions, 12 gold answers, and all referenced PDF files/pages validated.
- Data consistency: 1,500 processed chunks, 1,500 indexed chunk records, and 1,500 graph-manifest chunks; indexed IDs match processed IDs.

## What still needs to be checked on the interview laptop
The packaging environment does not have `google-genai`, `faiss-cpu`, `sentence-transformers`, `rank-bm25`, or Streamlit installed, so it could not launch the full UI or execute a live Gemini request. On Windows, install `requirements.txt`, add a valid API key to `.env`, run `python -m pytest -q`, then launch `streamlit run app.py`.

## Interpretation and limitations
- Hit@K, keyword coverage, citation validity, and lexical overlap are heuristics, not proof of correctness or semantic support.
- The graph is a metadata-driven candidate-expansion aid, not a verified ontology.
- The corpus contains two BASF reports. Verify important values, units, scope boundaries, baselines, and citations against the original PDF during the interview.
