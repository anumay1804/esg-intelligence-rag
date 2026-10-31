from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", "8"))
    top_k_bm25: int = int(os.getenv("TOP_K_BM25", "8"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))
    chunk_size_chars: int = int(os.getenv("CHUNK_SIZE_CHARS", "1800"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "250"))

    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir: str = os.path.join(project_root, "data", "raw")
    processed_dir: str = os.path.join(project_root, "data", "processed")
    indexes_dir: str = os.path.join(project_root, "data", "indexes")
    artifacts_dir: str = os.path.join(project_root, "artifacts")

settings = Settings()
