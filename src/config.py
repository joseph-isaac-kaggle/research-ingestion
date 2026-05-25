"""Global configuration via environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    claude_model: str = "claude-sonnet-4-6"

    # Storage — local, edge-only
    db_path: str = "./data/papers.db"
    chroma_path: str = "./data/chroma"
    bm25_index_path: str = "./data/bm25_index.pkl"

    # Embedding model (local, no cloud)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # Retrieval
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5
    dense_weight: float = 0.6
    sparse_weight: float = 0.4

    # Ingestion
    arxiv_max_results: int = 50
    default_domains: list[str] = [
        "cs.LG", "cs.AI", "stat.ML", "cs.CV", "cs.NE"
    ]


settings = Settings()
