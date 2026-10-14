from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    catalog_collection: str = "catalog_chunks"
    answer_cache_collection: str = "answer_cache"

    # Cache / session store
    redis_url: str = "redis://localhost:6379/0"

    # Rerank service
    rerank_url: str = "http://localhost:8001/rerank"
    rerank_timeout_seconds: float = 2.0

    # LLM provider — the mock provider is used automatically whenever
    # OPENAI_API_KEY is empty; setting it switches to the real OpenAI
    # adapter with no other flag needed.
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_chat_model: str = "gpt-4o-mini"

    # Embedding provider
    embedding_provider: str = "fastembed_local"
    dense_model_name: str = "BAAI/bge-small-en-v1.5"
    sparse_model_name: str = "Qdrant/bm25"
    rerank_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Retrieval tuning
    hybrid_fetch_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 3
    confidence_floor: float = -8.5

    # Cache tuning
    exact_cache_ttl_seconds: int = 3600
    semantic_cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.97

    # Memory tuning
    session_ttl_seconds: int = 86400
    recent_turns_window: int = 3
    summary_refresh_every_n_turns: int = 4
    active_entity_limit: int = 3
    active_entity_boost: float = 0.05

    # Catalog / admin
    catalog_admin_token: str = "change-me-local-dev-only"

    # Observability
    log_level: str = "INFO"


settings = Settings()
