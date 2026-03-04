"""
BotAcademia Engine - Central configuration via Pydantic Settings.
Reads from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ─────────────────────────────────────────
    APP_NAME: str = "BotAcademia Engine"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Google Gemini ────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    GEMINI_MAX_TOKENS: int = 1000          # RAG response: enough for 4-5 bullet items
    GEMINI_PREPROCESS_MAX_TOKENS: int = 100  # Preprocess: only needs ~60-token JSON
    GEMINI_TEMPERATURE: float = 0.3

    # ── PostgreSQL ───────────────────────────────────
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "botacademia"
    POSTGRES_USER: str = "botacademia"
    POSTGRES_PASSWORD: str = "botacademia_secret"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── ChromaDB ─────────────────────────────────────
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIR: str = "/app/data/chroma"

    # ── Redis (production) ────────────────────────────
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_TTL_SECONDS: int = 3600          # 1 hour session cache
    USE_REDIS: bool = False                # Disabled in POC

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    # ── Kafka (production) ────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"
    KAFKA_TOPIC_INCOMING: str = "botacademia.incoming_messages"
    KAFKA_TOPIC_PROCESSED: str = "botacademia.processed_queries"
    KAFKA_TOPIC_RESPONSES: str = "botacademia.responses"
    KAFKA_GROUP_ID_PREPROCESSOR: str = "botacademia-preprocessor"
    KAFKA_GROUP_ID_RAG: str = "botacademia-rag-engine"
    USE_KAFKA: bool = False                # Disabled in POC

    # ── LLM mock (set to True to skip Gemini API calls during dev/demo) ────
    USE_MOCK_LLM: bool = False

    # ── RAG settings ──────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RAG_TOP_K: int = 5                     # # of chunks sent to the LLM
    RAG_FETCH_K: int = 10                  # # of candidates fetched from ChromaDB (pre-rerank)
    USE_RERANKER: bool = True              # cross-encoder reranking of ChromaDB candidates
    EMBEDDING_MODEL: str = "models/embedding-001"   # Gemini embedding

    # ── Data paths ────────────────────────────────────
    MATERIAS_DATA_DIR: str = "/app/data/materias"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
