from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SHOPSENSE_",
        env_file=".env",
        extra="ignore",
    )

    database_path: str = "./data/shopsense.db"
    checkpoint_path: str = "./data/checkpoints.db"
    log_directory: str = "./logs"
    memory_directory: str = "./memories"
    memory_dreaming_interval: int = 3
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "shopsense_policies"
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "qwen3:8b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_message_chars: int = 4000
    top_k: int = 3
    chat_stage_timeout_seconds: float = 60.0
    approval_token: str = ""
    api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
