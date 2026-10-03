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
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "shopsense_policies"
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "qwen3:8b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_message_chars: int = 4000
    top_k: int = 3
    approval_token: str = ""
    api_url: str = "http://localhost:8000"
    demo_customer_id: str = "customer-1001"


@lru_cache
def get_settings() -> Settings:
    return Settings()
