from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIRECTORY = PROJECT_ROOT / "data" / "policies"
DATABASE_PATH = PROJECT_ROOT / "data" / "shopsense.db"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "checkpoints.db"
LOG_DIRECTORY = PROJECT_ROOT / "logs"
MCP_SERVER_URL = "http://127.0.0.1:8001/mcp"
EMBEDDING_MODEL = "sentence-transformers/all-minilm-l6-v2"
EMBEDDING_DIMENSION = 384
MAX_AUTO_REFUND_MINOR = 500_000
JWT_ISSUER = "shopsense"
JWT_AUDIENCE = "shopsense-local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True,
        populate_by_name=True,
    )

    groq_api_key: SecretStr = Field(validation_alias="GROQ_API_KEY")
    qdrant_url: str = Field(validation_alias="QDRANT_URL")
    qdrant_api_key: SecretStr = Field(validation_alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(validation_alias="QDRANT_COLLECTION")
    jwt_secret: SecretStr = Field(validation_alias="JWT_SECRET", min_length=32)
    mcp_server_url: str = Field(
        default=MCP_SERVER_URL,
        validation_alias="MCP_SERVER_URL",
    )
    supermemory_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="SUPERMEMORY_API_KEY",
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGFUSE_PUBLIC_KEY",
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGFUSE_SECRET_KEY",
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias="LANGFUSE_BASE_URL",
    )
    chat_model: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias="GROQ_MODEL",
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    max_message_chars: int = 4000
    top_k: int = 3
    supermemory_search_limit: int = 5

    @field_validator(
        "supermemory_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def normalize_optional_secrets(
        cls,
        value: SecretStr | str | None,
    ) -> SecretStr | None:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            return value if value.get_secret_value().strip() else None
        return SecretStr(value) if value.strip() else None

    @field_validator("langfuse_base_url", mode="before")
    @classmethod
    def default_blank_langfuse_url(cls, value: str | None) -> str:
        return value.strip() if value and value.strip() else "https://cloud.langfuse.com"

    @field_validator("qdrant_url", "langfuse_base_url")
    @classmethod
    def require_https_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("Cloud service URLs must use HTTPS.")
        return value.rstrip("/")

    @field_validator("langfuse_base_url")
    @classmethod
    def require_langfuse_eu(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if normalized != "https://cloud.langfuse.com":
            raise ValueError("LANGFUSE_BASE_URL must be https://cloud.langfuse.com.")
        return normalized

    @field_validator("qdrant_collection")
    @classmethod
    def require_collection_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("QDRANT_COLLECTION must not be empty.")
        return value.strip()

    @field_validator("mcp_server_url")
    @classmethod
    def enforce_local_mcp_endpoint(cls, value: str) -> str:
        if value != MCP_SERVER_URL:
            raise ValueError(f"MCP_SERVER_URL must be {MCP_SERVER_URL}.")
        return value

    @model_validator(mode="after")
    def validate_optional_cloud_services(self) -> Self:
        if bool(self.langfuse_public_key) != bool(self.langfuse_secret_key):
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured together."
            )
        return self

    @property
    def supermemory_enabled(self) -> bool:
        return bool(
            self.supermemory_api_key
            and self.supermemory_api_key.get_secret_value().strip()
        )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.langfuse_public_key
            and self.langfuse_public_key.get_secret_value().strip()
            and self.langfuse_secret_key
            and self.langfuse_secret_key.get_secret_value().strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
