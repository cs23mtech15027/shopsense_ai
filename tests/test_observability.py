import pytest

import app.observability as observability
from app.config import Settings


def _settings(**overrides) -> Settings:
    values = {
        "GROQ_API_KEY": "test-groq-key",
        "QDRANT_URL": "https://qdrant.example",
        "QDRANT_API_KEY": "test-qdrant-key",
        "QDRANT_COLLECTION": "shopsense_test",
        "JWT_SECRET": "j" * 48,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_optional_cloud_integrations_disable_when_keys_are_blank():
    settings = _settings(
        SUPERMEMORY_API_KEY="",
        LANGFUSE_PUBLIC_KEY="",
        LANGFUSE_SECRET_KEY="",
        LANGFUSE_BASE_URL="",
    )

    assert settings.supermemory_api_key is None
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.supermemory_enabled is False
    assert settings.langfuse_enabled is False
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"
    assert observability.get_client(settings) is None


def test_runtime_starts_without_optional_cloud_integrations():
    from app.main import create_runtime

    runtime = create_runtime(
        _settings(
            SUPERMEMORY_API_KEY="",
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
    )

    assert runtime.memory is None
    assert runtime.langfuse is None


def test_langfuse_requires_both_keys_but_defaults_to_eu_endpoint():
    settings = _settings()
    assert settings.langfuse_enabled is False
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"

    with pytest.raises(ValueError, match="configured together"):
        _settings(LANGFUSE_PUBLIC_KEY="public-only")


def test_langfuse_initializes_with_redaction_mask_and_eu_base_url(monkeypatch):
    observed = {}

    class FakeLangfuse:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(observability, "Langfuse", FakeLangfuse)
    monkeypatch.setattr(observability, "_client", None)
    monkeypatch.setattr(observability, "_client_identity", None)

    client = observability.get_client(
        _settings(
            LANGFUSE_PUBLIC_KEY="public-test",
            LANGFUSE_SECRET_KEY="private-test",
        )
    )

    assert isinstance(client, FakeLangfuse)
    assert observed["public_key"] == "public-test"
    assert observed["secret_key"] == "private-test"
    assert observed["base_url"] == "https://cloud.langfuse.com"
    assert observed["mask"] is observability.redact_sensitive_data


def test_sensitive_data_is_recursively_redacted_before_trace_export(monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-actual-secret")
    data = {
        "nested": [
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature",
            "Card 4111 1111 1111 1111 and 1234567890123",
            "QDRANT_API_KEY=qdrant-actual-secret",
        ],
        "numeric_card": 4111111111111111,
    }

    redacted = observability.redact_sensitive_data(data)
    rendered = repr(redacted)

    assert "eyJhbGci" not in rendered
    assert "4111 1111 1111 1111" not in rendered
    assert "1234567890123" not in rendered
    assert "qdrant-actual-secret" not in rendered
    assert redacted["numeric_card"] == "[REDACTED_PAYMENT_NUMBER]"
