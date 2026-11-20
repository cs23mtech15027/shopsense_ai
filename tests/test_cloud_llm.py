import json

import httpx
import pytest

from app.llm.openai_compatible import OpenAICompatibleClient
from app.logging_config import bind_thread_id, bind_trace_id


@pytest.mark.asyncio
async def test_groq_compatible_client_sends_key_only_in_authorization_header():
    observed = {}

    def handle(request):
        observed["authorization"] = request.headers["Authorization"]
        observed["url"] = str(request.url)
        observed["body"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Cloud response"}}
                ]
            },
        )

    client = OpenAICompatibleClient(
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "test-groq-secret",
        transport=httpx.MockTransport(handle),
    )

    assert await client.generate("system prompt", "user message") == "Cloud response"
    assert observed["authorization"] == "Bearer test-groq-secret"
    assert observed["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert "test-groq-secret" not in observed["body"]


@pytest.mark.asyncio
async def test_groq_json_generation_uses_strict_schema():
    observed = {}
    schema = {
        "type": "object",
        "properties": {"allowed": {"type": "boolean"}},
        "required": ["allowed"],
        "additionalProperties": False,
    }

    def handle(request):
        observed["body"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"allowed":true}'}}]},
        )

    client = OpenAICompatibleClient(
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "test-groq-secret",
        transport=httpx.MockTransport(handle),
    )

    assert await client.generate_json("system JSON", "user", schema=schema) == {
        "allowed": True
    }
    request_body = json.loads(observed["body"])
    assert request_body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "shopsense_response",
            "strict": True,
            "schema": schema,
        },
    }
    assert request_body["max_completion_tokens"] == 1024
    assert request_body["reasoning_effort"] == "low"
    assert "max_tokens" not in request_body


@pytest.mark.asyncio
async def test_groq_rate_limit_retries_after_provider_delay(monkeypatch):
    observed = {"calls": 0, "delays": []}

    def handle(_request):
        observed["calls"] += 1
        if observed["calls"] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "2"},
                json={"error": {"message": "rate limited"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Recovered response"}}]},
        )

    async def record_sleep(delay):
        observed["delays"].append(delay)

    monkeypatch.setattr("app.llm.openai_compatible.asyncio.sleep", record_sleep)
    client = OpenAICompatibleClient(
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "test-groq-secret",
        transport=httpx.MockTransport(handle),
    )

    assert await client.generate("system", "user") == "Recovered response"
    assert observed == {"calls": 2, "delays": [2.0]}


@pytest.mark.asyncio
async def test_groq_rate_limit_uses_token_reset_when_retry_after_missing(monkeypatch):
    observed = {"calls": 0, "delays": []}

    def handle(_request):
        observed["calls"] += 1
        if observed["calls"] == 1:
            return httpx.Response(
                429,
                headers={"x-ratelimit-reset-tokens": "2.5s"},
                json={"error": {"message": "rate limited"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Recovered response"}}]},
        )

    async def record_sleep(delay):
        observed["delays"].append(delay)

    monkeypatch.setattr("app.llm.openai_compatible.asyncio.sleep", record_sleep)
    client = OpenAICompatibleClient(
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "test-groq-secret",
        transport=httpx.MockTransport(handle),
    )

    assert await client.generate("system", "user") == "Recovered response"
    assert observed == {"calls": 2, "delays": [2.5]}


@pytest.mark.asyncio
async def test_groq_http_failure_logs_status_and_request_id_without_secrets(caplog):
    def handle(_request):
        return httpx.Response(
            401,
            headers={"x-request-id": "groq-request-123"},
            json={"error": {"message": "unauthorized test-groq-secret"}},
        )

    client = OpenAICompatibleClient(
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "test-groq-secret",
        transport=httpx.MockTransport(handle),
    )
    with bind_thread_id("thread123456"), bind_trace_id("a" * 32):
        with pytest.raises(RuntimeError, match="Cloud language model is unavailable"):
            await client.generate_json("system", "user", schema={"type": "object"})

    failure = next(
        record for record in caplog.records
        if record.name == "shopsense.llm"
        and record.msg.startswith("Cloud model request failed")
    )
    assert "status_code=401" in failure.getMessage()
    assert "request_id=groq-request-123" in failure.getMessage()
    assert "error_type=HTTPStatusError" in failure.getMessage()
    assert "provider_error=unauthorized [REDACTED_SECRET]" in failure.getMessage()
    assert "test-groq-secret" not in caplog.text


@pytest.mark.asyncio
async def test_groq_http_error_redacts_pii_from_provider_message(caplog):
    def handle(_request):
        return httpx.Response(
            400,
            headers={"x-request-id": "groq-request-400"},
            json={
                "error": {
                    "message": (
                        "Invalid request for customer@example.com, "
                        "card 4111 1111 1111 1111"
                    )
                }
            },
        )

    client = OpenAICompatibleClient(
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "test-groq-secret",
        transport=httpx.MockTransport(handle),
    )
    with pytest.raises(RuntimeError, match="Cloud language model is unavailable"):
        await client.generate_json("system", "user", schema={"type": "object"})

    failure = next(
        record for record in caplog.records
        if record.name == "shopsense.llm"
        and record.msg.startswith("Cloud model request failed")
        and "groq-request-400" in record.getMessage()
    )
    assert "[REDACTED_EMAIL]" in failure.getMessage()
    assert "[REDACTED_PAYMENT_NUMBER]" in failure.getMessage()
    assert "customer@example.com" not in caplog.text
    assert "4111 1111 1111 1111" not in caplog.text
