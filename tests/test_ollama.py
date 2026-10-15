import json

import httpx
import pytest

from app.llm.ollama import OllamaClient


@pytest.mark.asyncio
async def test_stream_forwards_answer_content_only_and_keeps_prompt_private():
    received_requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        received_requests.append(request)
        events = [
            {
                "message": {
                    "content": "Your order is on its way.",
                    "thinking": "Do not show this private reasoning.",
                },
                "done": False,
            },
            {
                "message": {"content": "", "thinking": ""},
                "done": True,
            },
        ]
        return httpx.Response(
            200,
            content="".join(json.dumps(event) + "\n" for event in events),
        )

    client = OllamaClient(
        "http://ollama.test",
        "qwen3:8b",
        transport=httpx.MockTransport(respond),
    )

    tokens = [
        token
        async for token in client.stream(
            "Private system prompt",
            "Private user prompt",
        )
    ]

    assert tokens == ["Your order is on its way."]
    request_body = json.loads(received_requests[0].content)
    assert request_body["think"] is False
    assert request_body["stream"] is True
    assert request_body["keep_alive"] == "10m"
    assert request_body["options"]["num_predict"] == 64
    assert request_body["messages"][1]["content"].endswith("/no_think")


@pytest.mark.asyncio
async def test_json_generation_allows_enough_tokens_for_complete_answer_object():
    received_requests = []
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    def respond(request: httpx.Request) -> httpx.Response:
        received_requests.append(request)
        return httpx.Response(
            200,
            json={"message": {"content": '{"answer":"Hello! How can I help?"}'}},
        )

    client = OllamaClient(
        "http://ollama.test",
        "qwen3:8b",
        transport=httpx.MockTransport(respond),
    )

    result = await client.generate_json(
        "Return answer JSON.",
        "Greet the user.",
        schema=schema,
    )

    assert result == {"answer": "Hello! How can I help?"}
    request_body = json.loads(received_requests[0].content)
    assert request_body["format"] == schema
    assert request_body["options"]["num_predict"] == 128
    assert request_body["messages"][1]["content"].endswith("/no_think")


@pytest.mark.asyncio
async def test_generation_logs_model_and_ollama_timing_breakdown(caplog):
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "Hello!"},
                "load_duration": 1_000_000,
                "prompt_eval_duration": 2_000_000,
                "eval_duration": 3_000_000,
                "prompt_eval_count": 10,
                "eval_count": 2,
            },
        )

    client = OllamaClient(
        "http://ollama.test",
        "qwen3:8b",
        transport=httpx.MockTransport(respond),
    )

    with caplog.at_level("INFO", logger="shopsense.ollama"):
        answer = await client.generate("system", "user")

    assert answer == "Hello!"
    assert "mode=answer" in caplog.text
    assert "prompt_eval_duration" in caplog.text
    assert "eval_duration" in caplog.text
