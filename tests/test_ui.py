import json

import pytest

import ui
from ui import _chat_settings_panel


def test_chat_settings_show_thread_id_as_read_only():
    panel = _chat_settings_panel(
        customer_id="customer-2",
        query_type="refund",
        thread_id="AbCdEf123_-0",
    )

    widgets = {widget.id: widget.to_dict() for widget in panel.inputs}

    assert widgets["thread_id"]["type"] == "textinput"
    assert widgets["thread_id"]["initial"] == "AbCdEf123_-0"
    assert widgets["thread_id"]["disabled"] is True
    assert widgets["customer_id"]["initial"] == "customer-2"
    assert widgets["query_type"]["initial"] == "refund"


@pytest.mark.asyncio
async def test_chat_stream_keeps_all_answer_tokens(monkeypatch):
    answer = "Hello there! How can I help?"
    events = [
        ("status", {"phase": "checking"}),
        ("status", {"phase": "answering"}),
        ("token", {"token": "Hello "}),
        ("token", {"token": "there! "}),
        ("token", {"token": "How "}),
        ("token", {"token": "can "}),
        ("token", {"token": "I "}),
        ("token", {"token": "help?"}),
        ("complete", {"answer": answer}),
    ]

    class MockResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for event_name, data in events:
                yield f"event: {event_name}"
                yield f"data: {json.dumps(data)}"
                yield ""

    class MockClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def stream(self, *_args, **_kwargs):
            return MockResponse()

    class MockMessage:
        content = ""

        async def update(self):
            return None

        async def stream_token(self, token):
            self.content += token

    monkeypatch.setattr(ui.httpx, "AsyncClient", MockClient)
    message = MockMessage()

    result = await ui._post_chat_stream(
        "hello",
        "conversation-id",
        "thread-id",
        "session-id",
        "auto",
        "customer-1",
        "access-token",
        message,
    )

    assert result["answer"] == answer
    assert message.content == answer
