import asyncio
from types import SimpleNamespace

from app.persistence.supermemory import SupermemoryStore
from app.workflow.router import SupportWorkflow


def test_supermemory_reads_profile_and_relevant_user_memories(monkeypatch):
    observed = {}

    class FakeSyncClient:
        def __init__(self, **kwargs):
            observed["sync_init"] = kwargs

        def profile(self, user_id, **kwargs):
            observed["profile_user"] = user_id
            observed["profile_timeout"] = kwargs["timeout"]
            return SimpleNamespace(
                profile=SimpleNamespace(
                    static=[SimpleNamespace(memory="Prefers email updates.")],
                    dynamic=[SimpleNamespace(memory="Recent order was delivered.")],
                )
            )

        def search(self, user_id, **kwargs):
            observed["search_user"] = user_id
            observed["search"] = kwargs
            return SimpleNamespace(
                results=[SimpleNamespace(memory="Asked about delivery tracking.")]
            )

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            observed["async_init"] = kwargs

    monkeypatch.setattr("app.persistence.supermemory.Supermemory", FakeSyncClient)
    monkeypatch.setattr(
        "app.persistence.supermemory.AsyncSupermemory",
        FakeAsyncClient,
    )
    store = SupermemoryStore("sm-test-key")

    result = store.load_context("customer-1", "Email me at jane@example.test")

    assert observed["sync_init"] == {
        "api_key": "sm-test-key",
        "timeout": 5.0,
    }
    assert observed["async_init"] == observed["sync_init"]
    assert observed["profile_user"] == observed["search_user"] == "customer-1"
    assert observed["search"]["search_mode"] == "memories"
    assert observed["search"]["limit"] == 5
    assert observed["search"]["timeout"] == 5.0
    assert "jane@example.test" not in observed["search"]["query"]
    assert result["memory_summary"] == (
        "Prefers email updates.\n"
        "Recent order was delivered.\n"
        "Asked about delivery tracking."
    )


def test_supermemory_adds_sanitized_resolved_context(monkeypatch):
    observed = {}

    class FakeSyncClient:
        def __init__(self, **kwargs):
            observed["sync_init"] = kwargs

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            observed["async_init"] = kwargs

        async def add(self, user_id, **kwargs):
            observed["user_id"] = user_id
            observed["add"] = kwargs

    monkeypatch.setattr("app.persistence.supermemory.Supermemory", FakeSyncClient)
    monkeypatch.setattr(
        "app.persistence.supermemory.AsyncSupermemory",
        FakeAsyncClient,
    )
    store = SupermemoryStore("sm-test-key")
    asyncio.run(
        store.remember_turn(
            user_id="customer-1",
            thread_id="thread123456",
            turn_id="turn-1",
            message=(
                "Email jane@example.test; card 4111 1111 1111 1111; "
                "password is verysecret; Bearer eyJhbGciOiJIUzI1NiJ9."
                "eyJzdWIiOiIxIn0.signature; QDRANT_API_KEY=sm-test-key"
            ),
            answer="Delivery updates will be sent by email.",
            active_agent="care",
        )
    )

    assert observed["user_id"] == "customer-1"
    assert observed["add"]["dreaming"] == "instant"
    assert observed["add"]["metadata"] == {
        "source": "shopsense_support",
        "agent": "care",
    }
    content = observed["add"]["content"]
    assert "jane@example.test" not in content
    assert "4111 1111 1111 1111" not in content
    assert "verysecret" not in content
    assert "eyJhbGci" not in content
    assert "sm-test-key" not in content
    assert "[REDACTED_EMAIL]" in content
    assert "[REDACTED_PAYMENT_NUMBER]" in content


def test_supermemory_failures_and_missing_key_degrade_gracefully(monkeypatch):
    class RaisingSyncClient:
        def __init__(self, **kwargs):
            pass

        def profile(self, *_args, **_kwargs):
            raise TimeoutError

        def search(self, *_args, **_kwargs):
            raise TimeoutError

    class RaisingAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def add(self, *_args, **_kwargs):
            raise TimeoutError

    monkeypatch.setattr("app.persistence.supermemory.Supermemory", RaisingSyncClient)
    monkeypatch.setattr(
        "app.persistence.supermemory.AsyncSupermemory",
        RaisingAsyncClient,
    )
    store = SupermemoryStore("sm-test-key")
    assert store.load_context("customer-1", "delivery") == {"memory_summary": ""}
    asyncio.run(
        store.remember_turn(
            user_id="customer-1",
            thread_id="thread123456",
            turn_id="turn-1",
            message="Where is my order?",
            answer="Please provide the order ID.",
            active_agent="order",
        )
    )

    disabled_store = SupermemoryStore("")
    assert disabled_store.load_context("customer-1", "delivery") == {
        "memory_summary": ""
    }
    asyncio.run(
        disabled_store.remember_turn(
            user_id="customer-1",
            thread_id="thread123456",
            turn_id="turn-2",
            message="Where is my order?",
            answer="Please provide the order ID.",
            active_agent="order",
        )
    )


async def test_workflow_injects_memory_before_agent_and_stores_after_guardrail():
    observed = {}

    class FakeMemory:
        def load_context(self, user_id, query):
            observed["read"] = (user_id, query)
            return {"memory_summary": "Prefers email delivery updates."}

        async def remember_turn(self, **kwargs):
            observed["write"] = kwargs

    class FakeLLM:
        async def generate_json(self, *_args, **_kwargs):
            return {
                "intent": "order",
                "order_id": "",
                "is_greeting": False,
                "return_eligibility_requested": False,
            }

    class FakeAgent:
        async def handle(self, state, _user_jwt, _trace_id):
            observed["memory_at_agent"] = state.get("memory_summary")
            return {
                "answer": "Please provide the order ID to check tracking.",
                "status": "needs_information",
                "sources": [],
                "guardrail_facts": None,
            }

    workflow = SupportWorkflow(
        care_agent=FakeAgent(),
        order_agent=FakeAgent(),
        refund_agent=FakeAgent(),
        memory=FakeMemory(),
        llm=FakeLLM(),
    )
    result = await workflow.graph.ainvoke(
        {
            "user_id": "customer-1",
            "thread_id": "thread123456",
            "turn_id": "turn-1",
            "message": "Where is my order?",
            "query_type": "auto",
        },
        context={"user_jwt": "jwt-not-in-state", "trace_id": "a" * 32},
    )

    assert observed["read"] == ("customer-1", "Where is my order?")
    assert observed["memory_at_agent"] == "Prefers email delivery updates."
    assert observed["write"] == {
        "user_id": "customer-1",
        "thread_id": "thread123456",
        "turn_id": "turn-1",
        "message": "Where is my order?",
        "answer": result["answer"],
        "active_agent": "order",
    }
