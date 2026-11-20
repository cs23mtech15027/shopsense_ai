import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from app.agents.mcp_tools import call_oms_tool, extract_order_id
from app.config import MCP_SERVER_URL, Settings
from app.agents.order_agent.agent import OrderAgent
from app.mcp_server import (
    _check_refund_eligibility,
    _evaluate_and_initiate_refund,
    _is_loopback_host,
    _is_loopback_origin,
    create_mcp_app,
)
from app.persistence.checkpoints import checkpoint_context
from app.persistence.database import Database
from app.persistence.orders import OrderRepository
from app.persistence.users import UserRepository
from app.workflow.router import SupportWorkflow


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


def test_settings_require_the_fixed_loopback_mcp_endpoint():
    assert _settings().mcp_server_url == MCP_SERVER_URL
    with pytest.raises(ValueError, match="MCP_SERVER_URL must be"):
        _settings(MCP_SERVER_URL="https://remote.example/mcp")


def test_settings_require_a_strong_jwt_secret_and_https_cloud_endpoints():
    with pytest.raises(ValueError):
        _settings(JWT_SECRET="short")
    with pytest.raises(ValueError, match="HTTPS"):
        _settings(QDRANT_URL="http://qdrant.example")


def test_mcp_host_and_origin_checks_are_loopback_only():
    assert _is_loopback_host("127.0.0.1:8001")
    assert _is_loopback_host("localhost:8001")
    assert not _is_loopback_host("localhost:8002")
    assert _is_loopback_origin("")
    assert _is_loopback_origin("http://localhost:8501")
    assert not _is_loopback_origin("https://localhost:8501")
    assert not _is_loopback_origin("http://localhost.attacker.test:8501")
    assert not _is_loopback_origin("http://localhost:9999")


def test_order_id_extraction_normalizes_allowed_input():
    assert extract_order_id("Please check ord 1101") == "ORD-1101"
    assert extract_order_id("Check ORD_1101") == "ORD-1101"
    assert extract_order_id("I need return help") is None


@pytest.mark.asyncio
async def test_mcp_adapter_passes_bearer_jwt_in_header_not_tool_arguments(monkeypatch):
    observed = {}

    class FakeTool:
        name = "get_order_details"

        async def ainvoke(self, arguments):
            observed["arguments"] = arguments
            return SimpleNamespace(
                content='{"found": true, "order_id": "ORD-1101"}'
            )

    class FakeClient:
        def __init__(self, servers):
            observed["servers"] = servers

        async def get_tools(self):
            return [FakeTool()]

    monkeypatch.setattr("app.agents.mcp_tools.MultiServerMCPClient", FakeClient)
    result = await call_oms_tool(
        "get_order_details",
        {"order_id": "ORD-1101"},
        "user-jwt-secret",
        thread_id="thread123456",
        trace_id="a" * 32,
    )
    assert result["found"] is True
    server = observed["servers"]["shopsense"]
    assert server["url"] == MCP_SERVER_URL
    assert server["headers"]["Authorization"] == "Bearer user-jwt-secret"
    assert server["headers"]["X-ShopSense-Thread-Id"] == "thread123456"
    assert server["headers"]["X-ShopSense-Trace-Id"] == "a" * 32
    assert observed["arguments"] == {"order_id": "ORD-1101"}


@pytest.mark.asyncio
async def test_order_agent_requires_explicit_order_id_for_cancellation(monkeypatch):
    observed = {}

    async def fake_call(name, arguments, user_jwt, *, thread_id, trace_id):
        observed.update(
            name=name,
            arguments=arguments,
            user_jwt=user_jwt,
            thread_id=thread_id,
            trace_id=trace_id,
        )
        return {"cancelled": True, "status": "cancelled"}

    class FakeLLM:
        async def generate_json(self, system_prompt, user_prompt, *, schema=None):
            return {"answer": "ORD-1102 was cancelled."}

    monkeypatch.setattr("app.agents.order_agent.agent.call_oms_tool", fake_call)
    result = await OrderAgent(FakeLLM()).handle(
        {"message": "Please cancel order ORD-1102."},
        "shopper-jwt",
        "b" * 32,
    )
    assert result["status"] == "cancelled"
    assert observed == {
        "name": "cancel_order",
        "arguments": {
            "order_id": "ORD-1102",
            "confirmed_order_id": "ORD-1102",
        },
        "user_jwt": "shopper-jwt",
        "thread_id": "",
        "trace_id": "b" * 32,
    }


def test_demo_seed_provisions_five_shoppers_and_order_scenarios(tmp_path):
    database = Database(str(tmp_path / "shopsense.db"))
    database.initialize()
    users = UserRepository(database)
    with database.connect() as connection:
        customers = connection.execute(
            "SELECT customer_id FROM customers ORDER BY customer_id"
        ).fetchall()
        orders = connection.execute(
            "SELECT order_id, customer_id FROM orders WHERE order_id LIKE 'ORD-1%'"
        ).fetchall()
    assert [row["customer_id"] for row in customers] == [
        f"customer-{number}" for number in range(1, 6)
    ]
    assert any(row["order_id"] == "ORD-1111" for row in orders)
    assert any(row["order_id"] == "ORD-1112" for row in orders)
    token = users.issue_token("customer-1", "customer", "j" * 48)
    assert users.get_token_claims(token, "j" * 48)["sub"] == "customer-1"
    users.revoke_token(token)
    assert users.get_token_claims(token, "j" * 48) is None


def test_refunds_enforce_owner_confirmation_category_and_auto_limit(tmp_path):
    database = Database(str(tmp_path / "shopsense.db"))
    database.initialize()
    orders = OrderRepository(database)

    wrong_owner = _evaluate_and_initiate_refund(
        orders, "customer-2", "ORD-1101", "ORD-1101", "refund please"
    )
    assert wrong_owner == {
        "created": False,
        "reason": "Order not found for this account.",
    }

    mismatch = _evaluate_and_initiate_refund(
        orders, "customer-2", "ORD-2101", "ORD-2102", "refund please"
    )
    assert mismatch["created"] is False
    assert "Exact order ID" in mismatch["reason"]

    innerwear = _evaluate_and_initiate_refund(
        orders, "customer-1", "ORD-1111", "ORD-1111", "return this item"
    )
    assert innerwear["created"] is False
    assert "non-returnable" in innerwear["reason"]

    open_box_damage = _evaluate_and_initiate_refund(
        orders,
        "customer-1",
        "ORD-1112",
        "ORD-1112",
        "The item arrived damaged",
    )
    assert open_box_damage["created"] is False
    assert "Open-Box" in open_box_damage["reason"]

    high_value = _evaluate_and_initiate_refund(
        orders, "customer-1", "ORD-1113", "ORD-1113", "please refund this order"
    )
    assert high_value["created"] is True
    assert high_value["status"] == "pending_tier2_approval"
    with database.connect() as connection:
        assert connection.execute(
            "SELECT 1 FROM refunds WHERE order_id = 'ORD-1113'"
        ).fetchone() is None
        order_1101 = connection.execute(
            "SELECT amount_minor FROM orders WHERE order_id = 'ORD-1101'"
        ).fetchone()
    assert order_1101["amount_minor"] == 75000

    low_value = _evaluate_and_initiate_refund(
        orders, "customer-2", "ORD-2101", "ORD-2101", "please refund this order"
    )
    assert low_value["created"] is True
    assert low_value["status"] == "initiated"
    assert low_value["refund_initiated"] is True


def test_refund_eligibility_check_has_no_side_effect_and_reports_minor_units(tmp_path):
    database = Database(str(tmp_path / "shopsense.db"))
    database.initialize()
    orders = OrderRepository(database)

    check = _check_refund_eligibility(
        orders,
        "customer-1",
        "ORD-1101",
        "ORD-1101",
        "Please refund this order.",
    )

    assert check["eligible"] is True
    assert check["amount_minor"] == 75000
    assert check["requires_supervisor_approval"] is False
    with database.connect() as connection:
        assert connection.execute(
            "SELECT 1 FROM refunds WHERE order_id = 'ORD-1101'"
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM approvals WHERE order_id = 'ORD-1101'"
        ).fetchone() is None


def test_return_request_requires_eligibility_and_is_idempotent(tmp_path):
    database = Database(str(tmp_path / "shopsense.db"))
    database.initialize()
    orders = OrderRepository(database)

    request = orders.create_return_request(
        "ORD-1101",
        "customer-1",
        "ORD-1101",
    )
    duplicate = orders.create_return_request(
        "ORD-1101",
        "customer-1",
        "ORD-1101",
    )

    assert request["status"] == "requested"
    assert duplicate == {
        "status": "already_requested",
        "request_id": request["request_id"],
    }

    ineligible = orders.create_return_request(
        "ORD-1111",
        "customer-1",
        "ORD-1111",
    )
    assert ineligible["status"] == "ineligible"


@pytest.mark.asyncio
async def test_mcp_rejects_missing_or_remote_browser_auth(tmp_path):
    database = Database(str(tmp_path / "shopsense.db"))
    database.initialize()
    application = create_mcp_app(
        OrderRepository(database),
        UserRepository(database),
        "j" * 48,
    )
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:8001",
    ) as client:
        missing_token = await client.post("/mcp", json={})
        remote_origin = await client.post(
            "/mcp",
            headers={
                "Authorization": "Bearer invalid",
                "Origin": "http://remote.example",
            },
            json={},
        )
    assert missing_token.status_code == 401
    assert remote_origin.status_code == 403


class _RunState(TypedDict):
    message: str
    answer: str


class _RunContext(TypedDict):
    user_jwt: str


@pytest.mark.asyncio
async def test_async_checkpoint_context_persists_state_but_not_runtime_jwt(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoints.db")
    captured = {}

    async def answer(state: _RunState, runtime: Runtime[_RunContext]):
        captured["jwt"] = runtime.context["user_jwt"]
        return {"answer": "done"}

    graph_builder = StateGraph(_RunState, context_schema=_RunContext)
    graph_builder.add_node("answer", answer)
    graph_builder.add_edge(START, "answer")
    graph_builder.add_edge("answer", END)
    config = {"configurable": {"thread_id": "thread123456"}}

    async with checkpoint_context(checkpoint_path) as checkpointer:
        graph = graph_builder.compile(checkpointer=checkpointer)
        result = await graph.ainvoke(
            {"message": "hello"},
            config,
            context={"user_jwt": "secret-jwt"},
        )
        stored = await graph.aget_state(config)
    assert result["answer"] == "done"
    assert captured["jwt"] == "secret-jwt"
    assert "secret-jwt" not in repr(stored.values)
    with sqlite3.connect(checkpoint_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.asyncio
async def test_workflow_forwards_jwt_as_runtime_context_only(tmp_path):
    captured = {}

    class FakeLLM:
        async def generate_json(self, system_prompt, user_prompt, *, schema):
            return {
                "intent": "order",
                "order_id": "ORD-1101",
                "is_greeting": False,
                "return_eligibility_requested": False,
            }

    class FakeAgent:
        async def handle(self, state, user_jwt, trace_id):
            captured["user_jwt"] = user_jwt
            captured["trace_id"] = trace_id
            return {
                "answer": "Order details.",
                "status": "answered",
                "sources": [],
                "guardrail_facts": None,
            }

    workflow = SupportWorkflow(
        care_agent=FakeAgent(),
        order_agent=FakeAgent(),
        refund_agent=FakeAgent(),
        llm=FakeLLM(),
    )
    config = {"configurable": {"thread_id": "workflow12345"}}
    async with checkpoint_context(str(tmp_path / "workflow.db")) as checkpointer:
        graph = workflow.compile(checkpointer)
        result = await graph.ainvoke(
            {
                "user_id": "customer-1",
                "thread_id": "workflow12345",
                "message": "Where is ORD-1101?",
                "query_type": "auto",
            },
            config,
            context={
                "user_jwt": "do-not-persist-this-jwt",
                "trace_id": "c" * 32,
            },
        )
        stored = await graph.aget_state(config)

    assert result["answer"] == "Order details."
    assert captured["user_jwt"] == "do-not-persist-this-jwt"
    assert captured["trace_id"] == "c" * 32
    assert "do-not-persist-this-jwt" not in repr(stored.values)
