import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.workflow.router import SupportWorkflow
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class FakeCareAgent:
    async def handle(self, state):
        return {"answer": "Policy details.", "status": "answered", "sources": []}


class FakeOrderAgent:
    async def handle(self, state):
        return {"answer": "Order details.", "status": "answered", "sources": []}


class FakeRefundAgent:
    async def handle(self, state):
        return {"answer": "Approval required.", "status": "pending_approval", "sources": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_intent", "sentiment"),
    [
        ("Where is order ORD-1001?", "order", "neutral"),
        ("I want a refund for ORD-1001, this is frustrating", "refund", "negative"),
        ("What is the refund policy?", "care", "neutral"),
        ("Thanks, that was great", "care", "positive"),
    ],
)
async def test_workflow_routes_to_one_specialist(message, expected_intent, sentiment):
    workflow = SupportWorkflow(FakeCareAgent(), FakeOrderAgent(), FakeRefundAgent())

    result = await workflow.graph.ainvoke(
        {"customer_id": "customer-1001", "message": message}
    )

    assert result["intent"] == expected_intent
    assert result["sentiment"] == sentiment


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query_type", "message", "expected_intent"),
    [
        ("order", "What is the return policy?", "order"),
        ("refund", "Where is my package?", "refund"),
        ("care", "Where is order ORD-1001?", "care"),
    ],
)
async def test_selected_query_type_overrides_automatic_routing(
    query_type,
    message,
    expected_intent,
):
    workflow = SupportWorkflow(FakeCareAgent(), FakeOrderAgent(), FakeRefundAgent())

    result = await workflow.graph.ainvoke(
        {
            "customer_id": "customer-1001",
            "message": message,
            "query_type": query_type,
        }
    )

    assert result["intent"] == expected_intent


@pytest.mark.asyncio
async def test_conversation_checkpoint_restores_prior_turns(tmp_path):
    workflow = SupportWorkflow(FakeCareAgent(), FakeOrderAgent(), FakeRefundAgent())
    checkpoint_path = str(tmp_path / "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
        await checkpointer.setup()
        graph = workflow.compile(checkpointer)
        config = {"configurable": {"thread_id": "customer-1001:conversation-1"}}

        await graph.ainvoke(
            {"customer_id": "customer-1001", "message": "Where is ORD-1001?"},
            config,
        )
        result = await graph.ainvoke(
            {"customer_id": "customer-1001", "message": "Thanks"},
            config,
        )

    assert [entry["role"] for entry in result["conversation"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_api_accepts_explicit_query_type_and_routes_to_order_agent(tmp_path):
    application = create_app(
        Settings(
            database_path=str(tmp_path / "orders.db"),
            checkpoint_path=str(tmp_path / "checkpoints.db"),
        )
    )

    with TestClient(application) as client:
        response = client.post(
            "/v1/chat",
            json={
                "customer_id": "customer-1001",
                "conversation_id": "selected-topic",
                "message": "Please check ORD-1001",
                "query_type": "order",
            },
        )

    assert response.status_code == 200
    assert response.json()["intent"] == "order"
    assert "Cotton T-shirt" in response.json()["answer"]


def test_api_refund_requires_authorized_approval_before_initiating(tmp_path):
    application = create_app(
        Settings(
            database_path=str(tmp_path / "orders.db"),
            checkpoint_path=str(tmp_path / "checkpoints.db"),
            approval_token="test-approval-token",
        )
    )

    with TestClient(application) as client:
        request = client.post(
            "/v1/chat",
            json={
                "customer_id": "customer-1001",
                "conversation_id": "refund-approval",
                "message": "Please return ORD-1001",
                "query_type": "refund",
            },
        )
        assert request.status_code == 200
        request_body = request.json()
        assert request_body["status"] == "pending_approval"

        with application.state.order_repository.database.connect() as connection:
            before = connection.execute(
                "SELECT refund_initiated FROM orders WHERE order_id = 'ORD-1001'"
            ).fetchone()
        assert before["refund_initiated"] == 0

        denied = client.post(
            f"/v1/approvals/{request_body['approval_id']}",
            json={"approved": True},
            headers={"X-Approval-Token": "incorrect"},
        )
        assert denied.status_code == 403

        approved = client.post(
            f"/v1/approvals/{request_body['approval_id']}",
            json={"approved": True},
            headers={"X-Approval-Token": "test-approval-token"},
        )

        with application.state.order_repository.database.connect() as connection:
            after = connection.execute(
                "SELECT refund_initiated, refund_initiated_date "
                "FROM orders WHERE order_id = 'ORD-1001'"
            ).fetchone()

    assert approved.status_code == 200
    assert after["refund_initiated"] == 1
    assert after["refund_initiated_date"] is not None
    assert "No money was transferred" in approved.json()["answer"]
