import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.workflow.router import SupportWorkflow
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def _select_customer(client, customer_id="customer-1"):
    response = client.post(
        "/v1/auth/demo-session",
        json={"customer_id": customer_id},
    )
    assert response.status_code == 200
    return response.json()


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
        ("ORD-1002", "order", "neutral"),
        ("I want a refund for ORD-1001, this is frustrating", "refund", "negative"),
        (
            "I want to return that product. Can you tell me steps to return my product?",
            "care",
            "neutral",
        ),
        ("How do I get a refund?", "care", "neutral"),
        ("Please do refund for ORD-1101", "refund", "neutral"),
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
async def test_greeting_gets_local_greeting_without_specialist():
    class FailIfCalled:
        async def handle(self, state):
            raise AssertionError("Greeting should not call a specialist.")

    workflow = SupportWorkflow(FailIfCalled(), FailIfCalled(), FailIfCalled())
    result = await workflow.graph.ainvoke(
        {"customer_id": "customer-1", "message": "Hello there!"}
    )

    assert result["status"] == "answered"
    assert "shopping support assistant" in result["answer"]
    assert result["sources"] == []


def test_api_authenticates_three_customers_and_seeds_ten_orders_each(tmp_path):
    application = create_app(
        Settings(
            database_path=str(tmp_path / "orders.db"),
            checkpoint_path=str(tmp_path / "checkpoints.db"),
        )
    )

    with TestClient(application) as client:
        unauthenticated = client.post(
            "/v1/chat",
            json={
                "customer_id": "customer-1",
                "conversation_id": "unauthenticated",
                "message": "Where is ORD-1101?",
            },
        )
        assert unauthenticated.status_code == 401
        for number, first_order, other_customer_order in (
            (1, 1101, 2101),
            (2, 2101, 3101),
            (3, 3101, 1101),
        ):
            login = _select_customer(client, f"customer-{number}")
            headers = {"Authorization": f"Bearer {login['access_token']}"}
            own_order = client.post(
                "/v1/chat",
                headers=headers,
                json={
                    "customer_id": f"customer-{number}",
                    "conversation_id": f"customer-{number}-orders",
                    "message": f"Where is ORD-{first_order}?",
                },
            )
            other_order = client.post(
                "/v1/chat",
                headers=headers,
                json={
                    "customer_id": f"customer-{number}",
                    "conversation_id": f"customer-{number}-other-order",
                    "message": f"Where is ORD-{other_customer_order}?",
                },
            )
            assert own_order.status_code == 200
            assert own_order.json()["status"] == "answered"
            assert other_order.status_code == 200
            assert "couldn't find that order for your account" in other_order.json()["answer"]
            spoofed_customer = client.post(
                "/v1/chat",
                headers=headers,
                json={
                    "customer_id": f"customer-{number % 3 + 1}",
                    "conversation_id": f"spoofed-customer-{number}",
                    "message": f"Where is ORD-{other_customer_order}?",
                },
            )
            assert spoofed_customer.status_code == 403

        with application.state.order_repository.database.connect() as connection:
            counts = connection.execute(
                "SELECT customer_id, COUNT(*) AS order_count FROM orders "
                "WHERE customer_id IN ('customer-1', 'customer-2', 'customer-3') "
                "GROUP BY customer_id"
            ).fetchall()
            assert {row["customer_id"]: row["order_count"] for row in counts} == {
                "customer-1": 10,
                "customer-2": 10,
                "customer-3": 10,
            }


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
        login = _select_customer(client)
        response = client.post(
            "/v1/chat",
            headers={"Authorization": f"Bearer {login['access_token']}"},
            json={
                "customer_id": "customer-1",
                "conversation_id": "selected-topic",
                "message": "Please check ORD-1101",
                "query_type": "order",
            },
        )

    assert response.status_code == 200
    assert response.json()["intent"] == "order"
    assert "Cotton shirt (Customer 1)" in response.json()["answer"]


def test_api_routes_standalone_order_id_to_order_agent(tmp_path):
    application = create_app(
        Settings(
            database_path=str(tmp_path / "orders.db"),
            checkpoint_path=str(tmp_path / "checkpoints.db"),
        )
    )

    with TestClient(application) as client:
        login = _select_customer(client)
        headers = {"Authorization": f"Bearer {login['access_token']}"}
        first = client.post(
            "/v1/chat",
            headers=headers,
            json={
                "customer_id": "customer-1",
                "conversation_id": "order-id-follow-up",
                "message": "Where is my order?",
            },
        )
        second = client.post(
            "/v1/chat",
            headers=headers,
            json={
                "customer_id": "customer-1",
                "conversation_id": "order-id-follow-up",
                "message": "ORD_1103",
            },
        )
        third = client.post(
            "/v1/chat",
            headers=headers,
            json={
                "customer_id": "customer-1",
                "conversation_id": "order-id-follow-up",
                "message": "When will it be delivered?",
            },
        )
        fourth = client.post(
            "/v1/chat",
            headers=headers,
            json={
                "customer_id": "customer-1",
                "conversation_id": "order-id-follow-up",
                "message": "I want help with my order delivery",
            },
        )

    assert first.status_code == 200
    assert first.json()["status"] == "needs_information"
    assert second.status_code == 200
    assert second.json()["intent"] == "order"
    assert "ORD-1103" in second.json()["answer"]
    assert third.status_code == 200
    assert third.json()["intent"] == "order"
    assert "Estimated delivery:" in third.json()["answer"]
    assert fourth.status_code == 200
    assert fourth.json()["intent"] == "order"
    assert fourth.json()["status"] == "needs_information"
    assert "Please share the order ID" in fourth.json()["answer"]
    assert "couldn't find that order" not in fourth.json()["answer"]


def test_api_refund_requires_authorized_approval_before_initiating(tmp_path):
    application = create_app(
        Settings(
            database_path=str(tmp_path / "orders.db"),
            checkpoint_path=str(tmp_path / "checkpoints.db"),
            approval_token="test-approval-token",
        )
    )

    with TestClient(application) as client:
        login = _select_customer(client)
        request = client.post(
            "/v1/chat",
            headers={"Authorization": f"Bearer {login['access_token']}"},
            json={
                "customer_id": "customer-1",
                "conversation_id": "refund-approval",
                "message": "Please initiate refund for ORD-1101",
                "query_type": "refund",
            },
        )
        assert request.status_code == 200
        request_body = request.json()
        assert request_body["status"] == "pending_approval"
        assert "Are you sure you want to proceed" in request_body["answer"]
        assert "ORD-1101" in request_body["answer"]
        assert "INR 2509.00" in request_body["answer"]
        assert "Select **Yes**" in request_body["answer"]

        with application.state.order_repository.database.connect() as connection:
            before = connection.execute(
                "SELECT refund_initiated FROM orders WHERE order_id = 'ORD-1101'"
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
                "FROM orders WHERE order_id = 'ORD-1101'"
            ).fetchone()

    assert approved.status_code == 200
    assert after["refund_initiated"] == 1
    assert after["refund_initiated_date"] is not None
    assert "No money was transferred" in approved.json()["answer"]
