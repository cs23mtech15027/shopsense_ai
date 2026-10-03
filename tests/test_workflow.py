import pytest

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
