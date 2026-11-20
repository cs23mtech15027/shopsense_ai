import pytest

from app.agents.care_agent.agent import CareAgent
from app.agents.refund_agent.agent import RefundAgent
from app.workflow.router import SupportWorkflow


class UnusedLLM:
    async def generate_json(self, *_args, **_kwargs):
        raise AssertionError("Confirmation handling should not call the LLM.")


@pytest.mark.asyncio
async def test_pending_refund_confirmation_routes_without_llm():
    workflow = SupportWorkflow(
        care_agent=None,
        order_agent=None,
        refund_agent=None,
        llm=UnusedLLM(),
    )

    triage = await workflow._triage(
        {
            "message": "yes",
            "pending_action": {
                "type": "refund",
                "order_id": "ORD-1101",
                "reason": "Please refund this order.",
            },
            "conversation": [],
        }
    )

    assert triage["intent"] == "refund"
    assert triage["order_id"] == "ORD-1101"
    assert triage["confirmation_accepted"] is True
    assert workflow._route(triage) == "refund"


@pytest.mark.asyncio
async def test_declining_pending_confirmation_clears_action():
    workflow = SupportWorkflow(
        care_agent=None,
        order_agent=None,
        refund_agent=None,
        llm=UnusedLLM(),
    )

    triage = await workflow._triage(
        {
            "message": "no",
            "pending_action": {"type": "return", "order_id": "ORD-1101"},
            "conversation": [],
        }
    )

    assert triage["confirmation_rejected"] is True
    assert triage["pending_action"] is None
    assert workflow._route(triage) == "confirmation"


@pytest.mark.asyncio
async def test_refund_is_only_submitted_after_confirmation(monkeypatch):
    calls = []

    async def fake_call_oms_tool(name, arguments, *_args, **_kwargs):
        calls.append((name, arguments))
        if name == "check_refund_eligibility":
            return {
                "eligible": True,
                "order_id": "ORD-1101",
                "amount_minor": 75000,
                "currency": "INR",
                "requires_supervisor_approval": False,
            }
        return {
            "created": True,
            "status": "initiated",
            "order_id": "ORD-1101",
            "amount_minor": 75000,
            "currency": "INR",
            "refund_initiated": True,
        }

    monkeypatch.setattr(
        "app.agents.refund_agent.agent.call_oms_tool",
        fake_call_oms_tool,
    )
    agent = RefundAgent(UnusedLLM())
    base_state = {
        "message": "Please refund ORD-1101.",
        "order_id": "ORD-1101",
        "conversation": [],
    }

    proposal = await agent.handle(base_state, "jwt", "trace")

    assert calls[0][0] == "check_refund_eligibility"
    assert proposal["status"] == "confirmation_required"
    assert "₹750.00" in proposal["answer"]
    assert proposal["pending_action"]["order_id"] == "ORD-1101"
    assert len(calls) == 1

    confirmed = await agent.handle(
        {
            **base_state,
            "message": "yes",
            "confirmation_accepted": True,
            "pending_action": proposal["pending_action"],
        },
        "jwt",
        "trace",
    )

    assert calls[1][0] == "evaluate_and_initiate_refund"
    assert confirmed["status"] == "initiated"
    assert confirmed["pending_action"] is None


@pytest.mark.asyncio
async def test_return_requires_confirmation_and_records_no_email_promise(monkeypatch):
    calls = []

    async def fake_call_oms_tool(name, arguments, *_args, **_kwargs):
        calls.append((name, arguments))
        if name == "get_order_details":
            return {
                "found": True,
                "order_id": "ORD-1101",
                "item_name": "Cotton shirt",
                "amount_minor": 75000,
                "currency": "INR",
                "refund_initiated": False,
                "status": "delivered",
                "delivered_at": "2026-11-15",
                "return_eligibility": {
                    "eligible": True,
                    "reasons": [],
                    "return_window_days": 30,
                    "days_since_delivery": 5,
                },
                "item_unused": True,
                "tags_attached": True,
                "final_sale": False,
                "open_box_delivery": False,
                "otp_verified": False,
            }
        return {"status": "requested", "request_id": "request-1"}

    class PolicyTool:
        def search_policy(self, _query):
            return [{"source": "refund_processing.txt", "text": "Return policy facts."}]

    monkeypatch.setattr(
        "app.agents.care_agent.agent.call_oms_tool",
        fake_call_oms_tool,
    )
    agent = CareAgent(PolicyTool(), UnusedLLM())
    base_state = {
        "message": "I want to return it.",
        "order_id": "ORD-1101",
        "return_eligibility_requested": True,
        "conversation": [],
    }

    proposal = await agent.handle(base_state, "jwt", "trace")

    assert calls[0][0] == "get_order_details"
    assert proposal["status"] == "confirmation_required"
    assert "Would you like me to record a return request?" in proposal["answer"]
    assert "does not send an email or return label" in proposal["answer"]
    assert proposal["pending_action"] == {
        "type": "return",
        "order_id": "ORD-1101",
    }
    assert len(calls) == 1

    confirmed = await agent.handle(
        {
            **base_state,
            "message": "yes",
            "confirmation_accepted": True,
            "pending_action": proposal["pending_action"],
        },
        "jwt",
        "trace",
    )

    assert calls[1] == (
        "request_return",
        {"order_id": "ORD-1101", "confirmed_order_id": "ORD-1101"},
    )
    assert confirmed["status"] == "return_requested"
    assert "No email or return label" in confirmed["answer"]
