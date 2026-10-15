import json

import pytest

from app.agents.intent_agent.agent import IntentDetectionAgent
from app.agents.intent_agent.tools.intent_tool import IntentTool


@pytest.mark.asyncio
async def test_intent_agent_uses_model_decision_and_includes_conversation_context():
    class FakeLLM:
        async def generate_json(self, system_prompt, user_prompt, *, schema):
            assert "not keyword matching" in system_prompt
            assert "Examples of intent decisions" in system_prompt
            assert "candidate_order_ids" in system_prompt
            assert schema["properties"]["return_eligibility_requested"]["type"] == "boolean"
            context = json.loads(
                user_prompt.split("Classify this shopping-support conversation:\n", 1)[1]
            )
            assert context["recent_conversation"] == [
                {"role": "user", "content": "I asked about a delivery."},
                {"role": "assistant", "content": "Please share the order ID."},
            ]
            assert context["candidate_order_ids"] == ["ORD-1102"]
            return {
                "intent": "order",
                "order_id": "ORD-1102",
                "is_greeting": False,
                "return_eligibility_requested": False,
            }

    result = await IntentDetectionAgent(FakeLLM()).classify(
        {
            "message": "When should it arrive?",
            "conversation": [
                {"role": "user", "content": "I asked about a delivery."},
                {"role": "assistant", "content": "Please share the order ID."},
            ],
            "memory_summary": "Active order ID: ORD-1102",
            "active_order_id": "ORD-1102",
            "query_type": "auto",
        }
    )

    assert result == {
        "intent": "order",
        "order_id": "ORD-1102",
        "is_greeting": False,
        "return_eligibility_requested": False,
    }


@pytest.mark.asyncio
async def test_intent_agent_rejects_order_id_not_present_in_context():
    class FakeLLM:
        async def generate_json(self, system_prompt, user_prompt, *, schema):
            return {
                "intent": "order",
                "order_id": "ORD-9999",
                "is_greeting": False,
                "return_eligibility_requested": False,
            }

    with pytest.raises(RuntimeError, match="unknown order ID"):
        await IntentDetectionAgent(FakeLLM()).classify(
            {"message": "Where is my order?", "conversation": []}
        )


@pytest.mark.asyncio
async def test_intent_agent_selects_the_only_active_order_for_return_followup():
    class FakeLLM:
        async def generate_json(self, system_prompt, user_prompt, *, schema):
            return {
                "intent": "care",
                "order_id": "",
                "is_greeting": False,
                "return_eligibility_requested": True,
            }

    result = await IntentDetectionAgent(FakeLLM()).classify(
        {
            "message": "Can this order be returned?",
            "active_order_id": "ORD-1101",
            "conversation": [],
        }
    )

    assert result["intent"] == "care"
    assert result["order_id"] == "ORD-1101"
    assert result["return_eligibility_requested"] is True


def test_intent_tool_builds_context_and_validates_result():
    tool = IntentTool()
    context = tool.build_context(
        {
            "message": "Can this order be returned?",
            "active_order_id": "ORD_1101",
            "conversation": [
                {"role": "user", "content": "I asked about ORD-1101 earlier."},
                {"role": "assistant", "content": "Please share order ID."},
            ],
        }
    )

    assert context["candidate_order_ids"] == ["ORD-1101"]
    assert context["active_order_id"] == "ORD_1101"
    assert tool.validate_decision(
        {
            "intent": "care",
            "order_id": "",
            "is_greeting": False,
            "return_eligibility_requested": True,
        },
        context,
    ) == {
        "intent": "care",
        "order_id": "ORD-1101",
        "is_greeting": False,
        "return_eligibility_requested": True,
    }
