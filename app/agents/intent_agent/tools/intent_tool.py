import re
from typing import Any

from app.llm.responses import recent_conversation


_INTENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["care", "order", "refund"]},
        "order_id": {"type": "string"},
        "is_greeting": {"type": "boolean"},
        "return_eligibility_requested": {"type": "boolean"},
    },
    "required": [
        "intent",
        "order_id",
        "is_greeting",
        "return_eligibility_requested",
    ],
    "additionalProperties": False,
}
_ORDER_ID = re.compile(r"\bORD[-_ ]?\d{4,}\b", re.IGNORECASE)
_VALID_INTENTS = {"care", "order", "refund"}


def _normalize_order_id(order_id: str) -> str:
    return order_id.upper().replace("_", "-").replace(" ", "-")


class IntentTool:
    @property
    def schema(self) -> dict[str, object]:
        return _INTENT_SCHEMA

    def build_context(self, state: dict[str, Any]) -> dict[str, Any]:
        message = state["message"]
        conversation = recent_conversation(state)
        prior_customer_messages = [
            entry["content"]
            for entry in conversation
            if entry.get("role") == "user"
        ]
        candidate_ids = {
            _normalize_order_id(match.group())
            for text in [
                message,
                *prior_customer_messages,
                state.get("active_order_id", ""),
            ]
            for match in _ORDER_ID.finditer(text)
        }
        return {
            "current_message": message,
            "recent_conversation": conversation,
            "memory_summary": state.get("memory_summary", ""),
            "active_order_id": state.get("active_order_id", ""),
            "candidate_order_ids": sorted(candidate_ids),
            "user_selected_query_type": state.get("query_type", "auto"),
        }

    def validate_decision(
        self,
        result: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        intent = result.get("intent")
        order_id = result.get("order_id")
        is_greeting = result.get("is_greeting")
        return_eligibility_requested = result.get("return_eligibility_requested")
        if (
            not isinstance(intent, str)
            or intent not in _VALID_INTENTS
            or not isinstance(order_id, str)
            or not isinstance(is_greeting, bool)
            or not isinstance(return_eligibility_requested, bool)
        ):
            raise RuntimeError("Local language model returned an invalid intent decision.")

        current_order_match = _ORDER_ID.search(context["current_message"])
        if current_order_match:
            order_id = _normalize_order_id(current_order_match.group())
        elif order_id:
            match = _ORDER_ID.fullmatch(order_id.strip())
            if match is None:
                raise RuntimeError("Local language model returned an invalid order ID.")
            order_id = _normalize_order_id(match.group())
            if order_id not in context["candidate_order_ids"]:
                raise RuntimeError("Local language model selected an unknown order ID.")
        elif (
            intent in {"order", "refund"} or return_eligibility_requested
        ) and len(context["candidate_order_ids"]) == 1:
            order_id = context["candidate_order_ids"][0]

        return {
            "intent": intent,
            "order_id": order_id,
            "is_greeting": is_greeting,
            "return_eligibility_requested": return_eligibility_requested,
        }
