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
_RETURN_POLICY_QUESTION = re.compile(
    r"\breturn\s+(?:policy|eligibility)\b|\beligible\s+for\s+return\b|"
    r"\breturnable\b",
    re.IGNORECASE,
)
_PRODUCT_RETURN_QUESTION = re.compile(
    r"\b(?:this|that|the|my)\s+(?:product|item|order)\b|"
    r"\b(?:can|could|would)\s+i\s+return\s+(?:this|that|the|my)?\s*"
    r"(?:product|item|order|it)\b|"
    r"\b(?:is|are)\s+(?:this|that|the|my)?\s*"
    r"(?:product|item|order|it)\s+(?:eligible|returnable)\b",
    re.IGNORECASE,
)
_RETURN_ACTION_REQUEST = re.compile(
    r"\b(?:want|would like|need|like)\s+to\s+(?:return|send back)\b|"
    r"\b(?:return|send back)\s+(?:it|this|that|the item|the product)\b|"
    r"\bstart\s+(?:a\s+)?return\b",
    re.IGNORECASE,
)
_EXPLICIT_REFUND_REQUEST = re.compile(
    r"\b(?:refund|money back)\b.{0,50}\b(?:initiate|start|process|request|issue)\b|"
    r"\b(?:initiate|start|process|request|issue)\b.{0,50}\b(?:refund|money back)\b",
    re.IGNORECASE,
)
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
            raise RuntimeError("Language model returned an invalid intent decision.")

        current_order_match = _ORDER_ID.search(context["current_message"])
        if current_order_match:
            order_id = _normalize_order_id(current_order_match.group())
        elif order_id:
            match = _ORDER_ID.fullmatch(order_id.strip())
            if match is None:
                raise RuntimeError("Language model returned an invalid order ID.")
            order_id = _normalize_order_id(match.group())
            if order_id not in context["candidate_order_ids"]:
                raise RuntimeError("Language model selected an unknown order ID.")
        elif (
            intent in {"order", "refund"} or return_eligibility_requested
        ) and len(context["candidate_order_ids"]) == 1:
            order_id = context["candidate_order_ids"][0]

        current_message = context["current_message"]
        return_policy_question = bool(
            _RETURN_POLICY_QUESTION.search(current_message)
        )
        order_specific_return_question = bool(
            _PRODUCT_RETURN_QUESTION.search(current_message)
            or _RETURN_ACTION_REQUEST.search(current_message)
        )
        if (
            (return_policy_question or _RETURN_ACTION_REQUEST.search(current_message))
            and not _EXPLICIT_REFUND_REQUEST.search(current_message)
        ):
            intent = "care"
            if order_specific_return_question and len(context["candidate_order_ids"]) == 1:
                return_eligibility_requested = True
                order_id = context["candidate_order_ids"][0]

        return {
            "intent": intent,
            "order_id": order_id,
            "is_greeting": is_greeting,
            "return_eligibility_requested": return_eligibility_requested,
        }
