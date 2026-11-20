import re

from app.agents.mcp_tools import call_oms_tool, extract_order_id
from app.agents.order_agent.callback import on_agent_complete
from app.llm.responses import (
    compose_customer_answer,
    recent_conversation,
    recent_customer_messages,
)

_CANCEL_ACTION = re.compile(
    r"\b(?:please\s+cancel|cancel\s+(?:my\s+)?(?:order\s+)?ORD|"
    r"i\s+want\s+to\s+cancel|i(?:'d| would)\s+like\s+to\s+cancel|"
    r"can\s+you\s+cancel|could\s+you\s+cancel)\b",
    re.IGNORECASE,
)


class OrderAgent:
    def __init__(self, llm):
        self.llm = llm

    async def handle(self, state: dict, user_jwt: str, trace_id: str) -> dict:
        message_order_id = extract_order_id(state["message"])
        if _CANCEL_ACTION.search(state["message"]):
            if message_order_id is None:
                answer = await self._answer(
                    state,
                    {"resolution": "request the order ID before cancelling an order"},
                )
                return {
                    "answer": answer,
                    "status": "needs_information",
                    "sources": [],
                    "guardrail_facts": None,
                }
            cancellation = await call_oms_tool(
                "cancel_order",
                {
                    "order_id": message_order_id,
                    "confirmed_order_id": message_order_id,
                },
                user_jwt,
                thread_id=state.get("thread_id", ""),
                trace_id=trace_id,
            )
            if not isinstance(cancellation, dict):
                raise RuntimeError("The OMS returned an invalid cancellation result.")
            if cancellation.get("cancelled"):
                facts = {
                    "order_id": message_order_id,
                    "status": "cancelled",
                    "resolution": "the order was cancelled",
                }
                status = "cancelled"
            else:
                facts = {
                    "order_id": message_order_id,
                    "status": cancellation.get("status", "unchanged"),
                    "resolution": cancellation.get(
                        "reason",
                        "the order was not found or is not cancellable",
                    ),
                }
                status = "needs_information"
            answer = await self._answer(state, facts)
            on_agent_complete(status)
            return {
                "answer": answer,
                "status": status,
                "sources": [],
                "guardrail_facts": facts,
            }

        order_id = (
            state.get("order_id")
            or extract_order_id(state["message"])
            or (
                state.get("active_order_id")
                if state.get("use_active_order")
                else None
            )
        )
        if order_id is None:
            answer = await self._answer(
                state,
                {"resolution": "request the order ID before looking up order details"},
            )
            return {
                "answer": answer,
                "status": "needs_information",
                "sources": [],
                "guardrail_facts": None,
            }

        order = await call_oms_tool(
            "get_order_details",
            {"order_id": order_id},
            user_jwt,
            thread_id=state.get("thread_id", ""),
            trace_id=trace_id,
        )
        if not isinstance(order, dict) or not order.get("found"):
            on_agent_complete("not_found")
            answer = await self._answer(
                state,
                {
                    "order_id": order_id,
                    "resolution": "no matching order was found for this customer account",
                },
            )
            return {
                "answer": answer,
                "status": "needs_information",
                "sources": [],
                "guardrail_facts": None,
            }

        facts = {
            "order_id": order["order_id"],
            "item_name": order["item_name"],
            "status": order["status"],
            "delivery_date": order.get("delivered_at"),
            "estimated_delivery_date": order.get("estimated_delivery_at"),
            "refund_initiated": bool(order.get("refund_initiated")),
            "refund_initiated_date": order.get("refund_initiated_date"),
        }
        answer = await self._answer(state, facts)
        on_agent_complete("answered")
        return {
            "answer": answer,
            "status": "answered",
            "sources": [],
            "guardrail_facts": facts,
        }

    async def _answer(self, state: dict, facts: dict) -> str:
        return await compose_customer_answer(
            self.llm,
            state["message"],
            facts,
            sentiment=state.get("sentiment", "neutral"),
            memory_summary=state.get("memory_summary", ""),
            recent_customer_messages=recent_customer_messages(state),
            conversation_history=recent_conversation(state),
        )
