import asyncio

from app.agents.order_agent.callback import on_agent_complete
from app.agents.order_agent.tools.order_tool import OrderTool
from app.llm.ollama import OllamaClient
from app.llm.responses import (
    compose_customer_answer,
    recent_conversation,
    recent_customer_messages,
)


class OrderAgent:
    def __init__(self, order_tool: OrderTool, llm: OllamaClient):
        self.order_tool = order_tool
        self.llm = llm

    async def handle(self, state: dict) -> dict:
        order_id = (
            state.get("order_id")
            or self.order_tool.extract_order_id(state["message"])
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
            }
        order = await asyncio.to_thread(
            self.order_tool.get_owned_order,
            order_id,
            state["customer_id"],
        )
        if order is None:
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
            }
        facts = {
            "order_id": order["order_id"],
            "item_name": order["item_name"],
            "status": order["status"],
            "delivery_date": order["delivered_at"],
            "estimated_delivery_date": order["estimated_delivery_at"],
            "refund_initiated": bool(order["refund_initiated"]),
            "refund_initiated_date": order["refund_initiated_date"],
        }
        answer = await self._answer(state, facts)
        on_agent_complete("answered")
        return {"answer": answer, "status": "answered", "sources": []}

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
