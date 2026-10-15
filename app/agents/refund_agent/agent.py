import asyncio

from app.agents.order_agent.tools.order_tool import OrderTool
from app.agents.refund_agent.callback import on_agent_complete
from app.agents.refund_agent.tools.refund_tool import RefundTool
from app.llm.ollama import OllamaClient
from app.llm.responses import (
    compose_customer_answer,
    recent_conversation,
    recent_customer_messages,
)


class RefundAgent:
    def __init__(self, order_tool: OrderTool, refund_tool: RefundTool, llm: OllamaClient):
        self.order_tool = order_tool
        self.refund_tool = refund_tool
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
                {"resolution": "request an order ID before checking return options"},
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
        if not await asyncio.to_thread(self.refund_tool.eligible, order):
            on_agent_complete("ineligible")
            answer = await self._answer(
                state,
                {
                    "order_id": order_id,
                    "refund_eligible": False,
                    "refund_initiated": False,
                    "resolution": "the verified return-window and order-status checks failed",
                },
            )
            return {
                "answer": answer,
                "status": "needs_information",
                "sources": [],
            }
        approval_id = await asyncio.to_thread(
            self.refund_tool.create_approval,
            order_id,
            state["customer_id"],
        )
        answer = await self._answer(
            state,
            {
                "order_id": order_id,
                "amount": order["amount_minor"] / 100,
                "currency": order["currency"],
                "refund_eligible": True,
                "refund_initiated": False,
                "approval_required": True,
                "resolution": "ask for explicit approval before initiating a refund",
            },
        )
        on_agent_complete("pending_approval")
        return {
            "answer": answer,
            "status": "pending_approval",
            "approval_id": approval_id,
            "sources": [],
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
