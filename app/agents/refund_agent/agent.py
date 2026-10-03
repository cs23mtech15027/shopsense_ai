import asyncio

from app.agents.order_agent.tools.order_tool import OrderTool
from app.agents.refund_agent.callback import on_agent_complete
from app.agents.refund_agent.tools.refund_tool import RefundTool


class RefundAgent:
    def __init__(self, order_tool: OrderTool, refund_tool: RefundTool):
        self.order_tool = order_tool
        self.refund_tool = refund_tool

    async def handle(self, state: dict) -> dict:
        order_id = self.order_tool.extract_order_id(state["message"])
        if order_id is None:
            return {
                "answer": "Please share the order ID so I can check the return options.",
                "status": "needs_information",
                "sources": [],
            }
        order = await asyncio.to_thread(
            self.order_tool.get_owned_order,
            order_id,
            state["customer_id"],
        )
        if order is None:
            return {
                "answer": (
                    "I couldn't find that order for your account. Please check the order ID "
                    "or contact support if you believe this is an error."
                ),
                "status": "needs_information",
                "sources": [],
            }
        if not await asyncio.to_thread(self.refund_tool.eligible, order):
            on_agent_complete("ineligible")
            return {
                "answer": (
                    f"Order {order_id} is not eligible for a refund under the current "
                    "return-window and order-status checks. I have not initiated a refund."
                ),
                "status": "needs_information",
                "sources": [],
            }
        approval_id = await asyncio.to_thread(
            self.refund_tool.create_approval,
            order_id,
            state["customer_id"],
        )
        amount = order["amount_minor"] / 100
        answer = (
            f"Order {order_id} is eligible for a refund of {order['currency']} {amount:.2f}. "
            "The return has been received and inspected. No refund has been issued; "
            "an authorized person must approve this request."
        )
        on_agent_complete("pending_approval")
        return {
            "answer": answer,
            "status": "pending_approval",
            "approval_id": approval_id,
            "sources": [],
        }
