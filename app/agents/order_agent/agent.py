import asyncio

from app.agents.order_agent.callback import on_agent_complete
from app.agents.order_agent.tools.order_tool import OrderTool


class OrderAgent:
    def __init__(self, order_tool: OrderTool):
        self.order_tool = order_tool

    async def handle(self, state: dict) -> dict:
        order_id = (
            self.order_tool.extract_order_id(state["message"])
            or (
                state.get("active_order_id")
                if state.get("use_active_order")
                else None
            )
        )
        if order_id is None:
            return {
                "answer": "Please share the order ID (for example, ORD-1101) so I can look it up.",
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
            return {
                "answer": (
                    "I couldn't find that order for your account. Please check the order ID "
                    "or contact support if you believe this is an error."
                ),
                "status": "needs_information",
                "sources": [],
            }
        if order["status"] == "delivered":
            delivery = f"Delivery date: {order['delivered_at']}."
        elif order["estimated_delivery_at"]:
            delivery = f"Estimated delivery: {order['estimated_delivery_at']}."
        else:
            delivery = "An estimated delivery date is not available yet."
        answer = (
            f"Order {order['order_id']} ({order['item_name']}) is currently "
            f"{order['status']}. {delivery}"
        )
        if order["refund_initiated"]:
            answer += (
                f" Refund initiation was recorded on "
                f"{order['refund_initiated_date']}."
            )
        on_agent_complete("answered")
        return {"answer": answer, "status": "answered", "sources": []}
