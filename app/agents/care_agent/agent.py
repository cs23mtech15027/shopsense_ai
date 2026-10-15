import json
from pathlib import Path
from typing import Any

from app.agents.care_agent.callback import on_agent_complete
from app.agents.care_agent.tools.policy_tool import PolicyTool
from app.agents.order_agent.tools.order_tool import OrderTool
from app.llm.ollama import OllamaClient
from app.persistence.orders import OrderRepository
from app.llm.responses import (
    compose_customer_answer,
    recent_conversation,
    recent_customer_messages,
)


_AGENT_DIRECTORY = Path(__file__).parent


class CareAgent:
    def __init__(
        self,
        policy_tool: PolicyTool,
        llm: OllamaClient,
        order_tool: OrderTool | None = None,
    ):
        self.policy_tool = policy_tool
        self.llm = llm
        self.order_tool = order_tool
        configuration = json.loads((_AGENT_DIRECTORY / "agent.json").read_text(encoding="utf-8"))
        self.instructions = (_AGENT_DIRECTORY / configuration["prompts"]["instructions"]).read_text(
            encoding="utf-8"
        )
        self.context_instructions = (
            _AGENT_DIRECTORY / configuration["prompts"]["context"]
        ).read_text(encoding="utf-8")
        self.example = (_AGENT_DIRECTORY / configuration["prompts"]["example"]).read_text(
            encoding="utf-8"
        )

    async def handle(self, state: dict) -> dict:
        if state.get("is_greeting"):
            answer = await compose_customer_answer(
                self.llm,
                state["message"],
                {"interaction": "greeting", "available_help": [
                    "order tracking",
                    "returns",
                    "refunds",
                    "store policies",
                ]},
                sentiment=state.get("sentiment", "neutral"),
                memory_summary=state.get("memory_summary", ""),
                style_instructions=self.instructions,
                recent_customer_messages=recent_customer_messages(state),
                conversation_history=recent_conversation(state),
            )
            on_agent_complete("answered", 0)
            return {"answer": answer, "status": "answered", "sources": []}

        import asyncio

        order_facts = None
        order_id = state.get("order_id", "")
        return_eligibility_requested = state.get(
            "return_eligibility_requested",
            False,
        )
        if return_eligibility_requested and order_id and self.order_tool is not None:
            order = await asyncio.to_thread(
                self.order_tool.get_owned_order,
                order_id,
                state["customer_id"],
            )
            if order is not None:
                assessment = OrderRepository.return_eligibility(order)
                order_facts = {
                    "order_id": order["order_id"],
                    "item_name": order["item_name"],
                    "status": order["status"],
                    "delivered_at": order["delivered_at"],
                    "return_window_days": assessment["return_window_days"],
                    "days_since_delivery": assessment["days_since_delivery"],
                    "item_unused": bool(order["item_unused"]),
                    "tags_attached": bool(order["tags_attached"]),
                    "final_sale": bool(order["final_sale"]),
                    "eligible": assessment["eligible"],
                    "reasons": assessment["reasons"],
                }

        policy_query = state["message"]
        if order_facts is not None:
            policy_query = (
                f"{policy_query} {order_facts['item_name']} "
                "return eligibility category"
            )
        passages = await asyncio.to_thread(
            self.policy_tool.search_policy,
            policy_query,
        )
        if not passages:
            resolution = (
                "request an order ID before checking return options"
                if return_eligibility_requested and not order_id
                else "request clarification"
            )
            answer = await compose_customer_answer(
                self.llm,
                state["message"],
                {
                    "policy_evidence": [],
                    "resolution": resolution,
                },
                sentiment=state.get("sentiment", "neutral"),
                memory_summary=state.get("memory_summary", ""),
                style_instructions=self.instructions,
                recent_customer_messages=recent_customer_messages(state),
                conversation_history=recent_conversation(state),
            )
            return {
                "answer": answer,
                "status": "needs_information",
                "sources": [],
            }

        context = "\n\n".join(
            f"[{item['source']}] {item['text']}" for item in passages
        )
        facts: dict[str, Any] = {"policy_passages": context}
        if return_eligibility_requested and not order_id:
            facts["resolution"] = "request an order ID before checking return options"
        elif return_eligibility_requested and order_id and order_facts is None:
            facts["order_return_assessment"] = {
                "order_id": order_id,
                "verified": False,
            }
        elif order_facts is not None:
            facts["order_return_assessment"] = order_facts
        answer = await compose_customer_answer(
            self.llm,
            state["message"],
            facts,
            sentiment=state.get("sentiment", "neutral"),
            memory_summary=state.get("memory_summary", ""),
            style_instructions=(
                f"{self.instructions}\n\n{self.context_instructions}\n\n"
                f"Example:\n{self.example}"
            ),
            recent_customer_messages=recent_customer_messages(state),
            conversation_history=recent_conversation(state),
        )
        sources = [item["source"] for item in passages]
        on_agent_complete("answered", len(sources))
        return {"answer": answer, "status": "answered", "sources": sources}
