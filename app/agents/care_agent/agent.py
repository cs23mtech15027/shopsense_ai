import asyncio
import json
from pathlib import Path
from typing import Any

from app.agents.mcp_tools import call_oms_tool
from app.agents.care_agent.callback import on_agent_complete
from app.agents.care_agent.tools.policy_tool import PolicyTool
from app.llm.responses import (
    compose_customer_answer,
    recent_conversation,
    recent_customer_messages,
)


_AGENT_DIRECTORY = Path(__file__).parent
_HYGIENE_RESTRICTED = ("innerwear", "underwear", "lingerie", "cosmetic", "makeup")


def _return_confirmation_message(order_facts: dict) -> str:
    if any(
        term in order_facts["item_name"].casefold()
        for term in _HYGIENE_RESTRICTED
    ):
        return "This item is non-returnable because it is a hygiene-restricted category."
    if order_facts.get("eligible"):
        return (
            f"{order_facts['order_id']} ({order_facts['item_name']}) is eligible "
            "for a return. Would you like me to record a return request? "
            "This demo does not send an email or return label."
        )
    reasons = "; ".join(order_facts.get("reasons", []))
    return (
        f"{order_facts['order_id']} is not eligible for a return"
        + (f": {reasons}" if reasons else ".")
    )


class CareAgent:
    def __init__(
        self,
        policy_tool: PolicyTool,
        llm,
    ):
        self.policy_tool = policy_tool
        self.llm = llm
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

    async def handle(self, state: dict, user_jwt: str, trace_id: str) -> dict:
        pending_action = state.get("pending_action")
        if (
            state.get("confirmation_accepted")
            and isinstance(pending_action, dict)
            and pending_action.get("type") == "return"
        ):
            order_id = pending_action.get("order_id", "")
            result = await call_oms_tool(
                "request_return",
                {
                    "order_id": order_id,
                    "confirmed_order_id": order_id,
                },
                user_jwt,
                thread_id=state.get("thread_id", ""),
                trace_id=trace_id,
            )
            if not isinstance(result, dict):
                raise RuntimeError("The OMS returned an invalid return-request result.")
            if result.get("status") in {"requested", "already_requested"}:
                answer = (
                    f"Your return request for {order_id} has been recorded. "
                    "No email or return label is sent by this demo."
                )
                status = "return_requested"
            else:
                answer = (
                    "I could not record the return request because the order is "
                    "no longer eligible. No request was submitted."
                )
                status = "needs_information"
            return {
                "answer": answer,
                "status": status,
                "pending_action": None,
                "sources": [],
                "guardrail_facts": None,
            }

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
            return {
                "answer": answer,
                "status": "answered",
                "sources": [],
                "guardrail_facts": None,
            }

        order_facts = None
        order_id = state.get("order_id", "")
        return_eligibility_requested = state.get(
            "return_eligibility_requested",
            False,
        )
        if return_eligibility_requested and order_id:
            order = await call_oms_tool(
                "get_order_details",
                {"order_id": order_id},
                user_jwt,
                thread_id=state.get("thread_id", ""),
                trace_id=trace_id,
            )
            if isinstance(order, dict) and order.get("found"):
                assessment = order["return_eligibility"]
                order_facts = {
                    "order_id": order["order_id"],
                    "item_name": order["item_name"],
                    "amount_minor": order["amount_minor"],
                    "currency": order["currency"],
                    "refund_initiated": order["refund_initiated"],
                    "status": order["status"],
                    "delivered_at": order["delivered_at"],
                    "return_window_days": assessment["return_window_days"],
                    "days_since_delivery": assessment["days_since_delivery"],
                    "item_unused": bool(order["item_unused"]),
                    "tags_attached": bool(order["tags_attached"]),
                    "final_sale": bool(order["final_sale"]),
                    "eligible": assessment["eligible"],
                    "reasons": assessment["reasons"],
                    "open_box_delivery": order["open_box_delivery"],
                    "otp_verified": order["otp_verified"],
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
            if order_facts is not None:
                pending = (
                    {"type": "return", "order_id": order_facts["order_id"]}
                    if order_facts.get("eligible")
                    else None
                )
                return {
                    "answer": _return_confirmation_message(order_facts),
                    "status": "confirmation_required" if pending else "ineligible",
                    "pending_action": pending,
                    "sources": [],
                    "guardrail_facts": order_facts,
                }
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
                "guardrail_facts": order_facts,
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
            pending = (
                {"type": "return", "order_id": order_facts["order_id"]}
                if order_facts.get("eligible")
                else None
            )
            on_agent_complete("answered", len(passages))
            return {
                "answer": _return_confirmation_message(order_facts),
                "status": "confirmation_required" if pending else "ineligible",
                "pending_action": pending,
                "sources": [item["source"] for item in passages],
                "guardrail_facts": order_facts,
            }
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
        return {
            "answer": answer,
            "status": "answered",
            "sources": sources,
            "guardrail_facts": order_facts,
        }
