from app.agents.mcp_tools import call_oms_tool, extract_order_id
from app.agents.refund_agent.callback import on_agent_complete
from app.llm.responses import (
    compose_customer_answer,
    recent_conversation,
    recent_customer_messages,
)


def _format_refund_amount(amount_minor: int, currency: str) -> str:
    if currency == "INR":
        return f"₹{amount_minor / 100:,.2f}"
    return f"{currency} {amount_minor} minor units"


class RefundAgent:
    def __init__(self, llm):
        self.llm = llm

    async def handle(self, state: dict, user_jwt: str, trace_id: str) -> dict:
        pending_action = state.get("pending_action")
        confirming_refund = (
            state.get("confirmation_accepted")
            and isinstance(pending_action, dict)
            and pending_action.get("type") == "refund"
        )
        order_id = (
            pending_action.get("order_id")
            if confirming_refund
            else (
                state.get("order_id")
                or extract_order_id(state["message"])
                or (
                    state.get("active_order_id")
                    if state.get("use_active_order")
                    else None
                )
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
                "guardrail_facts": None,
            }

        if not confirming_refund:
            eligibility = await call_oms_tool(
                "check_refund_eligibility",
                {
                    "order_id": order_id,
                    "confirmed_order_id": order_id,
                    "reason": state["message"],
                },
                user_jwt,
                thread_id=state.get("thread_id", ""),
                trace_id=trace_id,
            )
            if not isinstance(eligibility, dict):
                raise RuntimeError("The OMS returned an invalid refund evaluation.")
            if not eligibility.get("eligible"):
                return {
                    "answer": (
                        f"I cannot submit a refund for {order_id}: "
                        f"{eligibility.get('reason', 'the order is not eligible')}."
                    ),
                    "status": "ineligible",
                    "pending_action": None,
                    "sources": [],
                    "guardrail_facts": {"refund_initiated": False},
                }

            amount = _format_refund_amount(
                eligibility["amount_minor"],
                eligibility["currency"],
            )
            needs_tier2 = bool(eligibility["requires_supervisor_approval"])
            next_step = (
                "submit it for Tier-2 supervisor review"
                if needs_tier2
                else "record the refund request"
            )
            answer = (
                f"{order_id} is eligible for a refund of {amount}. "
                f"If you confirm, I will {next_step}; this demo does not transfer money. "
                "Would you like me to proceed?"
            )
            on_agent_complete("confirmation_required")
            return {
                "answer": answer,
                "status": "confirmation_required",
                "pending_action": {
                    "type": "refund",
                    "order_id": order_id,
                    "reason": state["message"],
                },
                "sources": [],
                "guardrail_facts": None,
            }

        result = await call_oms_tool(
            "evaluate_and_initiate_refund",
            {
                "order_id": order_id,
                "confirmed_order_id": order_id,
                "reason": pending_action.get("reason", ""),
            },
            user_jwt,
            thread_id=state.get("thread_id", ""),
            trace_id=trace_id,
        )
        if not isinstance(result, dict):
            raise RuntimeError("The OMS returned an invalid refund evaluation.")
        if not result.get("created"):
            on_agent_complete("ineligible")
            return {
                "answer": (
                    f"I could not submit the refund request for {order_id}: "
                    f"{result.get('reason', 'the order is not eligible')}."
                ),
                "status": "needs_information",
                "pending_action": None,
                "sources": [],
                "guardrail_facts": {
                    "order_id": order_id,
                    "refund_initiated": False,
                },
            }

        needs_tier2 = bool(result.get("requires_supervisor_approval"))
        initiated = result.get("status") == "initiated"
        amount = _format_refund_amount(
            result["amount_minor"],
            result["currency"],
        )
        facts = {
            "order_id": result["order_id"],
            "refund_eligible": True,
            "refund_initiated": initiated,
            "approval_required": not initiated,
            "requires_staff_review": needs_tier2,
            "requires_supervisor_approval": needs_tier2,
            "amount_minor": result["amount_minor"],
            "currency": result["currency"],
            "resolution": "refund request confirmed by customer",
        }
        status = "pending_approval" if needs_tier2 else (
            "initiated" if initiated else "pending_approval"
        )
        on_agent_complete(status)
        answer = (
            f"The refund request for {order_id} ({amount}) was submitted for "
            "Tier-2 supervisor review. No refund has been initiated yet."
            if needs_tier2
            else (
                f"The refund request for {order_id} ({amount}) has been recorded. "
                "This demo does not transfer money."
            )
        )
        if initiated:
            return {
                "answer": answer,
                "status": "initiated",
                "pending_action": None,
                "sources": [],
                "guardrail_facts": facts,
            }
        return {
            "answer": answer,
            "status": "pending_approval",
            "pending_action": None,
            "requires_staff_review": needs_tier2,
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
