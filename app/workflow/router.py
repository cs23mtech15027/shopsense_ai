import asyncio
import logging
import re
import time

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.agents.care_agent.agent import CareAgent
from app.agents.intent_agent.agent import IntentDetectionAgent
from app.agents.order_agent.agent import OrderAgent
from app.agents.refund_agent.agent import RefundAgent
from app.guardrails.input import redact_pii
from app.guardrails.output import validate_answer
from app.persistence.memory import MemoryStore
from app.workflow.state import SupportContext, SupportState

_NEGATIVE_SENTIMENT = re.compile(
    r"\b(angry|frustrated|frustrating|upset|disappointed|unacceptable|terrible|awful|late|"
    r"broken|complaint|worst)\b|\bnot\s+(happy|satisfied|pleased)\b",
    re.IGNORECASE,
)
_POSITIVE_SENTIMENT = re.compile(
    r"\b(happy|great|thanks|thank you|excellent|love|wonderful)\b",
    re.IGNORECASE,
)
_CONFIRMATION_YES = re.compile(
    r"^\s*(?:yes|y|yeah|yep|sure|please do|go ahead|proceed|confirm|"
    r"i confirm|that is correct|that's correct)[.! ]*\s*$",
    re.IGNORECASE,
)
_CONFIRMATION_NO = re.compile(
    r"^\s*(?:no|n|nope|do not|don't|cancel|not now|stop)[.! ]*\s*$",
    re.IGNORECASE,
)

logger = logging.getLogger("shopsense.workflow")


class SupportWorkflow:
    def __init__(
        self,
        care_agent: CareAgent,
        order_agent: OrderAgent,
        refund_agent: RefundAgent,
        memory: MemoryStore | None = None,
        llm=None,
    ):
        self.care_agent = care_agent
        self.order_agent = order_agent
        self.refund_agent = refund_agent
        self.memory = memory
        self.llm = llm
        self.intent_agent = IntentDetectionAgent(llm) if llm is not None else None
        graph = StateGraph(SupportState, context_schema=SupportContext)
        graph.add_node("memory_context", self._memory_context)
        graph.add_node("triage", self._triage)
        graph.add_node("care", self._care)
        graph.add_node("order", self._order)
        graph.add_node("refund", self._refund)
        graph.add_node("greeting", self._greeting)
        graph.add_node("confirmation", self._confirmation)
        graph.add_node("remember", self._remember)
        graph.add_edge(START, "memory_context")
        graph.add_edge("memory_context", "triage")
        graph.add_conditional_edges(
            "triage",
            self._route,
            {
                "care": "care",
                "order": "order",
                "refund": "refund",
                "greeting": "greeting",
                "confirmation": "confirmation",
            },
        )
        graph.add_edge("care", "remember")
        graph.add_edge("order", "remember")
        graph.add_edge("refund", "remember")
        graph.add_edge("greeting", "remember")
        graph.add_edge("remember", END)
        self.builder = graph
        self.graph = graph.compile()

    async def _triage(self, state: SupportState) -> dict:
        message = state["message"]
        if self.intent_agent is None:
            raise RuntimeError("The intent detection agent is not configured.")
        pending_action = state.get("pending_action")
        confirmation_accepted = bool(
            pending_action and _CONFIRMATION_YES.fullmatch(message)
        )
        confirmation_rejected = bool(
            pending_action and _CONFIRMATION_NO.fullmatch(message)
        )
        if confirmation_accepted:
            action_type = pending_action.get("type")
            classification = {
                "intent": "care" if action_type == "return" else "refund",
                "is_greeting": False,
                "order_id": pending_action.get("order_id", ""),
                "return_eligibility_requested": action_type == "return",
            }
        elif confirmation_rejected:
            classification = {
                "intent": "care",
                "is_greeting": False,
                "order_id": pending_action.get("order_id", ""),
                "return_eligibility_requested": False,
            }
        else:
            classification = await self.intent_agent.classify(state)
        intent = classification["intent"]
        greeting = classification["is_greeting"]
        order_id = classification["order_id"]
        return_eligibility_requested = classification["return_eligibility_requested"]
        active_order_id = order_id or state.get("active_order_id", "")
        use_active_order = bool(active_order_id and intent in {"order", "refund"})
        if _NEGATIVE_SENTIMENT.search(message):
            sentiment = "negative"
        elif _POSITIVE_SENTIMENT.search(message):
            sentiment = "positive"
        else:
            sentiment = "neutral"
        logger.info(
            "Workflow triage completed intent=%s sentiment=%s greeting=%s",
            intent,
            sentiment,
            greeting,
        )
        logger.debug(
            "Workflow triage input=%r result=%r",
            redact_pii(message)[:2000],
            {"intent": intent, "sentiment": sentiment, "order_id": order_id},
        )
        conversation = [*state.get("conversation", []), {"role": "user", "content": message}]
        return {
            "intent": intent,
            "active_agent": "care" if greeting else intent,
            "sentiment": sentiment,
            "conversation": conversation,
            "answer": "",
            "status": "",
            "approval_id": None,
            "sources": [],
            "order_id": order_id,
            "active_order_id": active_order_id,
            "use_active_order": use_active_order,
            "return_eligibility_requested": return_eligibility_requested,
            "pending_action": pending_action if confirmation_accepted else None,
            "confirmation_accepted": confirmation_accepted,
            "confirmation_rejected": confirmation_rejected,
            "is_greeting": greeting,
        }

    @staticmethod
    def _route(state: SupportState) -> str:
        if state.get("confirmation_rejected"):
            return "confirmation"
        route = "greeting" if state.get("is_greeting") else state["intent"]
        logger.info("Workflow route selected agent=%s", route)
        return route

    async def _confirmation(
        self, state: SupportState, runtime: Runtime[SupportContext]
    ) -> dict:
        return {
            "answer": "Okay, I have not submitted the request.",
            "status": "cancelled",
            "pending_action": None,
            "sources": [],
            "guardrail_facts": None,
        }

    async def _greeting(
        self, state: SupportState, runtime: Runtime[SupportContext]
    ) -> dict:
        return await self._invoke_agent(
            "care",
            self.care_agent,
            state,
            runtime.context["user_jwt"],
            runtime.context["trace_id"],
        )

    async def _memory_context(self, state: SupportState) -> dict:
        started = time.perf_counter()
        logger.info("Memory context load started")
        if self.memory is None or not state.get("user_id"):
            logger.info("Memory context load skipped")
            return {}
        try:
            memory_context = await asyncio.to_thread(
                self.memory.load_context,
                state["user_id"],
                state["message"],
            )
        except Exception as error:
            logger.warning(
                "Memory context load failed user_id=%s error_type=%s",
                state["user_id"],
                type(error).__name__,
            )
            return {}
        # Memory is shared across conversations; only checkpoint state can safely
        # supply an active order for the current conversation.
        memory_context.pop("active_order_id", None)
        logger.info(
            "Memory context load completed elapsed_ms=%.0f",
            (time.perf_counter() - started) * 1000,
        )
        return memory_context

    async def _remember(self, state: SupportState) -> dict:
        started = time.perf_counter()
        logger.info("Turn memory persistence started")
        conversation = [
            *state.get("conversation", []),
            {"role": "assistant", "content": state["answer"]},
        ]
        if self.memory is not None and state.get("user_id"):
            try:
                await self.memory.remember_turn(
                    user_id=state["user_id"],
                    thread_id=state.get("thread_id", ""),
                    turn_id=state.get("turn_id", ""),
                    message=state["message"],
                    answer=state["answer"],
                    active_agent=state.get("active_agent", "care"),
                )
            except Exception as error:
                logger.warning(
                    "Memory persistence failed user_id=%s error_type=%s",
                    state["user_id"],
                    type(error).__name__,
                )
        logger.info(
            "Turn memory persistence completed elapsed_ms=%.0f",
            (time.perf_counter() - started) * 1000,
        )
        return {"conversation": conversation[-12:]}

    def compile(self, checkpointer=None):
        self.graph = self.builder.compile(checkpointer=checkpointer)
        return self.graph

    async def _care(
        self, state: SupportState, runtime: Runtime[SupportContext]
    ) -> dict:
        return await self._invoke_agent(
            "care",
            self.care_agent,
            state,
            runtime.context["user_jwt"],
            runtime.context["trace_id"],
        )

    async def _order(
        self, state: SupportState, runtime: Runtime[SupportContext]
    ) -> dict:
        return await self._invoke_agent(
            "order",
            self.order_agent,
            state,
            runtime.context["user_jwt"],
            runtime.context["trace_id"],
        )

    async def _refund(
        self, state: SupportState, runtime: Runtime[SupportContext]
    ) -> dict:
        return await self._invoke_agent(
            "refund",
            self.refund_agent,
            state,
            runtime.context["user_jwt"],
            runtime.context["trace_id"],
        )

    async def _invoke_agent(
        self,
        agent_id: str,
        agent: CareAgent | OrderAgent | RefundAgent,
        state: SupportState,
        user_jwt: str,
        trace_id: str,
    ) -> dict:
        started = time.perf_counter()
        logger.info(
            "Specialist execution started agent=%s",
            agent_id,
        )
        logger.debug(
            "Specialist input agent=%s message=%r",
            agent_id,
            redact_pii(state["message"])[:2000],
        )
        try:
            result = await agent.handle(state, user_jwt, trace_id)
            guardrail_facts = result.pop("guardrail_facts", None)
            result["answer"] = validate_answer(
                result.get("answer", ""),
                guardrail_facts,
            )
            result["guardrail_facts"] = None
        except Exception:
            logger.exception("Specialist execution failed agent=%s", agent_id)
            raise
        logger.info(
            "Specialist execution completed agent=%s status=%s elapsed_ms=%.0f",
            agent_id,
            result.get("status"),
            (time.perf_counter() - started) * 1000,
        )
        logger.debug(
            "Specialist output agent=%s status=%s answer=%r",
            agent_id,
            result.get("status"),
            redact_pii(result.get("answer", ""))[:2000],
        )
        return result
