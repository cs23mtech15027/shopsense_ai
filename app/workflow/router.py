import logging
import re
import time

from langgraph.graph import END, START, StateGraph

from app.agents.care_agent.agent import CareAgent
from app.agents.intent_agent.agent import IntentDetectionAgent
from app.agents.order_agent.agent import OrderAgent
from app.agents.refund_agent.agent import RefundAgent
from app.llm.ollama import OllamaClient
from app.persistence.memory import AgentMemory
from app.workflow.state import SupportState

_NEGATIVE_SENTIMENT = re.compile(
    r"\b(angry|frustrated|frustrating|upset|disappointed|unacceptable|terrible|awful|late|"
    r"broken|complaint|worst)\b|\bnot\s+(happy|satisfied|pleased)\b",
    re.IGNORECASE,
)
_POSITIVE_SENTIMENT = re.compile(
    r"\b(happy|great|thanks|thank you|excellent|love|wonderful)\b",
    re.IGNORECASE,
)

logger = logging.getLogger("shopsense.workflow")


class SupportWorkflow:
    def __init__(
        self,
        care_agent: CareAgent,
        order_agent: OrderAgent,
        refund_agent: RefundAgent,
        memory: AgentMemory | None = None,
        llm: OllamaClient | None = None,
    ):
        self.care_agent = care_agent
        self.order_agent = order_agent
        self.refund_agent = refund_agent
        self.memory = memory
        self.llm = llm
        self.intent_agent = IntentDetectionAgent(llm) if llm is not None else None
        graph = StateGraph(SupportState)
        graph.add_node("memory_context", self._memory_context)
        graph.add_node("triage", self._triage)
        graph.add_node("care", self._care)
        graph.add_node("order", self._order)
        graph.add_node("refund", self._refund)
        graph.add_node("greeting", self._greeting)
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
        conversation = [*state.get("conversation", []), {"role": "user", "content": message}]
        return {
            "intent": intent,
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
            "is_greeting": greeting,
        }

    @staticmethod
    def _route(state: SupportState) -> str:
        route = "greeting" if state.get("is_greeting") else state["intent"]
        logger.info("Workflow route selected agent=%s", route)
        return route

    async def _greeting(self, state: SupportState) -> dict:
        return await self._invoke_agent("care", self.care_agent, state)

    async def _memory_context(self, state: SupportState) -> dict:
        started = time.perf_counter()
        logger.info("Memory context load started")
        if self.memory is None or not state.get("user_id"):
            logger.info("Memory context load skipped")
            return {}
        memory_context = self.memory.load_context(state["user_id"])
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
        if self.memory is not None and self.llm is not None:
            required_ids = (
                state.get("user_id"),
                state.get("thread_id"),
                state.get("session_id"),
                state.get("turn_id"),
            )
            if not all(required_ids):
                raise RuntimeError("Turn identifiers are required to persist agent memory.")
            await self.memory.remember_turn(
                user_id=state["user_id"],
                thread_id=state["thread_id"],
                session_id=state["session_id"],
                turn_id=state["turn_id"],
                message=state["message"],
                answer=state["answer"],
                order_id=state.get("active_order_id", ""),
                llm=self.llm,
            )
        logger.info(
            "Turn memory persistence completed elapsed_ms=%.0f",
            (time.perf_counter() - started) * 1000,
        )
        return {"conversation": conversation[-12:]}

    def compile(self, checkpointer=None):
        self.graph = self.builder.compile(checkpointer=checkpointer)
        return self.graph

    async def _care(self, state: SupportState) -> dict:
        return await self._invoke_agent("care", self.care_agent, state)

    async def _order(self, state: SupportState) -> dict:
        return await self._invoke_agent("order", self.order_agent, state)

    async def _refund(self, state: SupportState) -> dict:
        return await self._invoke_agent("refund", self.refund_agent, state)

    async def _invoke_agent(
        self,
        agent_id: str,
        agent: CareAgent | OrderAgent | RefundAgent,
        state: SupportState,
    ) -> dict:
        started = time.perf_counter()
        logger.info(
            "Specialist execution started agent=%s",
            agent_id,
        )
        try:
            result = await agent.handle(state)
        except Exception:
            logger.exception("Specialist execution failed agent=%s", agent_id)
            raise
        logger.info(
            "Specialist execution completed agent=%s status=%s elapsed_ms=%.0f",
            agent_id,
            result.get("status"),
            (time.perf_counter() - started) * 1000,
        )
        return result
