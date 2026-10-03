import re

from langgraph.graph import END, START, StateGraph

from app.agents.care_agent.agent import CareAgent
from app.agents.order_agent.agent import OrderAgent
from app.agents.refund_agent.agent import RefundAgent
from app.workflow.state import SupportState


_REFUND_TERMS = re.compile(r"\b(refund|return|exchange|money back)\b", re.IGNORECASE)
_REFUND_ACTION = re.compile(
    r"\b(want|need|request|initiate|start|process|submit|issue|cancel)\b",
    re.IGNORECASE,
)
_POLICY_QUESTION = re.compile(
    r"\b(policy|policies|how long|within|conditions|rules|time limit)\b",
    re.IGNORECASE,
)
_ORDER_TERMS = re.compile(
    r"\b(order|track|tracking|delivery|delivered|shipped|package)\b",
    re.IGNORECASE,
)
_ORDER_ID = re.compile(r"\bORD[- ]?\d{4,}\b", re.IGNORECASE)
_NEGATIVE_SENTIMENT = re.compile(
    r"\b(angry|frustrated|frustrating|upset|disappointed|unacceptable|terrible|awful|late|"
    r"broken|complaint|worst)\b|\bnot\s+(happy|satisfied|pleased)\b",
    re.IGNORECASE,
)
_POSITIVE_SENTIMENT = re.compile(
    r"\b(happy|great|thanks|thank you|excellent|love|wonderful)\b",
    re.IGNORECASE,
)


class SupportWorkflow:
    def __init__(self, care_agent: CareAgent, order_agent: OrderAgent, refund_agent: RefundAgent):
        self.care_agent = care_agent
        self.order_agent = order_agent
        self.refund_agent = refund_agent
        graph = StateGraph(SupportState)
        graph.add_node("triage", self._triage)
        graph.add_node("care", self._care)
        graph.add_node("order", self._order)
        graph.add_node("refund", self._refund)
        graph.add_node("remember", self._remember)
        graph.add_edge(START, "triage")
        graph.add_conditional_edges(
            "triage",
            lambda state: state["intent"],
            {"care": "care", "order": "order", "refund": "refund"},
        )
        graph.add_edge("care", "remember")
        graph.add_edge("order", "remember")
        graph.add_edge("refund", "remember")
        graph.add_edge("remember", END)
        self.builder = graph
        self.graph = graph.compile()

    @staticmethod
    async def _triage(state: SupportState) -> dict:
        message = state["message"]
        if _REFUND_TERMS.search(message) and (
            _ORDER_ID.search(message)
            or _REFUND_ACTION.search(message)
            or not _POLICY_QUESTION.search(message)
        ):
            intent = "refund"
        elif _ORDER_TERMS.search(message):
            intent = "order"
        else:
            intent = "care"
        if _NEGATIVE_SENTIMENT.search(message):
            sentiment = "negative"
        elif _POSITIVE_SENTIMENT.search(message):
            sentiment = "positive"
        else:
            sentiment = "neutral"
        conversation = [*state.get("conversation", []), {"role": "user", "content": message}]
        return {
            "intent": intent,
            "sentiment": sentiment,
            "conversation": conversation,
            "answer": "",
            "status": "",
            "approval_id": None,
            "sources": [],
        }

    @staticmethod
    async def _remember(state: SupportState) -> dict:
        conversation = [
            *state.get("conversation", []),
            {"role": "assistant", "content": state["answer"]},
        ]
        return {"conversation": conversation[-12:]}

    def compile(self, checkpointer=None):
        self.graph = self.builder.compile(checkpointer=checkpointer)
        return self.graph

    async def _care(self, state: SupportState) -> dict:
        return await self.care_agent.handle(state)

    async def _order(self, state: SupportState) -> dict:
        return await self.order_agent.handle(state)

    async def _refund(self, state: SupportState) -> dict:
        return await self.refund_agent.handle(state)
