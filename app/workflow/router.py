import re

from langgraph.graph import END, START, StateGraph

from app.agents.care_agent.agent import CareAgent
from app.agents.order_agent.agent import OrderAgent
from app.agents.refund_agent.agent import RefundAgent
from app.workflow.state import SupportState


_RETURN_TERMS = re.compile(r"\b(return|exchange)\b", re.IGNORECASE)
_REFUND_TERMS = re.compile(r"\b(refund|money back)\b", re.IGNORECASE)
_REFUND_EXECUTION = re.compile(
    r"\b(want|need|request|initiate|start|process|submit|issue|do|approve)\b"
    r".{0,30}\b(refund|money back)\b"
    r"|\b(refund|money back)\b.{0,30}\b(now|please|initiate|start|process|submit|issue)\b",
    re.IGNORECASE,
)
_RETURN_INFORMATION = re.compile(
    r"\b(steps?|process|procedure|instructions?|how to|how do i|how can i|"
    r"tell me how|explain|policy|policies|eligib(?:le|ility))\b",
    re.IGNORECASE,
)
_ORDER_TERMS = re.compile(
    r"\b(order|track|tracking|delivery|delivered|shipped|package|arrive|arrival|eta)\b"
    r"|\bwhen\b.*\b(deliver|arrive)\b",
    re.IGNORECASE,
)
_ORDER_ID = re.compile(r"\bORD[-_ ]?\d{4,}\b", re.IGNORECASE)
_ORDER_CONTEXT_REFERENCE = re.compile(r"\b(it|that|this|same)\b", re.IGNORECASE)
_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|howdy|good morning|good afternoon|good evening)"
    r"(?:\s+there)?[.!?,\s]*$",
    re.IGNORECASE,
)
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
        graph.add_node("greeting", self._greeting)
        graph.add_node("remember", self._remember)
        graph.add_edge(START, "triage")
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

    @staticmethod
    async def _triage(state: SupportState) -> dict:
        message = state["message"]
        greeting = _GREETING.fullmatch(message) is not None
        current_order_match = _ORDER_ID.search(message)
        previous_order_id = next(
            (
                match.group().upper().replace("_", "-").replace(" ", "-")
                for entry in reversed(state.get("conversation", []))
                if (match := _ORDER_ID.search(entry.get("content", ""))) is not None
            ),
            None,
        )
        active_order_id = (
            current_order_match.group().upper().replace("_", "-").replace(" ", "-")
            if current_order_match
            else previous_order_id or state.get("active_order_id", "")
        )
        use_active_order = bool(
            current_order_match
            or (
                active_order_id
                and _ORDER_CONTEXT_REFERENCE.search(message)
            )
        )
        query_type = state.get("query_type", "auto")
        if greeting:
            intent = "care"
        elif query_type in {"care", "order", "refund"}:
            intent = query_type
        elif (
            _RETURN_INFORMATION.search(message)
            and (_RETURN_TERMS.search(message) or _REFUND_TERMS.search(message))
        ):
            intent = "care"
        elif _REFUND_EXECUTION.search(message) or (
            _REFUND_TERMS.search(message) and _ORDER_ID.search(message)
        ):
            intent = "refund"
        elif _ORDER_TERMS.search(message) or _ORDER_ID.search(message):
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
            "active_order_id": active_order_id,
            "use_active_order": use_active_order,
            "is_greeting": greeting,
        }

    @staticmethod
    def _route(state: SupportState) -> str:
        return "greeting" if state.get("is_greeting") else state["intent"]

    @staticmethod
    async def _greeting(_: SupportState) -> dict:
        return {
            "answer": (
                "Hi! I'm ShopSense, your shopping support assistant. I can help with "
                "order tracking, returns, refunds, and store policies. What can I help "
                "you with today?"
            ),
            "status": "answered",
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
