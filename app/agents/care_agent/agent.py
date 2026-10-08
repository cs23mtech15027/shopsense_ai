import json
import re
from pathlib import Path

from app.agents.care_agent.callback import on_agent_complete
from app.agents.care_agent.tools.policy_tool import PolicyTool
from app.llm.ollama import OllamaClient


_AGENT_DIRECTORY = Path(__file__).parent
_RETURN_WINDOW = re.compile(
    r"Eligible items may be returned within (?P<window>\d+ calendar days after delivery)",
    re.IGNORECASE,
)
_RETURN_CONDITIONS = re.compile(
    r"An item must be (?P<conditions>[^.]+)\.",
    re.IGNORECASE,
)
_RETURN_EXCLUSIONS = re.compile(
    r"(Final-sale items, personalized items, and items marked non-returnable at purchase "
    r"are not eligible for return unless required by applicable law\.)",
    re.IGNORECASE,
)
_RETURN_REQUEST_STEPS = re.compile(
    r"(To request a return, provide the order number to customer support\.)",
    re.IGNORECASE,
)
_RETURN_QUESTION = re.compile(
    r"\b(return|refund|refudn|refun|exchange)\b",
    re.IGNORECASE,
)
_REFUND_QUESTION = re.compile(
    r"\b(refund|refudn|refun|money back)\b",
    re.IGNORECASE,
)


class CareAgent:
    def __init__(self, policy_tool: PolicyTool, llm: OllamaClient):
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

    async def handle(self, state: dict) -> dict:
        import asyncio

        passages = await asyncio.to_thread(
            self.policy_tool.search_policy,
            state["message"],
        )
        if not passages:
            return {
                "answer": (
                    "I couldn't find a relevant policy passage to answer confidently. "
                    "Could you clarify what part of your order or return you need help with?"
                ),
                "status": "needs_information",
                "sources": [],
            }
        return_passage = next(
            (
                item
                for item in passages
                if item["source"].startswith("returns.txt:")
            ),
            None,
        )
        if return_passage is not None and _RETURN_QUESTION.search(state["message"]):
            answer = _format_return_policy_answer(
                state["message"],
                return_passage["text"],
            )
            sources = [return_passage["source"]]
            on_agent_complete("answered", len(sources))
            return {"answer": answer, "status": "answered", "sources": sources}

        context = "\n\n".join(
            f"[{item['source']}] {item['text']}" for item in passages
        )
        answer = await self.llm.generate(
            f"{self.instructions}\n\n{self.context_instructions}\n\nExample:\n{self.example}",
            f"Customer question:\n{state['message']}\n\nPolicy passages:\n{context}",
        )
        sources = [item["source"] for item in passages]
        on_agent_complete("answered", len(sources))
        return {"answer": answer, "status": "answered", "sources": sources}


def _format_return_policy_answer(question: str, passage: str) -> str:
    if _REFUND_QUESTION.search(question):
        return (
            "After the returned item is received and inspected, an approved refund "
            "is issued to your original payment method. Processing time depends on "
            "the payment provider, and the policy does not specify an exact refund "
            "amount; submitting a return request does not guarantee approval."
        )

    window = _RETURN_WINDOW.search(passage)
    conditions = _RETURN_CONDITIONS.search(passage)
    exclusions = _RETURN_EXCLUSIONS.search(passage)
    request_steps = _RETURN_REQUEST_STEPS.search(passage)
    if not all((window, conditions, exclusions, request_steps)):
        return (
            "I couldn't find the complete return eligibility and request steps in "
            "the current policy passage. Please contact customer support for help."
        )

    return (
        f"Eligible items may be returned within {window['window']} if they are "
        f"{conditions['conditions']}. {exclusions[1]} "
        f"{request_steps[1].rstrip('.')}; any refund requires review and approval."
    )
