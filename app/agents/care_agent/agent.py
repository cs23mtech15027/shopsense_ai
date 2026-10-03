import json
from pathlib import Path

from app.agents.care_agent.callback import on_agent_complete
from app.agents.care_agent.tools.policy_tool import PolicyTool
from app.llm.ollama import OllamaClient


_AGENT_DIRECTORY = Path(__file__).parent


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
        context = "\n\n".join(
            f"[{item['source']}] {item['text']}" for item in passages
        )
        history = state.get("conversation", [])[:-1]
        history_context = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history[-4:]
        )
        answer = await self.llm.generate(
            f"{self.instructions}\n\n{self.context_instructions}\n\nExample:\n{self.example}",
            f"Recent conversation (context only):\n{history_context or '(none)'}"
            f"\n\nCustomer question:\n{state['message']}\n\nPolicy passages:\n{context}",
        )
        sources = [item["source"] for item in passages]
        on_agent_complete("answered", len(sources))
        return {"answer": answer, "status": "answered", "sources": sources}
