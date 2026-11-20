import json
from pathlib import Path

from app.agents.intent_agent.callback import on_agent_complete
from app.agents.intent_agent.tools.intent_tool import IntentTool


_AGENT_DIRECTORY = Path(__file__).parent


class IntentDetectionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.intent_tool = IntentTool()
        configuration = json.loads(
            (_AGENT_DIRECTORY / "agent.json").read_text(encoding="utf-8")
        )
        prompt_files = configuration["prompts"]
        self.instructions = (
            _AGENT_DIRECTORY / prompt_files["instructions"]
        ).read_text(encoding="utf-8")
        self.example = (_AGENT_DIRECTORY / prompt_files["example"]).read_text(
            encoding="utf-8"
        )
        self.context_instructions = (
            _AGENT_DIRECTORY / prompt_files["context"]
        ).read_text(encoding="utf-8")

    async def classify(self, state: dict) -> dict:
        context = self.intent_tool.build_context(state)
        result = await self.llm.generate_json(
            "\n\n".join(
                (self.instructions, self.example, self.context_instructions)
            ),
            self.context_instructions
            + json.dumps(context, ensure_ascii=False),
            schema=self.intent_tool.schema,
        )
        decision = self.intent_tool.validate_decision(result, context)
        on_agent_complete(decision["intent"], bool(decision["order_id"]))
        return decision
