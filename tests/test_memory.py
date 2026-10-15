import asyncio
import json

import pytest

from app.persistence.memory import AgentMemory


@pytest.mark.asyncio
async def test_agent_memory_tags_turns_and_consolidates_without_repeating_calls(tmp_path):
    class FakeLLM:
        prompts = []

        async def generate(self, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            return "Prefers email updates; has an unresolved delivery question."

    memory = AgentMemory(tmp_path / "memories", dreaming_interval=2)
    llm = FakeLLM()

    await memory.remember_turn(
        user_id="customer-1",
        thread_id="thread-1",
        session_id="session-1",
        turn_id="turn-1",
        message="Please send updates by email.",
        answer="I'll note that preference.",
        llm=llm,
    )
    assert memory.load_summary("customer-1") == ""

    await memory.remember_turn(
        user_id="customer-1",
        thread_id="thread-1",
        session_id="session-1",
        turn_id="turn-2",
        message="My delivery is still missing.",
        answer="I can check the order.",
        order_id="ORD-1102",
        llm=llm,
    )
    await memory.wait_for_dreaming("customer-1")
    await memory.remember_turn(
        user_id="customer-1",
        thread_id="thread-1",
        session_id="session-1",
        turn_id="turn-2",
        message="duplicate turn",
        answer="duplicate",
        llm=llm,
    )

    path = tmp_path / "memories" / "customer-1" / "history_summarization.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert len(stored["turns"]) == 2
    assert stored["turns"][0]["user_id"] == "customer-1"
    assert stored["turns"][0]["thread_id"] == "thread-1"
    assert stored["turns"][0]["session_id"] == "session-1"
    assert stored["turns"][0]["turn_id"] == "turn-1"
    assert stored["turns"][1]["order_id"] == "ORD-1102"
    assert memory.load_context("customer-1")["active_order_id"] == "ORD-1102"
    consolidation = json.loads(llm.prompts[0])
    assert consolidation["new_turns"][1]["order_id"] == "ORD-1102"
    assert memory.load_summary("customer-1").startswith("Prefers email")
    assert len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_memory_consolidation_runs_after_turn_write_without_blocking_response(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowLLM:
        async def generate(self, system_prompt, user_prompt):
            started.set()
            await release.wait()
            return "Customer asked about ORD-1102 and received delivery details."

    memory = AgentMemory(tmp_path / "memories", dreaming_interval=1)
    llm = SlowLLM()
    await asyncio.wait_for(
        memory.remember_turn(
            user_id="customer-1",
            thread_id="thread-1",
            session_id="session-1",
            turn_id="turn-1",
            message="Where is ORD-1102?",
            answer="It is on the way.",
            order_id="ORD-1102",
            llm=llm,
        ),
        timeout=0.1,
    )

    await asyncio.wait_for(started.wait(), timeout=1)
    stored = json.loads(
        (
            tmp_path
            / "memories"
            / "customer-1"
            / "history_summarization.json"
        ).read_text(encoding="utf-8")
    )
    assert stored["turns"][0]["message"] == "Where is ORD-1102?"
    assert stored["turns"][0]["answer"] == "It is on the way."
    assert stored["history_summarization"] == ""

    release.set()
    await memory.wait_for_dreaming("customer-1")
    assert memory.load_summary("customer-1").startswith("Customer asked about ORD-1102")
