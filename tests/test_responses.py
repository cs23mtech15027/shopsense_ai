import pytest

from app.llm.responses import compose_customer_answer


class ScriptedLLM:
    def __init__(self, answers, json_answers=()):
        self.answers = iter(answers)
        self.json_answers = iter(json_answers)
        self.prompts = []
        self.json_prompts = []
        self.schemas = []

    async def generate(self, system_prompt, user_prompt):
        self.prompts.append((system_prompt, user_prompt))
        return next(self.answers)

    async def generate_json(self, system_prompt, user_prompt, *, schema=None):
        self.json_prompts.append((system_prompt, user_prompt))
        self.schemas.append(schema)
        result = next(self.json_answers)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_customer_answer_retries_internal_reasoning_without_returning_it():
    llm = ScriptedLLM(
        ["Hi! How can I help you today?"],
        json_answers=[{"answer": "The customer said hello, so I should greet them."}],
    )

    answer = await compose_customer_answer(
        llm,
        "hello",
        {"interaction": "greeting"},
    )

    assert answer == "Hi! How can I help you today?"
    assert len(llm.prompts) == 1
    assert "offer help" in llm.json_prompts[0][0]
    assert "We are given" not in answer


@pytest.mark.asyncio
async def test_customer_answer_fails_closed_if_retry_still_contains_reasoning():
    llm = ScriptedLLM(
        ["We need to answer hello."],
        json_answers=[{"answer": "We need to answer hello."}],
    )

    with pytest.raises(RuntimeError, match="customer-facing response"):
        await compose_customer_answer(
            llm,
            "hello",
            {"interaction": "greeting"},
        )


@pytest.mark.asyncio
async def test_customer_answer_rejects_internal_reasoning_after_greeting_text():
    llm = ScriptedLLM(
        ["Hello! What can I help you with?"],
        json_answers=[
            {"answer": "Hi! We are given a context and need to describe our reasoning."}
        ],
    )

    answer = await compose_customer_answer(
        llm,
        "hello",
        {"interaction": "greeting"},
    )

    assert answer == "Hello! What can I help you with?"


@pytest.mark.asyncio
async def test_customer_answer_uses_constrained_json_retry_after_reasoning(caplog):
    llm = ScriptedLLM(
        [],
        json_answers=[{"answer": "Hi there! How can I help you today?"}],
    )

    with caplog.at_level("INFO", logger="shopsense.responses"):
        answer = await compose_customer_answer(
            llm,
            "hello",
            {"interaction": "greeting"},
        )

    assert answer == "Hi there! How can I help you today?"
    assert len(llm.json_prompts) == 1
    assert llm.schemas[0]["required"] == ["answer"]
    assert len(llm.prompts) == 0
    assert "structured generation" not in caplog.text


@pytest.mark.asyncio
async def test_customer_answer_recovers_when_json_generation_is_invalid():
    llm = ScriptedLLM(
        ["Hello! How can I help you today?"],
        json_answers=[RuntimeError("Local language model returned invalid JSON.")],
    )

    answer = await compose_customer_answer(
        llm,
        "hello",
        {"interaction": "greeting"},
    )

    assert answer == "Hello! How can I help you today?"
    assert len(llm.json_prompts) == 1
    assert len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_customer_answer_recovers_when_json_has_no_answer_field():
    llm = ScriptedLLM(
        ["Hello! How can I help you?"],
        json_answers=[{"response": "not the expected schema"}],
    )

    answer = await compose_customer_answer(
        llm,
        "hello",
        {"interaction": "greeting"},
    )

    assert answer == "Hello! How can I help you?"
    assert len(llm.json_prompts) == 1
    assert len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_greeting_response_omits_model_generated_emoji():
    llm = ScriptedLLM(
        [],
        json_answers=[{"answer": "Hello! How can I help you today? 😊"}],
    )

    answer = await compose_customer_answer(
        llm,
        "hello",
        {"interaction": "greeting", "available_help": ["order tracking"]},
    )

    assert answer == "Hello! How can I help you today?"
    assert "no emojis" in llm.json_prompts[0][0]


@pytest.mark.asyncio
async def test_missing_order_id_retries_irrelevant_answer_and_asks_for_id():
    llm = ScriptedLLM(
        ["me?"],
        json_answers=[{"answer": "me?"}],
    )

    answer = await compose_customer_answer(
        llm,
        "i want my order information",
        {"resolution": "request the order ID before looking up order details"},
    )

    assert "order ID" in answer
    assert "ORD-1101" in answer
    assert len(llm.prompts) == 1
    assert "Do not ask 'me?'" in llm.prompts[0][0]


@pytest.mark.asyncio
async def test_missing_order_id_accepts_relevant_llm_clarification():
    llm = ScriptedLLM(
        [],
        json_answers=[
            {"answer": "Could you share your order number so I can look it up?"}
        ],
    )

    answer = await compose_customer_answer(
        llm,
        "i want my order information",
        {"resolution": "request the order ID before looking up order details"},
    )

    assert answer == "Could you share your order number so I can look it up?"
    assert len(llm.prompts) == 0
