from typing import TypedDict


class SupportState(TypedDict, total=False):
    customer_id: str
    user_id: str
    conversation_id: str
    thread_id: str
    session_id: str
    turn_id: str
    query_type: str
    conversation: list[dict[str, str]]
    memory_summary: str
    order_id: str
    active_order_id: str
    use_active_order: bool
    return_eligibility_requested: bool
    pending_action: dict | None
    confirmation_accepted: bool
    confirmation_rejected: bool
    is_greeting: bool
    message: str
    intent: str
    active_agent: str
    sentiment: str
    answer: str
    status: str
    approval_id: str | None
    sources: list[str]
    guardrail_facts: dict | None


class SupportContext(TypedDict):
    user_jwt: str
    trace_id: str
