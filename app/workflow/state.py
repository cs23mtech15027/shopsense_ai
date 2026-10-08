from typing import TypedDict


class SupportState(TypedDict, total=False):
    customer_id: str
    query_type: str
    conversation: list[dict[str, str]]
    message: str
    intent: str
    sentiment: str
    answer: str
    status: str
    approval_id: str | None
    sources: list[str]
