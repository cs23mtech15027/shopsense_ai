from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    customer_id: str | None = Field(default=None, min_length=1, max_length=100)
    conversation_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    query_type: Literal["auto", "care", "order", "refund"] = "auto"


class ChatResponse(BaseModel):
    intent: Literal["care", "order", "refund"]
    sentiment: Literal["positive", "neutral", "negative"]
    conversation_id: str
    answer: str
    status: Literal["answered", "needs_information", "pending_approval"]
    approval_id: str | None = None
    sources: list[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    approved: bool


class ApprovalResponse(BaseModel):
    approval_id: str
    status: Literal["approved", "declined", "already_processed"]
    answer: str


class DemoSessionRequest(BaseModel):
    customer_id: Literal["customer-1", "customer-2", "customer-3"]


class DemoSessionResponse(BaseModel):
    customer_id: str
    access_token: str
    token_type: str = "bearer"
