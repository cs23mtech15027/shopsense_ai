import asyncio
import secrets

from fastapi import APIRouter, Header, HTTPException, Request

from app.guardrails.input import validate_message
from app.guardrails.output import validate_answer
from app.schemas import (
    ApprovalRequest,
    ApprovalResponse,
    ChatRequest,
    ChatResponse,
)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    settings = request.app.state.settings
    message = validate_message(payload.message, settings.max_message_chars)
    try:
        result = await request.app.state.workflow.graph.ainvoke(
            {
                "customer_id": payload.customer_id,
                "message": message,
                "query_type": payload.query_type,
            },
            {
                "configurable": {
                    "thread_id": f"{payload.customer_id}:{payload.conversation_id}"
                }
            },
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    answer = validate_answer(result.get("answer", ""))
    return ChatResponse(
        intent=result["intent"],
        sentiment=result["sentiment"],
        conversation_id=payload.conversation_id,
        answer=answer,
        status=result["status"],
        approval_id=result.get("approval_id"),
        sources=result.get("sources", []),
    )


@router.post(
    "/v1/approvals/{approval_id}",
    response_model=ApprovalResponse,
)
async def approval(
    approval_id: str,
    payload: ApprovalRequest,
    request: Request,
    x_approval_token: str | None = Header(default=None),
) -> ApprovalResponse:
    expected_token = request.app.state.settings.approval_token
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Refund approval is disabled until an approval token is configured.",
        )
    if x_approval_token is None or not secrets.compare_digest(
        x_approval_token, expected_token
    ):
        raise HTTPException(status_code=403, detail="Approval is not authorized.")

    result = await asyncio.to_thread(
        request.app.state.order_repository.process_approval,
        approval_id,
        payload.approved,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Approval request was not found.")
    status = result["status"]
    if status == "approved":
        amount = result["amount_minor"] / 100
        answer = (
            f"Refund initiation was approved and recorded for {result['currency']} "
            f"{amount:.2f} on {result['refund_initiated_date']}. "
            "No money was transferred: this prototype is not connected to a payment provider."
        )
    elif status == "declined":
        answer = "The refund request was declined. No refund was issued."
    else:
        answer = "This approval request has already been processed."
    return ApprovalResponse(approval_id=approval_id, status=status, answer=answer)
