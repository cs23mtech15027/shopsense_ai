import asyncio
import json
import logging
import re
import secrets
import sqlite3
import time
from collections.abc import AsyncIterator
from typing import TypeVar
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.guardrails.input import assess_message, sanitize_message, validate_message
from app.guardrails.output import validate_answer
from app.llm.responses import compose_customer_answer
from app.logging_config import bind_thread_id
from app.schemas import (
    ApprovalRequest,
    ApprovalResponse,
    ChatRequest,
    ChatResponse,
    DemoSessionRequest,
    DemoSessionResponse,
)

router = APIRouter()
logger = logging.getLogger("shopsense.api")
_ANSWER_CHUNKS = re.compile(r"\S+\s*")
_CHAT_PROGRESS_INTERVAL_SECONDS = 5.0
_TaskResult = TypeVar("_TaskResult")


async def _progress_events(
    task: asyncio.Task[_TaskResult],
    phase: str,
    timeout_seconds: float,
) -> AsyncIterator[str]:
    started = time.perf_counter()
    while not task.done():
        elapsed = time.perf_counter() - started
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            task.cancel()
            raise TimeoutError
        completed, _ = await asyncio.wait(
            {task},
            timeout=min(_CHAT_PROGRESS_INTERVAL_SECONDS, remaining),
        )
        if not completed:
            elapsed_seconds = int(time.perf_counter() - started)
            yield (
                "event: status\ndata: "
                f"{json.dumps({'phase': phase, 'elapsed_seconds': elapsed_seconds})}\n\n"
            )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/auth/demo-session", response_model=DemoSessionResponse)
async def create_demo_session(
    payload: DemoSessionRequest,
    request: Request,
) -> DemoSessionResponse:
    access_token = await asyncio.to_thread(
        request.app.state.user_repository.create_demo_session,
        payload.customer_id,
    )
    if access_token is None:
        raise HTTPException(status_code=404, detail="Demo customer was not found.")
    return DemoSessionResponse(
        customer_id=payload.customer_id,
        access_token=access_token,
    )


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> ChatResponse:
    return await _run_chat(payload, request, authorization)


@router.post("/v1/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> StreamingResponse:
    async def events():
        timeout_seconds = request.app.state.settings.chat_stage_timeout_seconds
        yield f"event: status\ndata: {json.dumps({'phase': 'checking'})}\n\n"
        prepare_task = asyncio.create_task(
            _prepare_chat(payload, request, authorization)
        )
        try:
            async for event in _progress_events(
                prepare_task,
                "checking",
                timeout_seconds,
            ):
                yield event
            turn = await prepare_task
        except HTTPException as exc:
            yield f"event: error\ndata: {json.dumps({'detail': exc.detail})}\n\n"
            return
        except TimeoutError:
            yield (
                "event: error\ndata: "
                f"{json.dumps({'detail': 'ShopSense timed out while checking your request. Please try again.'})}\n\n"
            )
            return
        finally:
            if not prepare_task.done():
                prepare_task.cancel()
            await asyncio.gather(prepare_task, return_exceptions=True)

        yield f"event: status\ndata: {json.dumps({'phase': 'answering'})}\n\n"
        task = asyncio.create_task(_execute_chat(turn, request))
        try:
            async for event in _progress_events(
                task,
                "answering",
                timeout_seconds,
            ):
                yield event
            result = await task
            for token in _ANSWER_CHUNKS.findall(result.answer):
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
            yield f"event: complete\ndata: {result.model_dump_json()}\n\n"
        except HTTPException as exc:
            yield f"event: error\ndata: {json.dumps({'detail': exc.detail})}\n\n"
        except TimeoutError:
            with bind_thread_id(turn["thread_id"]):
                logger.warning("Chat stage timed out phase=answering")
            yield (
                "event: error\ndata: "
                f"{json.dumps({'detail': 'ShopSense timed out while preparing your answer. Please try again.'})}\n\n"
            )
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(events(), media_type="text/event-stream")


async def _run_chat(
    payload: ChatRequest,
    request: Request,
    authorization: str | None,
) -> ChatResponse:
    turn = await _prepare_chat(payload, request, authorization)
    return await _execute_chat(turn, request)


async def _prepare_chat(
    payload: ChatRequest,
    request: Request,
    authorization: str | None,
) -> dict[str, str]:
    settings = request.app.state.settings
    message = sanitize_message(payload.message, settings.max_message_chars)
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    customer_id = await asyncio.to_thread(
        request.app.state.user_repository.get_session_customer,
        authorization.removeprefix("Bearer ").strip(),
    )
    if customer_id is None:
        raise HTTPException(status_code=401, detail="Your session has expired. Please log in again.")
    if payload.customer_id is not None and payload.customer_id != customer_id:
        raise HTTPException(status_code=403, detail="This order account is not authorized.")
    thread_id = payload.thread_id or secrets.token_urlsafe(9)
    session_id = payload.session_id or str(uuid5(NAMESPACE_URL, thread_id))
    turn_id = str(uuid4())
    with bind_thread_id(thread_id):
        logger.info("Chat authentication completed session_id=%s", session_id)
        logger.info("Input validation started")
        try:
            message = validate_message(message, settings.max_message_chars)
        except HTTPException as exc:
            logger.warning(
                "Manual input validation rejected status_code=%d",
                exc.status_code,
            )
            raise
        logger.info("Manual input validation completed")
        guardrail_started = time.perf_counter()
        try:
            logger.info("LLM input guardrail started")
            allowed = await assess_message(message, request.app.state.ollama)
        except asyncio.CancelledError:
            logger.warning("LLM input guardrail cancelled")
            raise
        except RuntimeError as exc:
            logger.warning(
                "LLM input guardrail failed elapsed_ms=%.0f error=%s",
                (time.perf_counter() - guardrail_started) * 1000,
                str(exc),
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        logger.info(
            "LLM input guardrail completed allowed=%s elapsed_ms=%.0f",
            allowed,
            (time.perf_counter() - guardrail_started) * 1000,
        )
        if not allowed:
            logger.warning("Input rejected by LLM guardrail")
            raise HTTPException(
                status_code=400,
                detail="The request could not be processed. Please rephrase your question.",
            )
        logger.info("Chat request prepared turn_id=%s", turn_id)
    return {
        "customer_id": customer_id,
        "message": message,
        "query_type": payload.query_type,
        "conversation_id": payload.conversation_id,
        "thread_id": thread_id,
        "session_id": session_id,
        "turn_id": turn_id,
    }


async def _execute_chat(
    turn: dict[str, str],
    request: Request,
) -> ChatResponse:
    with bind_thread_id(turn["thread_id"]):
        logger.info(
            "Chat workflow started session_id=%s turn_id=%s",
            turn["session_id"],
            turn["turn_id"],
        )
        try:
            response = await _execute_chat_with_thread_context(turn, request)
        except asyncio.CancelledError:
            logger.warning("Chat workflow cancelled")
            raise
        except Exception:
            logger.exception("Chat workflow failed")
            raise
        logger.info("Chat response ready status=%s", response.status)
        return response


async def _execute_chat_with_thread_context(
    turn: dict[str, str],
    request: Request,
) -> ChatResponse:
    customer_id = turn["customer_id"]
    thread_id = turn["thread_id"]
    workflow_started = time.perf_counter()
    try:
        result = await request.app.state.workflow.graph.ainvoke(
            {
                "customer_id": customer_id,
                "user_id": customer_id,
                "conversation_id": turn["conversation_id"],
                "thread_id": thread_id,
                "session_id": turn["session_id"],
                "turn_id": turn["turn_id"],
                "message": turn["message"],
                "query_type": turn["query_type"],
            },
            {
                "configurable": {
                    "thread_id": thread_id,
                },
                "metadata": {
                    "user_id": customer_id,
                    "conversation_id": turn["conversation_id"],
                    "session_id": turn["session_id"],
                    "turn_id": turn["turn_id"],
                },
            },
        )
    except RuntimeError as exc:
        logger.warning(
            "Chat workflow failed elapsed_ms=%.0f error=%s",
            (time.perf_counter() - workflow_started) * 1000,
            str(exc),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except sqlite3.DatabaseError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Conversation checkpoint storage is unreadable. Preserve the "
                "checkpoint database and check or restore it before retrying."
            ),
        ) from exc
    logger.info(
        "Chat workflow completed intent=%s elapsed_ms=%.0f",
        result.get("intent"),
        (time.perf_counter() - workflow_started) * 1000,
    )
    answer = validate_answer(result.get("answer", ""))
    logger.info(
        "Customer answer validated intent=%s status=%s",
        result.get("intent"),
        result.get("status"),
    )
    return ChatResponse(
        intent=result["intent"],
        sentiment=result["sentiment"],
        user_id=customer_id,
        conversation_id=turn["conversation_id"],
        thread_id=thread_id,
        session_id=turn["session_id"],
        turn_id=turn["turn_id"],
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
        facts = {
            "approval_status": "approved",
            "amount": result["amount_minor"] / 100,
            "currency": result["currency"],
            "refund_initiated_date": result["refund_initiated_date"],
            "payment_transferred": False,
        }
        prompt = "The customer approved a pending refund."
    elif status == "declined":
        facts = {"approval_status": "declined", "refund_issued": False}
        prompt = "The customer declined a pending refund."
    else:
        facts = {
            "approval_status": "already_processed",
            "previous_decision": result["approval"]["status"],
        }
        prompt = "The customer submitted an approval for a request already processed."
    try:
        answer = await compose_customer_answer(
            request.app.state.ollama,
            prompt,
            facts,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ApprovalResponse(approval_id=approval_id, status=status, answer=answer)
