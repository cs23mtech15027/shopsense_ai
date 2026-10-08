import logging
from uuid import uuid4

import chainlit as cl
import httpx

from app.config import get_settings

logger = logging.getLogger("shopsense.ui")
settings = get_settings()
api_url = settings.api_url.rstrip("/")


async def _post_chat(message: str, conversation_id: str, query_type: str) -> dict:
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            f"{api_url}/v1/chat",
            json={
                "customer_id": settings.demo_customer_id,
                "conversation_id": conversation_id,
                "message": message,
                "query_type": query_type,
            },
        )
        response.raise_for_status()
        return response.json()


async def _post_approval(approval_id: str, approved: bool) -> dict:
    if not settings.approval_token:
        raise RuntimeError(
            "Refund approval is disabled. Configure SHOPSENSE_APPROVAL_TOKEN "
            "in the API's .env file and restart the API."
        )
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{api_url}/v1/approvals/{approval_id}",
            headers={"X-Approval-Token": settings.approval_token},
            json={"approved": approved},
        )
        response.raise_for_status()
        return response.json()


def _error_message(error: httpx.HTTPStatusError) -> str:
    status = error.response.status_code
    if status == 403:
        return "Refund approval was not authorized. Check the approval token configured for the API."
    if status == 404:
        return "That approval request was not found. It may have already been removed."
    if status == 413:
        return "That message is too long. Please shorten it and try again."
    if status == 503:
        return "A required ShopSense service is unavailable or not configured. Check the API and local services."
    if status == 422 or status == 400:
        return "ShopSense could not process that request. Please rephrase it and try again."
    return f"ShopSense API returned HTTP {status}. Please try again."


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("conversation_id", str(uuid4()))
    settings_panel = cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="query_type",
                label="Help topic",
                items={
                    "Automatic": "auto",
                    "Order status / delivery": "order",
                    "Return / refund": "refund",
                    "Policy / shipping / other care": "care",
                },
                initial_value="auto",
            )
        ]
    )
    configured = await settings_panel.send()
    cl.user_session.set("query_type", configured.get("query_type", "auto"))
    await cl.Message(
        content=(
            "Welcome to ShopSense. Ask about an order or a store policy. "
            "This demo uses customer account **"
            f"{settings.demo_customer_id}**. This is a demo identifier, not verified "
            "authentication. Type `/new` to start a fresh conversation."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(updated_settings: dict) -> None:
    cl.user_session.set("query_type", updated_settings.get("query_type", "auto"))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = message.content.strip()
    if text.lower() == "/new":
        cl.user_session.set("conversation_id", str(uuid4()))
        await cl.Message(content="Started a new conversation. How can I help?").send()
        return

    conversation_id = cl.user_session.get("conversation_id")
    if not conversation_id:
        conversation_id = str(uuid4())
        cl.user_session.set("conversation_id", conversation_id)

    try:
        result = await _post_chat(
            text,
            conversation_id,
            cl.user_session.get("query_type", "auto"),
        )
    except httpx.HTTPStatusError as error:
        await cl.Message(content=_error_message(error)).send()
        return
    except httpx.RequestError:
        await cl.Message(
            content=(
                "I couldn't connect to the ShopSense API. Check that it is running "
                f"at `{api_url}` and try again."
            )
        ).send()
        return
    except (KeyError, ValueError, RuntimeError):
        logger.exception("Unexpected ShopSense UI request failure")
        await cl.Message(
            content="ShopSense couldn't complete that request. Please try again."
        ).send()
        return

    content = result["answer"]
    if result.get("sources"):
        source_list = "\n".join(f"- `{source}`" for source in result["sources"])
        content = f"{content}\n\n**Policy sources**\n{source_list}"
    content += f"\n\n_{result['intent'].title()} · sentiment: {result['sentiment']}_"

    actions = []
    approval_id = result.get("approval_id")
    if result.get("status") == "pending_approval" and approval_id:
        actions = [
            cl.Action(
                name="approve_refund",
                label="Approve refund",
                payload={"approval_id": approval_id},
                tooltip="Approve this demo refund request",
            ),
            cl.Action(
                name="decline_refund",
                label="Decline",
                payload={"approval_id": approval_id},
                tooltip="Decline this refund request",
            ),
        ]
    await cl.Message(content=content, actions=actions).send()


@cl.action_callback("approve_refund")
async def approve_refund(action: cl.Action) -> None:
    await _handle_approval(action, approved=True)


@cl.action_callback("decline_refund")
async def decline_refund(action: cl.Action) -> None:
    await _handle_approval(action, approved=False)


async def _handle_approval(action: cl.Action, approved: bool) -> None:
    approval_id = action.payload.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id:
        await cl.Message(content="This approval request is invalid.").send()
        return
    try:
        result = await _post_approval(approval_id, approved)
    except httpx.HTTPStatusError as error:
        await cl.Message(content=_error_message(error)).send()
        return
    except httpx.RequestError:
        await cl.Message(
            content="I couldn't connect to the ShopSense API. The approval was not confirmed."
        ).send()
        return
    except RuntimeError as error:
        await cl.Message(content=str(error)).send()
        return

    await cl.Message(content=result["answer"]).send()
