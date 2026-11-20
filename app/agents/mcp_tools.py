import json
import logging
import re

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import MCP_SERVER_URL
from app.guardrails.input import redact_pii

_ORDER_ID = re.compile(r"\bORD[-_ ]?\d{4,}\b", re.IGNORECASE)
_SENSITIVE_LOG_FIELDS = {
    "approval_id",
    "amount_minor",
    "confirmed_order_id",
    "currency",
    "order_id",
}
logger = logging.getLogger("shopsense.mcp.client")


def extract_order_id(message: str) -> str | None:
    match = _ORDER_ID.search(message)
    if match is None:
        return None
    return match.group().upper().replace("_", "-").replace(" ", "-")


def _redact_tool_log(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if key.casefold() in _SENSITIVE_LOG_FIELDS
            else _redact_tool_log(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_tool_log(item) for item in value]
    if isinstance(value, str):
        return redact_pii(_ORDER_ID.sub("[REDACTED_ORDER_ID]", value))
    return value


async def call_oms_tool(
    name: str,
    arguments: dict[str, object],
    user_jwt: str,
    *,
    thread_id: str,
    trace_id: str,
) -> dict | list[dict]:
    client = MultiServerMCPClient(
        {
            "shopsense": {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
                "headers": {
                    "Authorization": f"Bearer {user_jwt}",
                    "X-ShopSense-Thread-Id": thread_id,
                    "X-ShopSense-Trace-Id": trace_id,
                },
            }
        }
    )
    tools = await client.get_tools()
    tool = next((candidate for candidate in tools if candidate.name == name), None)
    if tool is None:
        raise RuntimeError(f"The local ShopSense MCP server does not expose {name!r}.")

    logger.info(
        "MCP tool started name=%s thread_id=%s trace_id=%s",
        name,
        thread_id,
        trace_id,
    )
    logger.debug(
        "MCP tool input name=%s arguments=%s",
        name,
        json.dumps(_redact_tool_log(arguments), ensure_ascii=False)[:2000],
    )
    response = await tool.ainvoke(arguments)
    content = getattr(response, "content", response)
    if isinstance(content, str):
        decoded = json.loads(content)
    elif isinstance(content, list):
        text_blocks = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if not text_blocks:
            raise RuntimeError(f"MCP tool {name!r} returned no structured text.")
        decoded = json.loads("\n".join(text_blocks))
    else:
        decoded = content
    if not isinstance(decoded, (dict, list)):
        raise RuntimeError(f"MCP tool {name!r} returned an invalid response.")
    logger.debug(
        "MCP tool output name=%s result=%s",
        name,
        json.dumps(
            _redact_tool_log(decoded),
            ensure_ascii=False,
            default=str,
        )[:2000],
    )
    logger.info(
        "MCP tool completed name=%s thread_id=%s trace_id=%s",
        name,
        thread_id,
        trace_id,
    )
    return decoded
