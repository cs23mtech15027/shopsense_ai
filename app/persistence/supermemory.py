import hashlib
import logging
import re
from typing import Any

from supermemory import AsyncSupermemory, Supermemory

from app.guardrails.input import redact_pii

logger = logging.getLogger("shopsense.memory.supermemory")
_PASSWORD = re.compile(
    r"(?i)\b(?:password|passcode|pin)\b\s*(?:is|:|=)\s*['\"]?([^\s,;'\".]+)"
)
_MAX_MEMORY_CHARS = 1400


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _sanitize(value: str, api_key: str = "") -> str:
    cleaned = redact_pii(value)
    if api_key:
        cleaned = cleaned.replace(api_key, "[REDACTED_SECRET]")
    return _PASSWORD.sub("[REDACTED_SECRET]", cleaned)


class SupermemoryStore:
    def __init__(
        self,
        api_key: str | None,
        search_limit: int = 5,
        timeout_seconds: float = 5.0,
    ):
        if search_limit < 1:
            raise ValueError("Supermemory search limit must be positive.")
        self.api_key = api_key.strip() if api_key else ""
        self.search_limit = min(search_limit, 5)
        self.timeout_seconds = timeout_seconds
        self.client = (
            Supermemory(api_key=self.api_key, timeout=timeout_seconds)
            if self.api_key
            else None
        )
        self.async_client = (
            AsyncSupermemory(api_key=self.api_key, timeout=timeout_seconds)
            if self.api_key
            else None
        )

    def load_context(self, user_id: str, query: str = "") -> dict[str, str]:
        if self.client is None:
            return {"memory_summary": ""}

        facts: list[str] = []
        try:
            profile_response = self.client.profile(
                user_id,
                timeout=self.timeout_seconds,
            )
            profile = _read(profile_response, "profile", {})
            for category in ("static", "dynamic"):
                category_facts = _read(profile, category, [])
                if isinstance(category_facts, list):
                    facts.extend(
                        memory
                        for item in category_facts
                        if isinstance((memory := _read(item, "memory")), str)
                        and memory.strip()
                    )
        except Exception as error:
            logger.warning(
                "Supermemory read failed",
                extra={
                    "operation": "profile",
                    "user_id": user_id,
                    "error_type": type(error).__name__,
                },
            )

        try:
            results_response = self.client.search(
                user_id,
                query=_sanitize(query, self.api_key)
                or "Shopper preferences and resolved support context",
                search_mode="memories",
                limit=self.search_limit,
                timeout=self.timeout_seconds,
            )
            results = _read(results_response, "results", [])
            if isinstance(results, list):
                facts.extend(
                    memory
                    for item in results
                    if isinstance(
                        (
                            memory := _read(item, "memory")
                            or _read(item, "chunk")
                        ),
                        str,
                    )
                    and memory.strip()
                )
        except Exception as error:
            logger.warning(
                "Supermemory read failed",
                extra={
                    "operation": "search",
                    "user_id": user_id,
                    "error_type": type(error).__name__,
                },
            )

        sanitized_facts: list[str] = []
        for fact in facts:
            sanitized = _sanitize(fact, self.api_key).strip()
            if sanitized and sanitized not in sanitized_facts:
                sanitized_facts.append(sanitized)
        logger.info(
            "Supermemory context retrieved user_id=%s facts=%d",
            user_id,
            len(sanitized_facts),
        )
        return {"memory_summary": "\n".join(sanitized_facts)[:_MAX_MEMORY_CHARS]}

    async def remember_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        turn_id: str,
        message: str,
        answer: str,
        active_agent: str,
    ) -> None:
        if self.async_client is None:
            return

        content = (
            "Resolved ShopSense support interaction. "
            f"Customer context: {_sanitize(message, self.api_key)[:500]} "
            f"Resolution: {_sanitize(answer, self.api_key)[:700]}"
        )[:_MAX_MEMORY_CHARS]
        memory_id = hashlib.sha256(
            f"{thread_id}:{turn_id}".encode("utf-8")
        ).hexdigest()
        try:
            await self.async_client.add(
                user_id,
                content=content,
                id=f"shopsense-{memory_id}",
                dreaming="instant",
                metadata={
                    "source": "shopsense_support",
                    "agent": active_agent,
                },
                timeout=self.timeout_seconds,
            )
            logger.info(
                "Supermemory turn stored user_id=%s thread_id=%s agent=%s",
                user_id,
                thread_id,
                active_agent,
            )
        except Exception as error:
            logger.warning(
                "Supermemory write failed",
                extra={
                    "operation": "add",
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "error_type": type(error).__name__,
                },
            )
