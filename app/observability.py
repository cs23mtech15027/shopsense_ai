import atexit
import hashlib
import logging
import os
import re
import threading
from contextlib import contextmanager
from collections.abc import Generator, Mapping
from typing import Any

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.config import Settings, get_settings
from app.guardrails.input import redact_pii

logger = logging.getLogger("shopsense.observability")
_API_KEY_ASSIGNMENT = re.compile(
    r"(?i)\b(GROQ_API_KEY|QDRANT_API_KEY|SUPERMEMORY_API_KEY)\b"
    r"(\s*[:=]\s*)(['\"]?)[^\s,'\";]+"
)
_CARD_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){12,15}\d(?!\d)")
_CLIENT_LOCK = threading.Lock()
_client: Langfuse | None = None
_client_identity: tuple[str, str, str] | None = None
_configured_secrets: set[str] = set()


def _register_configured_secrets(settings: Settings) -> None:
    for secret in (
        settings.groq_api_key,
        settings.qdrant_api_key,
        settings.supermemory_api_key,
        settings.langfuse_secret_key,
        settings.jwt_secret,
    ):
        if secret is not None:
            value = secret.get_secret_value()
            if value:
                _configured_secrets.add(value)


def redact_sensitive_data(data: Any) -> Any:
    if isinstance(data, str):
        redacted = redact_pii(data)
        redacted = _API_KEY_ASSIGNMENT.sub(
            lambda match: f"{match.group(1)}{match.group(2)}"
            f"{match.group(3)}[REDACTED_SECRET]",
            redacted,
        )
        redacted = _CARD_NUMBER.sub("[REDACTED_PAYMENT_NUMBER]", redacted)
        for secret in _configured_secrets:
            redacted = redacted.replace(secret, "[REDACTED_SECRET]")
        for name in (
            "GROQ_API_KEY",
            "QDRANT_API_KEY",
            "SUPERMEMORY_API_KEY",
        ):
            value = os.getenv(name, "")
            if value:
                redacted = redacted.replace(value, "[REDACTED_SECRET]")
        return redacted
    if isinstance(data, Mapping):
        return {
            redact_sensitive_data(key) if isinstance(key, str) else key:
            redact_sensitive_data(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [redact_sensitive_data(value) for value in data]
    if isinstance(data, tuple):
        return tuple(redact_sensitive_data(value) for value in data)
    if isinstance(data, set):
        return {redact_sensitive_data(value) for value in data}
    if hasattr(data, "model_dump"):
        return redact_sensitive_data(data.model_dump())
    if isinstance(data, int) and not isinstance(data, bool):
        digits = str(abs(data))
        if 13 <= len(digits) <= 16:
            return "[REDACTED_PAYMENT_NUMBER]"
    return data


def get_client(settings: Settings | None = None) -> Langfuse | None:
    global _client, _client_identity
    settings = settings or get_settings()
    if (
        not settings.langfuse_enabled
        or settings.langfuse_public_key is None
        or settings.langfuse_secret_key is None
    ):
        return None

    public_key = settings.langfuse_public_key.get_secret_value()
    secret_key = settings.langfuse_secret_key.get_secret_value()
    identity = (
        public_key,
        hashlib.sha256(secret_key.encode("utf-8")).hexdigest(),
        settings.langfuse_base_url,
    )
    with _CLIENT_LOCK:
        _register_configured_secrets(settings)
        if _client is not None and _client_identity == identity:
            return _client
        try:
            _client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=settings.langfuse_base_url,
                mask=redact_sensitive_data,
            )
        except Exception as error:
            logger.warning(
                "Langfuse initialization failed error_type=%s",
                type(error).__name__,
            )
            _client = None
            _client_identity = None
            return None
        _client_identity = identity
        return _client


def get_callback_handler(client: Langfuse | None) -> CallbackHandler | None:
    return CallbackHandler() if client is not None else None


@contextmanager
def trace_turn(
    client: Langfuse | None,
    *,
    user_id: str,
    thread_id: str,
    trace_id: str,
    message: str,
) -> Generator[Any | None, None, None]:
    if client is None:
        yield None
        return

    with client.start_as_current_observation(
        as_type="span",
        name="shopsense-turn",
        trace_context={"trace_id": trace_id},
        input={"message": redact_sensitive_data(message)},
    ) as span:
        with propagate_attributes(
            user_id=user_id,
            session_id=thread_id,
            tags=["shopsense", "hybrid"],
        ):
            yield span


def flush_observability(client: Langfuse | None = None) -> None:
    target = client or _client
    if target is None:
        return
    try:
        target.flush()
    except Exception as error:
        logger.warning(
            "Langfuse flush failed error_type=%s",
            type(error).__name__,
        )


atexit.register(flush_observability)
