import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

import httpx

from app.guardrails.input import redact_pii

logger = logging.getLogger("shopsense.llm")


class CloudModelUnavailableError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 45.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_rate_limit_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.max_rate_limit_retries = max(0, max_rate_limit_retries)
        self._token_callback: ContextVar[Callable[[str], Awaitable[None]] | None] = ContextVar(
            f"openai_token_callback_{id(self)}", default=None
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        answer = await self._generate(system_prompt, user_prompt)
        if not answer:
            raise RuntimeError("Cloud language model returned an empty response.")
        return answer

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema: dict[str, object] | None = None,
    ) -> dict[str, object]:
        extra = {}
        if schema is not None:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "shopsense_response",
                    "strict": True,
                    "schema": schema,
                },
            }
        answer = await self._generate(system_prompt, user_prompt, extra=extra)
        try:
            result = json.loads(answer)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Cloud language model returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Cloud language model returned an invalid JSON object.")
        return result

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        extra: dict[str, object] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.35,
            "max_completion_tokens": 1024,
            "stream": False,
        }
        if self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            payload["reasoning_effort"] = "low"
        if extra:
            payload.update(extra)
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                for attempt in range(self.max_rate_limit_retries + 1):
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
                    if (
                        response.status_code == 429
                        and attempt < self.max_rate_limit_retries
                    ):
                        delay = self._rate_limit_delay(response, attempt)
                        logger.warning(
                            "Cloud model rate limited model=%s retry=%s/%s "
                            "retry_after_seconds=%.2f",
                            self.model,
                            attempt + 1,
                            self.max_rate_limit_retries,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    response.raise_for_status()
                    break
        except httpx.HTTPError as exc:
            response = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
            provider_error = self._provider_error(response)
            logger.warning(
                "Cloud model request failed model=%s elapsed_ms=%.0f "
                "error_type=%s status_code=%s request_id=%s provider_error=%s",
                self.model,
                (time.perf_counter() - started) * 1000,
                type(exc).__name__,
                response.status_code if response is not None else "-",
                response.headers.get("x-request-id", "-") if response is not None else "-",
                provider_error,
            )
            raise CloudModelUnavailableError(
                "Cloud language model is unavailable."
            ) from exc
        try:
            body = response.json()
            answer = body["choices"][0]["message"]["content"]
            if not isinstance(answer, str):
                raise ValueError("Content is not text")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Cloud language model returned an invalid response.") from exc
        logger.info(
            "Cloud model request completed model=%s elapsed_ms=%.0f",
            self.model,
            (time.perf_counter() - started) * 1000,
        )
        return answer.strip()

    @staticmethod
    def _rate_limit_delay(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after", "").strip()
        try:
            delay = float(retry_after)
        except ValueError:
            reset = response.headers.get("x-ratelimit-reset-tokens", "")
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)(ms|s|m)?\s*", reset)
            if match:
                value = float(match.group(1))
                unit = match.group(2) or "s"
                delay = value / 1000 if unit == "ms" else value * (60 if unit == "m" else 1)
            else:
                delay = float(2**attempt)
        return min(max(delay, 0.1), 60.0)

    def _provider_error(self, response: httpx.Response | None) -> str:
        if response is None:
            return "-"
        try:
            body = response.json()
        except ValueError:
            return "-"
        if not isinstance(body, dict):
            return "-"
        error = body.get("error")
        if not isinstance(error, dict):
            return "-"
        message = error.get("message")
        if not isinstance(message, str):
            return "-"
        safe_message = redact_pii(message).replace(self.api_key, "[REDACTED_SECRET]")
        safe_message = " ".join(safe_message.split())
        return safe_message[:300] or "-"

    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        answer = await self.generate(system_prompt, user_prompt)
        for token in answer.split():
            yield token + " "

    @contextmanager
    def stream_to(self, callback: Callable[[str], Awaitable[None]]) -> Generator[None, None, None]:
        token = self._token_callback.set(callback)
        try:
            yield
        finally:
            self._token_callback.reset(token)

    @contextmanager
    def without_streaming(self) -> Generator[None, None, None]:
        token = self._token_callback.set(None)
        try:
            yield
        finally:
            self._token_callback.reset(token)
