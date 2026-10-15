import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

import httpx


logger = logging.getLogger("shopsense.ollama")
_ANSWER_MAX_TOKENS = 64
_JSON_MAX_TOKENS = 128


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 45.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self._token_callback: ContextVar[
            Callable[[str], Awaitable[None]] | None
        ] = ContextVar(f"ollama_token_callback_{id(self)}", default=None)

    def _prepare_user_prompt(self, user_prompt: str) -> str:
        if self.model.split(":", 1)[0].lower().startswith("qwen3"):
            return f"{user_prompt.rstrip()}\n/no_think"
        return user_prompt

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        callback = self._token_callback.get()
        if callback is not None:
            tokens: list[str] = []
            async for token in self.stream(system_prompt, user_prompt):
                tokens.append(token)
                await callback(token)
            answer = "".join(tokens).strip()
            if not answer:
                raise RuntimeError("Local language model returned an empty response.")
            return answer
        return await self._generate(system_prompt, user_prompt)

    @contextmanager
    def stream_to(
        self,
        callback: Callable[[str], Awaitable[None]],
    ) -> Generator[None, None, None]:
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

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema: dict[str, object] | None = None,
    ) -> dict[str, object]:
        answer = await self._generate(
            system_prompt,
            user_prompt,
            json_format=True,
            json_schema=schema,
        )
        try:
            result = json.loads(answer)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Local language model returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Local language model returned an invalid JSON object.")
        return result

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_format: bool = False,
        json_schema: dict[str, object] | None = None,
    ) -> str:
        request = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._prepare_user_prompt(user_prompt)},
            ],
            "options": {
                "temperature": 0.35,
                "num_predict": _JSON_MAX_TOKENS if json_format else _ANSWER_MAX_TOKENS,
            },
        }
        if json_format:
            request["format"] = json_schema if json_schema is not None else "json"
        started = time.perf_counter()
        logger.info(
            "Ollama request started model=%s mode=%s",
            self.model,
            "json" if json_format else "answer",
        )
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=request,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "Ollama request failed model=%s mode=%s elapsed_ms=%.0f error=%s",
                self.model,
                "json" if json_format else "answer",
                (time.perf_counter() - started) * 1000,
                type(exc).__name__,
            )
            raise RuntimeError("Local language model is unavailable.") from exc

        try:
            body = response.json()
            answer = body["message"]["content"].strip()
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Ollama returned invalid response model=%s mode=%s elapsed_ms=%.0f",
                self.model,
                "json" if json_format else "answer",
                (time.perf_counter() - started) * 1000,
            )
            raise RuntimeError("Local language model returned an invalid response.") from exc
        if not answer:
            logger.warning(
                "Ollama returned empty response model=%s mode=%s elapsed_ms=%.0f",
                self.model,
                "json" if json_format else "answer",
                (time.perf_counter() - started) * 1000,
            )
            raise RuntimeError("Local language model returned an empty response.")
        timing_fields = {
            name: body.get(name)
            for name in (
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "prompt_eval_count",
                "eval_count",
            )
            if body.get(name) is not None
        }
        logger.info(
            "Ollama request completed model=%s mode=%s elapsed_ms=%.0f timings=%s",
            self.model,
            "json" if json_format else "answer",
            (time.perf_counter() - started) * 1000,
            timing_fields,
        )
        return answer

    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        request = {
            "model": self.model,
            "stream": True,
            "think": False,
            "keep_alive": "10m",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._prepare_user_prompt(user_prompt)},
            ],
            "options": {"temperature": 0.35, "num_predict": _ANSWER_MAX_TOKENS},
        }
        started = time.perf_counter()
        logger.info("Ollama streaming request started model=%s", self.model)
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=request,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            if not isinstance(event, dict):
                                raise RuntimeError(
                                    "Local language model returned an invalid stream."
                                )
                            if event.get("error"):
                                raise RuntimeError(
                                    "Local language model returned an error."
                                )
                            token = event["message"]["content"]
                        except (json.JSONDecodeError, KeyError, TypeError) as exc:
                            raise RuntimeError(
                                "Local language model returned an invalid stream."
                            ) from exc
                        if token:
                            yield token
            logger.info(
                "Ollama streaming request completed model=%s elapsed_ms=%.0f",
                self.model,
                (time.perf_counter() - started) * 1000,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "Ollama streaming request failed model=%s elapsed_ms=%.0f error=%s",
                self.model,
                (time.perf_counter() - started) * 1000,
                type(exc).__name__,
            )
            raise RuntimeError("Local language model is unavailable.") from exc
