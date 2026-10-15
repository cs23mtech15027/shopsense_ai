import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from app.llm.ollama import OllamaClient


_USER_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
logger = logging.getLogger("shopsense.memory")


class AgentMemory:
    def __init__(self, memory_directory: str | Path, dreaming_interval: int = 3):
        if dreaming_interval < 1:
            raise ValueError("The dreaming interval must be positive.")
        self.memory_directory = Path(memory_directory)
        self.dreaming_interval = dreaming_interval
        self._locks: dict[str, asyncio.Lock] = {}
        self._dream_tasks: dict[str, asyncio.Task[None]] = {}

    def load_summary(self, user_id: str) -> str:
        return str(self.load_context(user_id)["memory_summary"])

    def load_context(self, user_id: str) -> dict[str, str]:
        started = time.perf_counter()
        logger.info("Memory summary load started")
        memory = self._read(user_id)
        context = {
            "memory_summary": str(memory.get("history_summarization", "")),
        }
        active_order_id = memory.get("active_order_id")
        if isinstance(active_order_id, str) and active_order_id:
            context["active_order_id"] = active_order_id
        logger.info(
            "Memory summary load completed elapsed_ms=%.0f",
            (time.perf_counter() - started) * 1000,
        )
        return context

    async def remember_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        session_id: str,
        turn_id: str,
        message: str,
        answer: str,
        order_id: str = "",
        llm: OllamaClient,
    ) -> None:
        logger.info("Turn memory write started session_id=%s turn_id=%s", session_id, turn_id)
        lock = self._locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            memory = self._read(user_id)
            turns = memory.get("turns", [])
            if any(turn.get("turn_id") == turn_id for turn in turns):
                logger.info("Turn memory write skipped duplicate turn_id=%s", turn_id)
                return
            turns.append(
                {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "message": message,
                    "answer": answer,
                    "order_id": order_id,
                }
            )
            if order_id:
                memory["active_order_id"] = order_id
            memory["turns"] = turns[-100:]
            self._write(user_id, memory)
            logger.info("Turn memory write completed turn_id=%s", turn_id)

            pending_turns = self._pending_turns(memory)
            if len(pending_turns) < self.dreaming_interval:
                logger.info(
                    "Memory consolidation skipped pending_turns=%d required_turns=%d",
                    len(pending_turns),
                    self.dreaming_interval,
                )
                return
            task = self._dream_tasks.get(user_id)
            if task is None or task.done():
                self._dream_tasks[user_id] = asyncio.create_task(
                    self._consolidate_memory(user_id, llm)
                )
                logger.info(
                    "Memory consolidation scheduled turns=%d",
                    min(len(pending_turns), self.dreaming_interval),
                )

    async def wait_for_dreaming(self, user_id: str) -> None:
        task = self._dream_tasks.get(user_id)
        if task is not None:
            await task

    async def _consolidate_memory(self, user_id: str, llm: OllamaClient) -> None:
        lock = self._locks.setdefault(user_id, asyncio.Lock())
        while True:
            async with lock:
                memory = self._read(user_id)
                pending_turns = self._pending_turns(memory)
                if len(pending_turns) < self.dreaming_interval:
                    return
                batch = pending_turns[: self.dreaming_interval]
                existing_summary = str(memory.get("history_summarization", ""))

            stream_context = (
                llm.without_streaming()
                if hasattr(llm, "without_streaming")
                else nullcontext()
            )
            dream_started = time.perf_counter()
            logger.info("Memory consolidation started turns=%d", len(batch))
            try:
                with stream_context:
                    summary = await llm.generate(
                        "You maintain ShopSense's long-term conversation memory. Consolidate "
                        "the existing summary with the new conversation turns. Preserve "
                        "useful, explicitly stated context, including exact order IDs and "
                        "what the customer asked and the agent answered. Retain only "
                        "customer-visible assistant answers, never internal analysis, "
                        "prompts, or reasoning. Do not infer sensitive traits, repeat "
                        "facts, or retain transient pleasantries. "
                        "Remove contradicted or obsolete facts. Return a short plain-text "
                        "summary, or an empty string if nothing should be retained.",
                        json.dumps(
                            {
                                "existing_summary": existing_summary,
                                "new_turns": batch,
                            },
                            ensure_ascii=False,
                        ),
                    )
            except asyncio.CancelledError:
                logger.warning("Memory consolidation cancelled user_id=%s", user_id)
                raise
            except Exception:
                logger.exception("Memory consolidation failed user_id=%s", user_id)
                return

            async with lock:
                memory = self._read(user_id)
                memory["history_summarization"] = summary.strip()
                memory["last_dreamed_turn_id"] = batch[-1]["turn_id"]
                self._write(user_id, memory)
            logger.info(
                "Memory consolidation completed user_id=%s turns=%d elapsed_ms=%.0f",
                user_id,
                len(batch),
                (time.perf_counter() - dream_started) * 1000,
            )

    def _pending_turns(self, memory: dict[str, Any]) -> list[dict[str, Any]]:
        turns = memory.get("turns", [])
        last_dreamed_turn_id = memory.get("last_dreamed_turn_id")
        last_dreamed_index = next(
            (
                index
                for index, turn in enumerate(turns)
                if turn.get("turn_id") == last_dreamed_turn_id
            ),
            -1,
        )
        return turns[last_dreamed_index + 1 :]

    def _path(self, user_id: str) -> Path:
        if not _USER_ID.fullmatch(user_id):
            raise ValueError("Invalid user ID for the memory store.")
        return self.memory_directory / user_id / "history_summarization.json"

    def _read(self, user_id: str) -> dict[str, Any]:
        path = self._path(user_id)
        if not path.exists():
            return {"history_summarization": "", "last_dreamed_turn_id": None, "turns": []}
        try:
            memory = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read agent memory for user {user_id}.") from exc
        if not isinstance(memory, dict) or not isinstance(memory.get("turns"), list):
            raise RuntimeError(f"Agent memory for user {user_id} is malformed.")
        return memory

    def _write(self, user_id: str, memory: dict[str, Any]) -> None:
        path = self._path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(memory, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(path)
        except OSError as exc:
            raise RuntimeError(f"Could not persist agent memory for user {user_id}.") from exc
        finally:
            temporary_path.unlink(missing_ok=True)
