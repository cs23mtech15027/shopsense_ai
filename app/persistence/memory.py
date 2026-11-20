from typing import Protocol


class MemoryStore(Protocol):
    def load_context(self, user_id: str, query: str = "") -> dict[str, str]: ...

    async def remember_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        turn_id: str,
        message: str,
        answer: str,
        active_agent: str,
    ) -> None: ...
