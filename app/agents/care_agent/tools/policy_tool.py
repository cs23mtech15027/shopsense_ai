from threading import Lock

from app.retrieval.policies import PolicyStore


class PolicyTool:
    def __init__(self, store_factory, top_k: int):
        self.store_factory = store_factory
        self._store: PolicyStore | None = None
        self._store_lock = Lock()
        self.top_k = top_k

    def search_policy(self, query: str) -> list[dict[str, str]]:
        if self._store is None:
            with self._store_lock:
                if self._store is None:
                    self._store = self.store_factory()
        return self._store.search(query, self.top_k)
