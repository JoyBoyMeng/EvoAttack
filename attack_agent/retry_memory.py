from __future__ import annotations

from typing import Any

_STATS = {
    "mem0_read_failed": 0,
    "mem0_write_failed": 0,
}


class FaultTolerantVectorDB:
    """
    Keep the experiment driver alive when Mem0 still fails after API-level retries.

    API retries happen in attack_agent.api_retry. This wrapper only decides the
    fallback semantics after those retries are exhausted: memory read returns no
    retrieved memories; memory write is skipped.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def similarity_search(self, *args: Any, **kwargs: Any) -> list[Any]:
        try:
            return self.inner.similarity_search(*args, **kwargs)
        except Exception as exc:
            _STATS["mem0_read_failed"] += 1
            print(
                f"[AdaptiveAttack::MEM0_READ_FAILED] returning empty memory results error={repr(exc)}",
                flush=True,
            )
            return []

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return self.similarity_search(*args, **kwargs)

    def similarity_search_with_score(self, *args: Any, **kwargs: Any) -> list[Any]:
        try:
            return self.inner.similarity_search_with_score(*args, **kwargs)
        except Exception as exc:
            _STATS["mem0_read_failed"] += 1
            print(
                f"[AdaptiveAttack::MEM0_READ_FAILED] returning empty scored memory results error={repr(exc)}",
                flush=True,
            )
            return []

    def add_texts(self, *args: Any, **kwargs: Any) -> list[Any]:
        try:
            return self.inner.add_texts(*args, **kwargs)
        except Exception as exc:
            _STATS["mem0_write_failed"] += 1
            print(
                f"[AdaptiveAttack::MEM0_WRITE_FAILED] skipped target memory write error={repr(exc)}",
                flush=True,
            )
            return []

    def add_documents(self, *args: Any, **kwargs: Any) -> list[Any]:
        try:
            return self.inner.add_documents(*args, **kwargs)
        except Exception as exc:
            _STATS["mem0_write_failed"] += 1
            print(
                f"[AdaptiveAttack::MEM0_WRITE_FAILED] skipped target memory document write error={repr(exc)}",
                flush=True,
            )
            return []

    def clone_memory_records(self, *args: Any, **kwargs: Any) -> list[Any]:
        try:
            return self.inner.clone_memory_records(*args, **kwargs)
        except Exception as exc:
            _STATS["mem0_write_failed"] += 1
            print(
                "[AdaptiveAttack::MEM0_WRITE_FAILED] skipped target memory "
                f"replay error={repr(exc)}",
                flush=True,
            )
            return []

    def persist(self) -> Any:
        if hasattr(self.inner, "persist"):
            return self.inner.persist()
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def get_memory_failure_stats() -> dict[str, int]:
    return dict(_STATS)
