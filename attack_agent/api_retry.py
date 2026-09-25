from __future__ import annotations

import functools
import time
from typing import Any, Callable


_INSTALLED = False
_STATS = {
    "api_retry_failures": 0,
    "api_retry_exhausted": 0,
}


def _retry_call(
    fn: Callable[..., Any],
    *,
    api_name: str,
    max_retries: int,
    retry_sleep: float,
) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        attempts = max(1, int(max_retries))
        for attempt in range(1, attempts + 1):
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:
                _STATS["api_retry_failures"] += 1
                print(
                    f"[AdaptiveAttack::API_RETRY] api={api_name} "
                    f"attempt={attempt}/{attempts} error={repr(exc)}",
                    flush=True,
                )
                if attempt >= attempts:
                    _STATS["api_retry_exhausted"] += 1
                    raise
                if retry_sleep > 0:
                    time.sleep(retry_sleep)
        raise RuntimeError("unreachable retry state")

    return wrapped


def install_openai_api_retries(*, max_retries: int = 3, retry_sleep: float = 5.0) -> None:
    """
    Install per-OpenAI-API-call retries without modifying ASB source files.

    This wraps the OpenAI SDK resource methods used by both ASB and Mem0:
    chat.completions.create and embeddings.create. A failed API call is retried
    in place; the target-agent episode is not restarted.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from openai.resources.chat.completions import Completions
    from openai.resources.embeddings import Embeddings

    Completions.create = _retry_call(
        Completions.create,
        api_name="chat.completions.create",
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    Embeddings.create = _retry_call(
        Embeddings.create,
        api_name="embeddings.create",
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    _INSTALLED = True


def get_api_retry_stats() -> dict[str, int]:
    return dict(_STATS)
