from __future__ import annotations

import os
from functools import wraps
from typing import Any, Dict, Optional

from mem0 import Memory

from aios.llm_core.thinking import openai_thinking_extra_body
from eval_all_agents_mem0_agent_isolated import Mem0VectorDBAdapter, _drop_none

from .local_model_config import resolve_openai_endpoint


def _provider_config(
    *,
    provider: str,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {"model": model}
    if provider == "openai":
        endpoint = resolve_openai_endpoint(api_key=api_key, base_url=base_url)
        config.update(
            {
                "api_key": endpoint.api_key,
                "openai_base_url": endpoint.base_url,
            }
        )
    else:
        config.update(
            {
                "api_key": api_key,
                "openai_base_url": base_url,
            }
        )
    if temperature is not None:
        config["temperature"] = temperature
    return config


def build_configurable_mem0_adapter(
    *,
    path: str,
    collection_name: str,
    user_id: str,
    agent_id: str,
    top_k: int,
    infer: bool,
    llm_model: str,
    embedding_model: str,
    namespace: str,
    llm_provider: str = "openai",
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    embedding_provider: str = "openai",
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    enable_thinking: bool = False,
    retrieval_mode: str = "similarity",
    use_real_embedding: bool = True,
    write_embedding_mode: str = "memory_text",
) -> Mem0VectorDBAdapter:
    os.makedirs(path, exist_ok=True)
    embedding_config = _provider_config(
        provider=embedding_provider,
        model=embedding_model,
        api_key=embedding_api_key,
        base_url=embedding_base_url,
    )
    if not use_real_embedding and embedding_provider == "openai":
        # The OpenAI client requires a non-empty key at construction time even
        # though placeholder-vector mode never sends an embedding request.
        embedding_config["api_key"] = embedding_config.get("api_key") or "placeholder-not-used"

    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": collection_name,
                "path": path,
            },
        },
        "llm": {
            "provider": llm_provider,
            "config": _provider_config(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
                temperature=0.0,
            ),
        },
        "embedder": {
            "provider": embedding_provider,
            "config": embedding_config,
        },
    }
    memory = Memory.from_config(_drop_none(config))
    extra_body = openai_thinking_extra_body(
        model_name=llm_model,
        enable_thinking=enable_thinking,
    )
    if extra_body is not None and llm_provider == "ollama":
        chat = memory.llm.client.chat

        @wraps(chat)
        def chat_with_thinking(*args: Any, **kwargs: Any) -> Any:
            kwargs["think"] = bool(enable_thinking)
            return chat(*args, **kwargs)

        memory.llm.client.chat = chat_with_thinking
    elif extra_body is not None:
        generate_response = memory.llm.generate_response

        @wraps(generate_response)
        def generate_response_with_thinking(*args: Any, **kwargs: Any) -> Any:
            request_extra_body = dict(kwargs.get("extra_body") or {})
            request_extra_body.update(extra_body)
            kwargs["extra_body"] = request_extra_body
            return generate_response(*args, **kwargs)

        memory.llm.generate_response = generate_response_with_thinking
    return Mem0VectorDBAdapter(
        memory=memory,
        user_id=user_id,
        agent_id=agent_id,
        top_k=top_k,
        infer=infer,
        namespace=namespace,
        retrieval_mode=retrieval_mode,
        use_real_embedding=use_real_embedding,
        write_embedding_mode=write_embedding_mode,
        embedding_model_name=embedding_model,
    )
