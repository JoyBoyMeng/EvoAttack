from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class OpenAIEndpoint:
    api_key: Optional[str]
    base_url: Optional[str]


def resolve_openai_endpoint(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
) -> OpenAIEndpoint:
    load_dotenv()
    return OpenAIEndpoint(
        api_key=api_key if api_key is not None else os.getenv(api_key_env),
        base_url=base_url if base_url is not None else os.getenv(base_url_env),
    )


def make_openai_client(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
) -> OpenAI:
    endpoint = resolve_openai_endpoint(
        api_key=api_key,
        base_url=base_url,
        api_key_env=api_key_env,
        base_url_env=base_url_env,
    )
    return OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url)


def configure_target_openai_env(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    endpoint = resolve_openai_endpoint(api_key=api_key, base_url=base_url)
    if endpoint.api_key is not None:
        os.environ["OPENAI_API_KEY"] = endpoint.api_key
    if endpoint.base_url is not None:
        os.environ["OPENAI_BASE_URL"] = endpoint.base_url


def register_openai_compatible_target_model(model_name: str) -> None:
    """
    Let ASB's LLMKernel route an arbitrary OpenAI-compatible model name through
    the existing GPT wrapper. This is process-local and does not modify ASB code.
    """
    from aios.llm_core.llm_classes.gpt_llm import GPTLLM
    from aios.llm_core.llm_classes.model_registry import MODEL_REGISTRY

    if model_name not in MODEL_REGISTRY:
        MODEL_REGISTRY[model_name] = GPTLLM
