from __future__ import annotations

from typing import Any, Dict, Optional


def is_qwen_model(model_name: Optional[str]) -> bool:
    return "qwen" in str(model_name or "").lower()


def openai_thinking_extra_body(
    *,
    model_name: Optional[str],
    enable_thinking: bool,
) -> Optional[Dict[str, Any]]:
    """Build the Qwen OpenAI-compatible thinking controls.

    Non-Qwen providers do not understand these extension fields, so leave their
    request bodies unchanged. Qwen receives an explicit boolean in both modes;
    omitting the field when disabled can leave server-side thinking defaults on.
    """
    if not is_qwen_model(model_name):
        return None
    enabled = bool(enable_thinking)
    return {
        "chat_template_kwargs": {"enable_thinking": enabled},
        "include_reasoning": enabled,
    }
