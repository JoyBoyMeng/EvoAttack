"""Canonical text templates for target-memory embedding and retrieval.

These helpers only build text. They do not call an embedding model, query
Mem0, perform BM25 search, or rank retrieved memories. The full memory payload
should be stored separately from the compact text returned here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any, TypeAlias


ToolLike: TypeAlias = str | Mapping[str, Any]

__all__ = [
    "build_memory_embedding_text",
    "build_memory_retrieval_query",
    "extract_tools_used",
]


ACTION_TOOL_PATTERN = re.compile(
    r"\[Action\]\s*:\s*I will call the\s+([A-Za-z0-9_.-]+)\s+with\b",
    flags=re.IGNORECASE,
)


def _normalize_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _extract_tool_name(tool: ToolLike) -> str:
    if isinstance(tool, str):
        return _normalize_text(tool, field_name="tool name")

    if not isinstance(tool, Mapping):
        raise TypeError("each tool must be a string or mapping")

    function = tool.get("function")
    if isinstance(function, Mapping) and function.get("name") is not None:
        return _normalize_text(str(function["name"]), field_name="tool name")

    if tool.get("name") is not None:
        return _normalize_text(str(tool["name"]), field_name="tool name")

    raise ValueError("tool mapping must contain 'name' or 'function.name'")


def _canonical_tool_names(tools: Iterable[ToolLike]) -> tuple[str, ...]:
    if isinstance(tools, (str, bytes)):
        tools = [tools.decode() if isinstance(tools, bytes) else tools]
    try:
        names = {_extract_tool_name(tool) for tool in tools}
    except TypeError as exc:
        raise TypeError("tools must be an iterable of tool names or mappings") from exc
    return tuple(sorted(names))


def _format_tool_names(tools: Iterable[ToolLike]) -> str:
    names = _canonical_tool_names(tools)
    return ", ".join(names) if names else "NONE"


def extract_tools_used(memory_text: str) -> tuple[str, ...]:
    """Extract actually executed tool names from an ASB episode memory body."""

    return tuple(sorted(set(ACTION_TOOL_PATTERN.findall(str(memory_text)))))


def build_memory_embedding_text(
    task: str,
    available_tools: Iterable[ToolLike],
    tools_used: Iterable[ToolLike],
) -> str:
    """Build the compact text embedded when a memory is written.

    ``available_tools`` contains the complete tool set associated with the
    memory. For benign init memories this is the two normal tools plus the
    attack-tool provenance label. The attack tool is not exposed at runtime.
    ``tools_used`` contains only tools actually called in the remembered
    interaction. Tool descriptions and the full memory body are deliberately
    excluded so that the embedding focuses on task and tool identity.

    Example output::

        TASK: Monitor unauthorized network access.
        AVAILABLE_TOOLS: CredentialHarvesting, sys_monitor, update_manager
        TOOLS_USED: sys_monitor
    """

    normalized_task = _normalize_text(task, field_name="task")
    normalized_available_tools = _format_tool_names(available_tools)
    normalized_used_tools = _format_tool_names(tools_used)
    return (
        f"TASK: {normalized_task}\n"
        f"AVAILABLE_TOOLS: {normalized_available_tools}\n"
        f"TOOLS_USED: {normalized_used_tools}"
    )


def build_memory_retrieval_query(
    task: str,
    available_tools: Iterable[ToolLike],
) -> str:
    """Build the query text used for semantic target-memory candidate recall.

    ``available_tools`` may contain plain names, OpenAI function-tool objects,
    or tool-call mappings. The returned query is used for top-50 semantic
    candidate recall. Within those candidates, exact ``task_hash`` and
    order-independent ``available_tools`` matches receive priority.

    Example output::

        TASK: Monitor unauthorized network access.
        AVAILABLE_TOOLS: firewall_manager, sys_monitor
    """

    normalized_task = _normalize_text(task, field_name="task")
    normalized_tools = _format_tool_names(available_tools)
    return f"TASK: {normalized_task}\nAVAILABLE_TOOLS: {normalized_tools}"
