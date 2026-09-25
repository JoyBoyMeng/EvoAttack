from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


ASB_ROOT = Path(__file__).resolve().parents[1]


def _config_path(agent_path: str) -> Path:
    candidate = Path(agent_path)
    if candidate.is_absolute():
        return candidate / "config.json"
    if candidate.parts[:2] == ("pyopenagi", "agents"):
        return ASB_ROOT / candidate / "config.json"
    return ASB_ROOT / "pyopenagi" / "agents" / candidate / "config.json"


@lru_cache(maxsize=None)
def load_agent_generation_context(
    agent_name: str,
    agent_path: str,
    tools_info_path: str,
) -> tuple[str, List[Dict[str, str]]]:
    """Return the target role and its configured normal-tool descriptions."""
    with _config_path(agent_path).open("r", encoding="utf-8") as f:
        config = json.load(f)

    raw_description = config.get("description", "")
    if isinstance(raw_description, list):
        agent_description = " ".join(str(part).strip() for part in raw_description).strip()
    else:
        agent_description = str(raw_description).strip()

    configured_names = [
        str(item).rstrip("/").split("/")[-1]
        for item in config.get("tools", [])
    ]
    tools_path = Path(tools_info_path)
    if not tools_path.is_absolute():
        tools_path = ASB_ROOT / tools_path
    by_name: Dict[str, Dict[str, Any]] = {}
    with tools_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("Corresponding Agent", "")) != agent_name:
                continue
            name = str(record.get("Tool Name", ""))
            if name:
                by_name[name] = record

    normal_tools: List[Dict[str, str]] = []
    for name in configured_names:
        record = by_name.get(name, {})
        normal_tools.append(
            {
                "name": name,
                "description": str(record.get("Description", "")),
            }
        )
    return agent_description, normal_tools
