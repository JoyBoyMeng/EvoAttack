from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


AttackUnit = Tuple[str, str, str]
MULTI_ROUND_HISTORY_FORMAT = "state_aware_multi_round_v1"


@dataclass
class AttackState:
    agent: str
    task: str
    attack_tool: str
    local_history: List[Tuple[str | int, int]] = field(default_factory=list)
    step_id: int = 0
    history_format: str = "tool_then_global_all_results_no_score_v5"
    state_score: float = 0.0
    attack_history: List[str] = field(default_factory=list)
    train_round: int = 0
    attack_round: int = 1

    def key(self) -> AttackUnit:
        return (self.agent, self.task, self.attack_tool)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["local_history"] = [
            [action_id, int(success)]
            for action_id, success in self.local_history
        ]
        data["attack_history"] = [str(action_id) for action_id in self.attack_history]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackState":
        history = data.get("local_history", [])
        parsed_history: List[Tuple[str | int, int]] = []
        for item in history:
            if isinstance(item, dict):
                action_id = item.get("action_id", 0)
                action_text = str(action_id)
                action_id = (
                    int(action_text)
                    if action_text.lstrip("-").isdigit()
                    else action_text
                )
                parsed_history.append((action_id, int(item.get("success", 0))))
            elif isinstance(item, Sequence) and len(item) >= 2:
                action_text = str(item[0])
                action_id = (
                    int(action_text)
                    if action_text.lstrip("-").isdigit()
                    else action_text
                )
                parsed_history.append((action_id, int(item[1])))
        return cls(
            agent=str(data.get("agent", "")),
            task=str(data.get("task", "")),
            attack_tool=str(data.get("attack_tool", "")),
            local_history=parsed_history,
            step_id=int(data.get("step_id", 0)),
            history_format=str(data.get("history_format", "legacy_local_history")),
            state_score=float(data.get("state_score", 0.0)),
            attack_history=[str(item) for item in data.get("attack_history", [])],
            train_round=int(data.get("train_round", 0)),
            attack_round=int(data.get("attack_round", data.get("step_id", 0) + 1)),
        )


@dataclass
class AttackStrategy:
    strategy_id: str
    name: str
    description: str
    template: str
    instruction: str = ""
    delivery_mode: str = "task_text"

    def render(self, *, task: str, attack_tool: str, attacker_instruction: str) -> str:
        return self.template.format(
            task=task,
            attack_tool=attack_tool,
            attacker_instruction=attacker_instruction,
        )

    def to_prompt_block(self) -> str:
        if self.instruction:
            return f"{self.strategy_id}. {self.name}: {self.description} Instruction: {self.instruction}"
        return f"{self.strategy_id}. {self.name}: {self.description}"


@dataclass
class TargetObservation:
    agent: str
    task: str
    attack_tool: str
    attacked: bool
    called_attack_tool: bool
    attack_success: bool
    original_success: bool
    malicious_memory_retrieved: Optional[bool] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    raw_result: Dict[str, Any] = field(default_factory=dict)

    def to_context_message(self, max_chars: int = 3000) -> str:
        status = "malicious" if self.attacked else "benign"
        called = "called" if self.called_attack_tool else "did not call"
        original = "completed" if self.original_success else "did not complete"
        header = (
            f"Previous {status} interaction: target agent {called} the attack tool "
            f"{self.attack_tool}; original task was {original}."
        )
        transcript_parts = []
        for message in self.messages[-8:]:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "unknown")
            content = str(message.get("content", ""))
            if content:
                transcript_parts.append(f"{role}: {content}")
        transcript = "\n".join(transcript_parts)
        text = header if not transcript else f"{header}\nTranscript excerpt:\n{transcript}"
        if len(text) > max_chars:
            text = text[:max_chars] + " ...[truncated]"
        return text


@dataclass
class Transition:
    state: AttackState
    action_id: str
    action_text: str
    reward: float
    next_state: AttackState
    malicious_success: int = 0
    benign_success: int = 0
    persistent_success: int = 0
    trajectory_id: str = ""
    step_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "action_id": self.action_id,
            "action_text": self.action_text,
            "reward": self.reward,
            "next_state": self.next_state.to_dict(),
            "malicious_success": self.malicious_success,
            "benign_success": self.benign_success,
            "persistent_success": self.persistent_success,
            "trajectory_id": self.trajectory_id,
            "step_id": self.step_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transition":
        return cls(
            state=AttackState.from_dict(data.get("state", {})),
            action_id=str(data.get("action_id", "")),
            action_text=str(data.get("action_text", "")),
            reward=float(data.get("reward", 0.0)),
            next_state=AttackState.from_dict(data.get("next_state", {})),
            malicious_success=int(data.get("malicious_success", 0)),
            benign_success=int(data.get("benign_success", 0)),
            persistent_success=int(data.get("persistent_success", 0)),
            trajectory_id=str(data.get("trajectory_id", "")),
            step_id=int(data.get("step_id", 0)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ActionDecision:
    candidates: List[AttackStrategy]
    base_logits: List[float]
    advantage_clip: List[float]
    selected: AttackStrategy
    selected_index: int
    used_memory_count: int = 0
    value_estimate: float = 0.0
    q_estimates: Dict[str, Optional[float]] = field(default_factory=dict)
    raw_advantages: List[float] = field(default_factory=list)
    base_probs: List[float] = field(default_factory=list)
    advantage_probs: List[float] = field(default_factory=list)
    updated_probs: List[float] = field(default_factory=list)
