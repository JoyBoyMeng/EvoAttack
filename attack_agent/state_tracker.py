from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, List, Sequence, Tuple

from .models import AttackState, AttackStrategy, AttackUnit, Transition


POLICY_STATE_PADDING_CODE = -1
POLICY_STATE_NO_STRATEGY_CODE = 0
POLICY_HISTORY_EVENTS_KEY = "policy_history_events"


def make_policy_history_event(
    *,
    kind: str,
    strategy_id: str | None,
    success: bool,
    sequence_ns: int,
) -> Dict[str, Any]:
    return {
        "kind": str(kind),
        "strategy_id": None if strategy_id is None else str(strategy_id),
        "success": int(success),
        "sequence_ns": int(sequence_ns),
    }


class StateTracker:
    def __init__(
        self,
        history_k: int = 10,
        strategy_ids: Sequence[str] | None = None,
    ) -> None:
        self.history_k = history_k
        strategy_ids = [str(strategy_id) for strategy_id in (strategy_ids or [])]
        if len(strategy_ids) != len(set(strategy_ids)):
            raise ValueError("strategy_ids must be unique.")
        self.strategy_to_code = {
            strategy_id: index
            for index, strategy_id in enumerate(strategy_ids, start=1)
        }
        self.histories: Dict[AttackUnit, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=history_k)
        )
        self.global_histories: Dict[
            Tuple[str, str],
            Deque[Tuple[str, int, int]],
        ] = defaultdict(deque)
        self.attack_outcomes: Dict[AttackUnit, Deque[Dict[str, int | str]]] = defaultdict(
            lambda: deque(maxlen=10)
        )

    def get(self, *, agent: str, task: str, attack_tool: str, step_id: int = 0) -> AttackState:
        key = (agent, task, attack_tool)
        history = list(reversed(self.histories[key]))[: self.history_k]
        if len(history) < self.history_k:
            for historical_tool, action_id, success in reversed(
                self.global_histories[(agent, task)]
            ):
                if historical_tool == attack_tool:
                    continue
                history.append((action_id, success))
                if len(history) >= self.history_k:
                    break
        history.extend(
            [(POLICY_STATE_PADDING_CODE, 0)] * (self.history_k - len(history))
        )
        return AttackState(
            agent=agent,
            task=task,
            attack_tool=attack_tool,
            local_history=history,
            step_id=step_id,
        )

    def add_attack_history(
        self,
        *,
        agent: str,
        task: str,
        attack_tool: str,
        action_id: str | None,
        success: bool,
    ) -> None:
        strategy_code = self._strategy_code(action_id)
        self.histories[(agent, task, attack_tool)].append(
            (strategy_code, int(success))
        )
        self.global_histories[(agent, task)].append(
            (attack_tool, strategy_code, int(success))
        )

    def add_no_strategy_history(
        self,
        *,
        agent: str,
        task: str,
        attack_tool: str,
        success: bool,
    ) -> None:
        self.add_attack_history(
            agent=agent,
            task=task,
            attack_tool=attack_tool,
            action_id=None,
            success=success,
        )

    def _strategy_code(self, action_id: str | None) -> int:
        if action_id is None:
            return POLICY_STATE_NO_STRATEGY_CODE
        action_id = str(action_id)
        if self.strategy_to_code:
            if action_id not in self.strategy_to_code:
                raise ValueError(
                    f"Unknown strategy_id for policy state: {action_id}"
                )
            return self.strategy_to_code[action_id]

        # Compatibility for direct StateTracker users. Production
        # training/testing supplies the complete ordered strategy ID list.
        if action_id.startswith("S") and action_id[1:].isdigit():
            return int(action_id[1:])
        raise ValueError(
            "strategy_ids are required when action IDs are not in S<number> format."
        )

    def restore_attack_history(
        self,
        transitions: Iterable[Transition],
        *,
        history_format: str = "tool_then_global_all_results_no_score_v5",
    ) -> int:
        pending_events: List[
            Tuple[int, int, int, str, str, str, str | None, bool]
        ] = []
        for transition_index, transition in enumerate(transitions):
            state = transition.state
            if state.history_format != history_format:
                continue
            if not state.agent or not state.task or not state.attack_tool:
                continue
            raw_events = transition.metadata.get(POLICY_HISTORY_EVENTS_KEY, [])
            if not isinstance(raw_events, list):
                continue
            for event_index, event in enumerate(raw_events):
                if not isinstance(event, dict):
                    continue
                strategy_id = event.get("strategy_id")
                if strategy_id is not None:
                    strategy_id = str(strategy_id)
                try:
                    sequence_ns = int(event.get("sequence_ns", 0))
                    success = bool(int(event.get("success", 0)))
                except (TypeError, ValueError):
                    continue
                pending_events.append(
                    (
                        sequence_ns,
                        transition_index,
                        event_index,
                        state.agent,
                        state.task,
                        state.attack_tool,
                        strategy_id,
                        success,
                    )
                )

        pending_events.sort(key=lambda item: (item[0], item[1], item[2]))
        for (
            _,
            _,
            _,
            agent,
            task,
            attack_tool,
            strategy_id,
            success,
        ) in pending_events:
            self.add_attack_history(
                agent=agent,
                task=task,
                attack_tool=attack_tool,
                action_id=strategy_id,
                success=success,
            )
        return len(pending_events)

    def add_attack_outcome(
        self,
        *,
        agent: str,
        task: str,
        attack_tool: str,
        action_id: str,
        malicious_success: bool,
        benign_success: bool,
    ) -> None:
        self.attack_outcomes[(agent, task, attack_tool)].append(
            {
                "action_id": str(action_id),
                "malicious_success": int(malicious_success),
                "benign_success": int(benign_success),
            }
        )

    def recent_attack_summary(
        self,
        *,
        agent: str,
        task: str,
        attack_tool: str,
        candidates: Sequence[AttackStrategy],
        limit: int = 5,
    ) -> str:
        rows = list(self.attack_outcomes[(agent, task, attack_tool)])[-limit:]
        if not rows:
            return "none"

        label_by_action = {
            candidate.strategy_id: str(index + 1)
            for index, candidate in enumerate(candidates)
        }
        parts: List[str] = []
        for row in rows:
            action_id = str(row.get("action_id", ""))
            label = label_by_action.get(action_id, action_id)
            malicious = int(row.get("malicious_success", 0))
            benign = int(row.get("benign_success", 0))
            malicious_text = "malicious attack success" if malicious else "malicious attack fail"
            benign_text = "benign attack success" if benign else "benign attack fail"
            parts.append(f"strategy {label}: {malicious_text}, {benign_text}")

        return "; ".join(parts)


def jaccard_history(
    a: Iterable[Tuple[str | int, int]],
    b: Iterable[Tuple[str | int, int]],
) -> float:
    set_a = {
        f"{action}:{success}"
        for action, success in a
        if action != POLICY_STATE_PADDING_CODE
    }
    set_b = {
        f"{action}:{success}"
        for action, success in b
        if action != POLICY_STATE_PADDING_CODE
    }
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
