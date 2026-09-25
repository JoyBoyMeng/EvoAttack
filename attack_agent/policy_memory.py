from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import AttackState, MULTI_ROUND_HISTORY_FORMAT, Transition
from .state_tracker import POLICY_HISTORY_EVENTS_KEY, jaccard_history


class JsonlPolicyMemory:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, transition: Transition) -> None:
        self.add_many([transition])

    def add_many(self, transitions: Iterable[Transition]) -> None:
        new_rows = list(transitions)
        if not new_rows:
            return
        temp_path = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            with temp_path.open("w", encoding="utf-8") as output:
                if self.path.exists():
                    with self.path.open("r", encoding="utf-8") as existing:
                        for line in existing:
                            output.write(line)
                for transition in new_rows:
                    output.write(
                        json.dumps(transition.to_dict(), ensure_ascii=False) + "\n"
                    )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def add_trajectory_history_event(
        self,
        *,
        trajectory_id: str,
        event: Dict[str, Any],
    ) -> int:
        rows = self.load_all()
        matching_indexes = [
            index
            for index, row in enumerate(rows)
            if row.trajectory_id == trajectory_id
        ]
        if not matching_indexes:
            return 0

        target = rows[matching_indexes[-1]]
        events = target.metadata.get(POLICY_HISTORY_EVENTS_KEY)
        if not isinstance(events, list):
            events = []
            target.metadata[POLICY_HISTORY_EVENTS_KEY] = events
        events.append(dict(event))

        with self.path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        return 1

    def add_trajectory_reward_bonus(
        self,
        *,
        trajectory_id: str,
        bonus: float,
        persistent_success: bool,
    ) -> int:
        rows = self.load_all()
        updated = 0
        for row in rows:
            if row.trajectory_id == trajectory_id:
                row.reward += bonus
                row.persistent_success = int(persistent_success)
                row.metadata["persistent_reward"] = float(row.metadata.get("persistent_reward", 0.0)) + bonus
                updated += 1

        if updated:
            with self.path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        return updated

    def load_all(self) -> List[Transition]:
        if not self.path.exists():
            return []
        rows: List[Transition] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(Transition.from_dict(json.loads(line)))
                except Exception as exc:
                    print(
                        f"[AdaptiveAttack::POLICY_MEMORY_BAD_ROW] "
                        f"path={self.path} line={line_no} error={repr(exc)}",
                        flush=True,
                    )
        return rows

    def retrieve(
        self,
        *,
        state: AttackState,
        eps: float,
        top_k: int,
        same_task: bool = True,
        same_attack_tool: bool = True,
        state_top_k: int = 400,
        cross_attack_round_retrieval: bool = False,
    ) -> List[Transition]:
        # Policy experience is never shared across agents.  When both scope
        # flags are enabled, their match is a union: an experience from the
        # same task or from the same attack tool is eligible.  This lets the
        # policy transfer a tool-specific strategy across tasks without
        # discarding task-specific experience from other tools.
        _ = eps
        if state.history_format == MULTI_ROUND_HISTORY_FORMAT:
            return self._retrieve_multi_round(
                state=state,
                top_k=top_k,
                state_top_k=state_top_k,
                same_task=same_task,
                same_attack_tool=same_attack_tool,
                cross_attack_round_retrieval=cross_attack_round_retrieval,
            )

        candidates: List[tuple[float, int, Transition]] = []

        for index, item in enumerate(self.load_all()):
            m_state = item.state
            if m_state.agent != state.agent:
                continue

            task_matches = m_state.task == state.task
            attack_tool_matches = m_state.attack_tool == state.attack_tool
            if same_task and same_attack_tool:
                if not (task_matches or attack_tool_matches):
                    continue
            elif same_task and not task_matches:
                continue
            elif same_attack_tool and not attack_tool_matches:
                continue
            if m_state.history_format != state.history_format:
                continue
            score = jaccard_history(state.local_history, m_state.local_history)
            candidates.append((score, index, item))

        candidates.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
        return [item for _, _, item in candidates]

    @staticmethod
    def _scope_matches(
        memory_state: AttackState,
        query_state: AttackState,
        *,
        same_task: bool,
        same_attack_tool: bool,
    ) -> bool:
        task_matches = memory_state.task == query_state.task
        attack_tool_matches = memory_state.attack_tool == query_state.attack_tool
        if same_task and same_attack_tool:
            return task_matches or attack_tool_matches
        if same_task:
            return task_matches
        if same_attack_tool:
            return attack_tool_matches
        return True

    def _retrieve_multi_round(
        self,
        *,
        state: AttackState,
        top_k: int,
        state_top_k: int,
        same_task: bool,
        same_attack_tool: bool,
        cross_attack_round_retrieval: bool,
    ) -> List[Transition]:
        candidates: List[tuple[float, int, Transition]] = []
        for index, item in enumerate(self.load_all()):
            memory_state = item.state
            if memory_state.agent != state.agent:
                continue
            if memory_state.history_format != MULTI_ROUND_HISTORY_FORMAT:
                continue
            if not self._scope_matches(
                memory_state,
                state,
                same_task=same_task,
                same_attack_tool=same_attack_tool,
            ):
                continue
            if (
                not cross_attack_round_retrieval
                and memory_state.attack_round != state.attack_round
            ):
                continue
            state_distance = abs(memory_state.state_score - state.state_score)
            candidates.append((state_distance, index, item))

        candidates.sort(key=lambda row: (row[0], -row[1]))
        if state_top_k > 0:
            candidates = candidates[:state_top_k]

        ranked: List[tuple[int, int, int, float, int, Transition]] = []
        for state_distance, index, item in candidates:
            memory_history = item.state.attack_history
            query_history = state.attack_history
            prefix = longest_common_prefix(query_history, memory_history)
            same_position = sum(
                left == right
                for left, right in zip(query_history, memory_history)
            )
            edit_distance = sequence_edit_distance(query_history, memory_history)
            ranked.append(
                (-prefix, -same_position, edit_distance, state_distance, -index, item)
            )
        ranked.sort(key=lambda row: row[:-1])
        if top_k > 0:
            ranked = ranked[:top_k]
        return [row[-1] for row in ranked]


def longest_common_prefix(left: List[str], right: List[str]) -> int:
    length = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        length += 1
    return length


def sequence_edit_distance(left: List[str], right: List[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]
