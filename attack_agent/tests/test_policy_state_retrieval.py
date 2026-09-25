import tempfile
import unittest
from pathlib import Path

from attack_agent.advantage_policy import AdvantagePolicy, estimate_advantages
from attack_agent.models import (
    AttackState,
    AttackStrategy,
    MULTI_ROUND_HISTORY_FORMAT,
    Transition,
)
from attack_agent.policy_memory import JsonlPolicyMemory
from attack_agent.state_tracker import (
    POLICY_HISTORY_EVENTS_KEY,
    StateTracker,
    jaccard_history,
    make_policy_history_event,
)


def make_transition(
    *,
    action_id: str,
    history,
    reward: float,
    history_format: str = "tool_then_global_all_results_no_score_v5",
    malicious_success: int = 0,
    history_events=None,
    agent: str = "system_admin_agent",
    task: str = "Inspect services",
    attack_tool: str = "ProcessInjection",
) -> Transition:
    state = AttackState(
        agent=agent,
        task=task,
        attack_tool=attack_tool,
        local_history=list(history),
        history_format=history_format,
    )
    return Transition(
        state=state,
        action_id=action_id,
        action_text="attack",
        reward=reward,
        next_state=state,
        malicious_success=malicious_success,
        metadata={
            POLICY_HISTORY_EVENTS_KEY: list(history_events or []),
        },
    )


class PolicyStateRetrievalTests(unittest.TestCase):
    @staticmethod
    def _multi_round_transition(
        *,
        action_id: str,
        state_score: float,
        attack_history,
        attack_round: int,
        train_round: int,
    ) -> Transition:
        state = AttackState(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="ProcessInjection",
            history_format=MULTI_ROUND_HISTORY_FORMAT,
            state_score=state_score,
            attack_history=list(attack_history),
            attack_round=attack_round,
            train_round=train_round,
        )
        return Transition(
            state=state,
            action_id=action_id,
            action_text="attack",
            reward=1.0,
            next_state=state,
        )

    def test_multi_round_retrieval_filters_attack_round_not_train_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = JsonlPolicyMemory(str(Path(tmp) / "policy_memory.jsonl"))
            memory.add_many(
                [
                    self._multi_round_transition(
                        action_id="round1_train1",
                        state_score=0.0,
                        attack_history=[],
                        attack_round=1,
                        train_round=1,
                    ),
                    self._multi_round_transition(
                        action_id="round1_train9",
                        state_score=0.0,
                        attack_history=[],
                        attack_round=1,
                        train_round=9,
                    ),
                    self._multi_round_transition(
                        action_id="round2",
                        state_score=0.0,
                        attack_history=["S1"],
                        attack_round=2,
                        train_round=1,
                    ),
                ]
            )
            query = self._multi_round_transition(
                action_id="query",
                state_score=0.0,
                attack_history=[],
                attack_round=1,
                train_round=10,
            ).state

            same_round = memory.retrieve(
                state=query,
                eps=0.0,
                top_k=200,
                state_top_k=400,
            )
            all_rounds = memory.retrieve(
                state=query,
                eps=0.0,
                top_k=200,
                state_top_k=400,
                cross_attack_round_retrieval=True,
            )

        self.assertEqual(
            {item.action_id for item in same_round},
            {"round1_train1", "round1_train9"},
        )
        self.assertEqual(
            {item.action_id for item in all_rounds},
            {"round1_train1", "round1_train9", "round2"},
        )

    def test_multi_round_retrieval_applies_state_then_history_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = JsonlPolicyMemory(str(Path(tmp) / "policy_memory.jsonl"))
            memory.add_many(
                [
                    self._multi_round_transition(
                        action_id="prefix",
                        state_score=0.2,
                        attack_history=["S1", "S2"],
                        attack_round=3,
                        train_round=1,
                    ),
                    self._multi_round_transition(
                        action_id="position",
                        state_score=0.1,
                        attack_history=["S1", "S9"],
                        attack_round=3,
                        train_round=1,
                    ),
                    self._multi_round_transition(
                        action_id="near_but_other_history",
                        state_score=0.0,
                        attack_history=["S8", "S2"],
                        attack_round=3,
                        train_round=1,
                    ),
                ]
            )
            query = self._multi_round_transition(
                action_id="query",
                state_score=0.0,
                attack_history=["S1", "S2"],
                attack_round=3,
                train_round=2,
            ).state
            retrieved = memory.retrieve(
                state=query,
                eps=0.0,
                state_top_k=3,
                top_k=2,
            )

        self.assertEqual(
            [item.action_id for item in retrieved],
            ["prefix", "position"],
        )

    def test_retrieval_uses_same_task_or_same_attack_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = JsonlPolicyMemory(str(Path(tmp) / "policy_memory.jsonl"))
            common_history = [(1, 1)] + [(-1, 0)] * 9
            memory.add_many(
                [
                    make_transition(
                        action_id="same_task",
                        history=common_history,
                        reward=1.0,
                        attack_tool="UpdateManager",
                    ),
                    make_transition(
                        action_id="same_tool",
                        history=common_history,
                        reward=1.0,
                        task="Audit access",
                    ),
                    make_transition(
                        action_id="neither",
                        history=common_history,
                        reward=1.0,
                        task="Audit access",
                        attack_tool="UpdateManager",
                    ),
                    make_transition(
                        action_id="other_agent",
                        history=common_history,
                        reward=1.0,
                        agent="financial_analyst_agent",
                    ),
                ]
            )

            query = AttackState(
                agent="system_admin_agent",
                task="Inspect services",
                attack_tool="ProcessInjection",
                local_history=common_history,
            )
            retrieved = memory.retrieve(
                state=query,
                eps=0.0,
                top_k=1,
                same_task=True,
                same_attack_tool=True,
            )

            self.assertEqual(
                {item.action_id for item in retrieved},
                {"same_task", "same_tool"},
            )

    def test_unseen_strategy_uses_zero_q_and_normal_advantage(self):
        memories = [
            make_transition(
                action_id="S1",
                history=[(-1, 0)] * 10,
                reward=1.0,
            )
        ]
        candidates = [
            AttackStrategy("S1", "seen", "seen", "{task}"),
            AttackStrategy("S2", "unseen", "unseen", "{task}"),
        ]

        value, q_estimates, advantages = estimate_advantages(
            memories,
            candidates,
        )

        self.assertEqual(value, 1.0)
        self.assertEqual(q_estimates, {"S1": 1.0, "S2": 0.0})
        self.assertEqual(advantages, [0.0, -1.0])

    def test_all_reported_probability_vectors_are_normalized_without_memory_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = JsonlPolicyMemory(str(Path(tmp) / "policy_memory.jsonl"))
            policy = AdvantagePolicy(
                memory=memory,
                selection_mode="argmax",
            )
            candidates = [
                AttackStrategy("S1", "one", "one", "{task}"),
                AttackStrategy("S2", "two", "two", "{task}"),
                AttackStrategy("S3", "three", "three", "{task}"),
            ]
            decision = policy.decide(
                state=AttackState(
                    agent="system_admin_agent",
                    task="Inspect services",
                    attack_tool="ProcessInjection",
                    local_history=[(-1, 0)] * 10,
                ),
                candidates=candidates,
                base_logits=[2.0, 1.0, 0.0],
                use_memory=True,
            )

            for probabilities in (
                decision.base_probs,
                decision.advantage_probs,
                decision.updated_probs,
            ):
                self.assertEqual(len(probabilities), len(candidates))
                self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertEqual(decision.advantage_probs, [1 / 3] * 3)
            self.assertEqual(decision.updated_probs, decision.base_probs)

    def test_state_prioritizes_current_tool_then_global_recent_and_pads(self):
        tracker = StateTracker(
            history_k=10,
            strategy_ids=["S1", "S2", "S3"],
        )
        tracker.add_attack_history(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="ProcessInjection",
            action_id="S1",
            success=True,
        )
        tracker.add_attack_history(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="UpdateManager",
            action_id="S2",
            success=False,
        )
        tracker.add_attack_history(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="ProcessInjection",
            action_id="S3",
            success=False,
        )

        state = tracker.get(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="ProcessInjection",
        )

        self.assertEqual(
            state.local_history[:3],
            [(3, 0), (1, 1), (2, 0)],
        )
        self.assertEqual(len(state.local_history), 10)
        self.assertEqual(state.local_history[3:], [(-1, 0)] * 7)

    def test_unseen_tool_uses_global_actions_from_newest_to_oldest(self):
        tracker = StateTracker(
            history_k=10,
            strategy_ids=["S1", "S2", "S3"],
        )
        for attack_tool, action_id, success in [
            ("ProcessInjection", "S1", True),
            ("UpdateManager", "S2", False),
            ("ProcessInjection", "S3", False),
        ]:
            tracker.add_attack_history(
                agent="system_admin_agent",
                task="Inspect services",
                attack_tool=attack_tool,
                action_id=action_id,
                success=success,
            )

        state = tracker.get(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="NewAttackTool",
        )

        self.assertEqual(
            state.local_history[:3],
            [(3, 0), (2, 0), (1, 1)],
        )

    def test_all_result_events_restore_state_in_actual_sequence(self):
        tracker = StateTracker(
            history_k=10,
            strategy_ids=["S1", "S2"],
        )
        restored = tracker.restore_attack_history(
            [
                make_transition(
                    action_id="S1",
                    history=[(-1, 0)] * 10,
                    reward=1.0,
                    malicious_success=1,
                    history_events=[
                        make_policy_history_event(
                            kind="malicious",
                            strategy_id="S1",
                            success=True,
                            sequence_ns=10,
                        ),
                        make_policy_history_event(
                            kind="immediate_benign",
                            strategy_id=None,
                            success=False,
                            sequence_ns=20,
                        ),
                        make_policy_history_event(
                            kind="persistent",
                            strategy_id=None,
                            success=False,
                            sequence_ns=50,
                        ),
                    ],
                ),
                make_transition(
                    action_id="S2",
                    history=[(1, 1)] + [(-1, 0)] * 9,
                    reward=-1.0,
                    history_events=[
                        make_policy_history_event(
                            kind="malicious",
                            strategy_id="S2",
                            success=False,
                            sequence_ns=30,
                        ),
                        make_policy_history_event(
                            kind="immediate_benign",
                            strategy_id=None,
                            success=True,
                            sequence_ns=40,
                        ),
                    ],
                ),
                make_transition(
                    action_id="S9",
                    history=[("S1", 1)],
                    reward=10.0,
                    history_format="legacy_local_history",
                ),
            ]
        )

        state = tracker.get(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="ProcessInjection",
        )

        self.assertEqual(restored, 5)
        self.assertEqual(
            state.local_history[:5],
            [(0, 0), (0, 1), (2, 0), (0, 0), (1, 1)],
        )

    def test_persistent_event_is_written_and_restored_from_policy_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = JsonlPolicyMemory(str(Path(tmp) / "policy_memory.jsonl"))
            transition = make_transition(
                action_id="S1",
                history=[(-1, 0)] * 10,
                reward=-1.0,
                history_events=[
                    make_policy_history_event(
                        kind="malicious",
                        strategy_id="S1",
                        success=False,
                        sequence_ns=10,
                    ),
                    make_policy_history_event(
                        kind="immediate_benign",
                        strategy_id=None,
                        success=False,
                        sequence_ns=20,
                    ),
                ],
            )
            transition.trajectory_id = "trajectory-1"
            memory.add(transition)

            written = memory.add_trajectory_history_event(
                trajectory_id="trajectory-1",
                event=make_policy_history_event(
                    kind="persistent",
                    strategy_id=None,
                    success=True,
                    sequence_ns=30,
                ),
            )
            tracker = StateTracker(history_k=10, strategy_ids=["S1"])
            restored = tracker.restore_attack_history(memory.load_all())
            state = tracker.get(
                agent="system_admin_agent",
                task="Inspect services",
                attack_tool="ProcessInjection",
            )

            self.assertEqual(written, 1)
            self.assertEqual(restored, 3)
            self.assertEqual(
                state.local_history[:3],
                [(0, 1), (0, 0), (1, 0)],
            )

    def test_persistent_bonus_applies_even_when_malicious_attack_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = JsonlPolicyMemory(str(Path(tmp) / "policy_memory.jsonl"))
            transition = make_transition(
                action_id="S1",
                history=[(-1, 0)] * 10,
                reward=-1.0,
                malicious_success=0,
            )
            transition.trajectory_id = "trajectory-1"
            transition.metadata["persistent_reward"] = 0.0
            memory.add(transition)

            updated = memory.add_trajectory_reward_bonus(
                trajectory_id="trajectory-1",
                bonus=1.0,
                persistent_success=True,
            )
            stored = memory.load_all()[0]

            self.assertEqual(updated, 1)
            self.assertEqual(stored.malicious_success, 0)
            self.assertEqual(stored.persistent_success, 1)
            self.assertAlmostEqual(stored.metadata["persistent_reward"], 1.0)
            self.assertAlmostEqual(stored.reward, 0.0)

    def test_jaccard_ignores_padding_but_keeps_no_strategy(self):
        self.assertEqual(
            jaccard_history(
                [(1, 1), (-1, 0), (-1, 0)],
                [(1, 1), (-1, 0)],
            ),
            1.0,
        )
        self.assertEqual(
            jaccard_history(
                [(-1, 0)] * 10,
                [(-1, 0)] * 10,
            ),
            1.0,
        )
        self.assertEqual(
            jaccard_history(
                [(1, 1), (-1, 0)],
                [(1, 0), (-1, 0)],
            ),
            0.0,
        )
        self.assertEqual(
            jaccard_history(
                [(0, 0), (-1, 0)],
                [(-1, 0), (-1, 0)],
            ),
            0.0,
        )
        self.assertEqual(
            jaccard_history(
                [(0, 0), (-1, 0)],
                [(0, 0), (-1, 0)],
            ),
            1.0,
        )

    def test_complete_strategy_space_maps_to_stable_numeric_categories(self):
        tracker = StateTracker(
            history_k=10,
            strategy_ids=["custom_a", "custom_b", "custom_c"],
        )
        tracker.add_attack_history(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="ProcessInjection",
            action_id="custom_c",
            success=True,
        )

        state = tracker.get(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="ProcessInjection",
        )

        self.assertEqual(state.local_history[0], (3, 1))
        self.assertEqual(state.local_history[1:], [(-1, 0)] * 9)
        with self.assertRaisesRegex(ValueError, "Unknown strategy_id"):
            tracker.add_attack_history(
                agent="system_admin_agent",
                task="Inspect services",
                attack_tool="ProcessInjection",
                action_id="not_in_strategy_space",
                success=False,
            )

    def test_retrieve_uses_history_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = JsonlPolicyMemory(str(Path(tmp) / "policy_memory.jsonl"))
            exact = make_transition(
                action_id="S1",
                history=[(1, 1), (2, 0)] + [(-1, 0)] * 8,
                reward=1.0,
            )
            partial = make_transition(
                action_id="S2",
                history=[(1, 1)] + [(-1, 0)] * 9,
                reward=0.5,
            )
            unrelated = make_transition(
                action_id="S3",
                history=[(9, 0)] + [(-1, 0)] * 9,
                reward=-1.0,
            )
            legacy = make_transition(
                action_id="S4",
                history=[("S1", 1), ("S2", 0)],
                reward=10.0,
                history_format="legacy_local_history",
            )
            memory.add_many([unrelated, partial, exact, legacy])

            query = AttackState(
                agent="system_admin_agent",
                task="Inspect services",
                attack_tool="UpdateManager",
                local_history=[(1, 1), (2, 0)] + [(-1, 0)] * 8,
            )
            retrieved = memory.retrieve(
                state=query,
                eps=0.0,
                top_k=2,
                same_task=True,
                same_attack_tool=False,
            )

            self.assertEqual(
                [item.action_id for item in retrieved],
                ["S1", "S2", "S3"],
            )

    def test_old_policy_state_without_format_is_marked_legacy(self):
        state = AttackState.from_dict(
            {
                "agent": "system_admin_agent",
                "task": "Inspect services",
                "attack_tool": "ProcessInjection",
                "local_history": [{"action_id": "S1", "success": 1}],
            }
        )

        self.assertEqual(state.history_format, "legacy_local_history")

    def test_new_policy_state_serializes_as_ten_by_two_array(self):
        state = AttackState(
            agent="system_admin_agent",
            task="Inspect services",
            attack_tool="ProcessInjection",
            local_history=[(1, 1)] + [(-1, 0)] * 9,
        )

        serialized = state.to_dict()

        self.assertEqual(len(serialized["local_history"]), 10)
        self.assertTrue(all(len(row) == 2 for row in serialized["local_history"]))
        self.assertEqual(serialized["local_history"][0], [1, 1])
        self.assertEqual(serialized["local_history"][-1], [-1, 0])
        restored = AttackState.from_dict(serialized)
        self.assertEqual(restored.local_history[0], (1, 1))
        self.assertEqual(restored.local_history[-1], (-1, 0))


if __name__ == "__main__":
    unittest.main()
