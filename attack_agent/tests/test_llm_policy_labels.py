import unittest

from attack_agent.llm_policy import (
    _choice_labels,
    _complete_choice_logprobs,
    _state_prompt,
)
from attack_agent.models import AttackState


class ChoiceLabelTests(unittest.TestCase):
    def test_fourteen_candidates_receive_distinct_single_character_labels(self):
        self.assertEqual(_choice_labels(14), list("ABCDEFGHIJKLMN"))

    def test_choice_labels_reject_unsupported_strategy_space_size(self):
        with self.assertRaises(ValueError):
            _choice_labels(53)

    def test_missing_tail_label_is_conservatively_imputed(self):
        logits, missing, imputed = _complete_choice_logprobs(
            labels=["A", "B", "C"],
            token_rows=[
                {"stripped": "A", "logprob": -1.0},
                {"stripped": "C", "logprob": -3.0},
                {"stripped": "other", "logprob": -4.0},
            ],
        )

        self.assertEqual(missing, ["B"])
        self.assertEqual(imputed, -5.0)
        self.assertEqual(logits, [-1.0, -5.0, -3.0])


class StatePromptTests(unittest.TestCase):
    def test_state_prompt_includes_score_round_and_ordered_attack_history(self):
        prompt = _state_prompt(
            AttackState(
                agent="system_admin_agent",
                task="Inspect services",
                attack_tool="ProcessInjection",
                state_score=1.25,
                attack_history=["S2", "S10"],
                attack_round=3,
            ),
            attack_tool_description="Runs code inside another process.",
        )

        self.assertIn("Attack round in current episode: 3", prompt)
        self.assertIn("Observable state score: 1.25", prompt)
        self.assertIn(
            "Prior attack strategy history (oldest to newest): S2 -> S10",
            prompt,
        )

    def test_state_prompt_marks_empty_attack_history(self):
        prompt = _state_prompt(
            AttackState(
                agent="system_admin_agent",
                task="Inspect services",
                attack_tool="ProcessInjection",
                state_score=0.0,
                attack_history=[],
                attack_round=1,
            )
        )

        self.assertIn("Observable state score: 0", prompt)
        self.assertIn(
            "Prior attack strategy history (oldest to newest): none",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
