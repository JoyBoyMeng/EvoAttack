import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attack_agent.run_final_policy_test import (
    FINAL_RESULT_HEADER,
    parse_args,
    prepare_final_test_memory,
)


class FinalTestInitMemoryTests(unittest.TestCase):
    def test_final_csv_matches_compact_attack_probe_schema(self):
        self.assertEqual(
            FINAL_RESULT_HEADER,
            [
                "agent",
                "task",
                "attack_tool",
                "trajectory_index",
                "attack_index",
                "probe_index",
                "stage",
                "action_id",
                "total_reward",
                "attack_success",
                "state_score_before",
                "state_score_after",
                "row_elapsed_seconds",
            ],
        )

    def test_multi_round_defaults_match_training(self):
        with patch(
            "sys.argv",
            [
                "run_final_policy_test",
                "--policy_memory_path",
                "policy.jsonl",
                "--res_file",
                "results.csv",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.attack_rounds, 3)
        self.assertEqual(args.attack_text_attempts, 1)
        self.assertEqual(args.state_retrieve_top_k, 400)
        self.assertEqual(args.retrieve_top_k, 200)
        self.assertFalse(args.cross_attack_round_retrieval)

    def test_init_memory_replaces_test_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_memory = root / "init"
            init_memory.mkdir()
            (init_memory / "marker.txt").write_text("clean", encoding="utf-8")
            test_memory = root / "test"
            test_memory.mkdir()
            (test_memory / "marker.txt").write_text("stale", encoding="utf-8")

            prepare_final_test_memory(
                SimpleNamespace(
                    init_memory=str(init_memory),
                    memory_db_dir=str(root),
                    target_mem0_path=str(test_memory),
                )
            )

            self.assertEqual((test_memory / "marker.txt").read_text(encoding="utf-8"), "clean")

    def test_init_and_test_memory_must_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_memory = Path(tmp) / "init"
            init_memory.mkdir()

            with self.assertRaisesRegex(ValueError, "must be different"):
                prepare_final_test_memory(
                    SimpleNamespace(
                        init_memory=str(init_memory),
                        memory_db_dir=tmp,
                        target_mem0_path=str(init_memory),
                    )
                )


if __name__ == "__main__":
    unittest.main()
