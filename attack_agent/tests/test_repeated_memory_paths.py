import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attack_agent import run_repeated_adaptive_attack
from attack_agent.run_repeated_adaptive_attack import (
    round_memory_paths,
    run_prepare_train_memory,
    unique_round_memory_paths,
)


class RepeatedMemoryPathTests(unittest.TestCase):
    def test_start_round_is_parsed_for_resume(self):
        with patch(
            "sys.argv",
            ["run_repeated_adaptive_attack", "--rounds", "10", "--start_round", "2"],
        ):
            args, extra_args = run_repeated_adaptive_attack.parse_args()

        self.assertEqual(args.rounds, 10)
        self.assertEqual(args.start_round, 2)
        self.assertEqual(args.attack_text_attempts, 1)
        self.assertEqual(args.max_new_tokens, 128)
        self.assertEqual(args.attack_rounds, 3)
        self.assertEqual(args.state_retrieve_top_k, 400)
        self.assertEqual(args.retrieve_top_k, 200)
        self.assertFalse(args.cross_attack_round_retrieval)
        self.assertEqual(
            args.attack_text_cache_path,
            "memory_db/attack_text_cache",
        )
        self.assertEqual(extra_args, [])

    def test_round_number_is_a_directory_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_db = Path(tmp) / "memory_db"
            init_memory = memory_db / "target_agent_mem0_init_v07"
            init_memory.mkdir(parents=True)

            train_path, test_path = round_memory_paths(
                str(memory_db),
                init_memory.name,
                run_name="system_admin_agent_train_v8",
                round_index=2,
            )

            expected_round = (
                memory_db
                / "system_admin_agent_train_v8"
            )
            self.assertEqual(train_path, expected_round / "train" / "round_02")
            self.assertEqual(test_path, expected_round / "test" / "round_02")

    def test_existing_round_uses_non_overwriting_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_db = Path(tmp) / "memory_db"
            init_memory = memory_db / "target_agent_mem0_init_v07"
            init_memory.mkdir(parents=True)
            occupied_train, _ = round_memory_paths(
                str(memory_db),
                init_memory.name,
                run_name="system_admin_agent_train_v8",
                round_index=1,
            )
            occupied_train.mkdir(parents=True)

            train_path, test_path = unique_round_memory_paths(
                str(memory_db),
                init_memory.name,
                run_name="system_admin_agent_train_v8",
                round_index=1,
            )

            self.assertEqual(train_path.parent.name, "train")
            self.assertEqual(test_path.parent.name, "test")
            self.assertEqual(train_path.name, "round_01_01")
            self.assertEqual(test_path.name, "round_01_01")

    def test_explicit_train_and_test_memory_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_db = root / "memory_db"
            init_memory = memory_db / "target_agent_mem0_init_v07"
            init_memory.mkdir(parents=True)
            train_root = root / "target_memories" / "train"
            test_root = root / "target_memories" / "test"

            train_path, test_path = round_memory_paths(
                str(memory_db),
                init_memory.name,
                run_name="ignored_for_explicit_roots",
                round_index=3,
                train_memory_dir=str(train_root),
                test_memory_dir=str(test_root),
            )

            self.assertEqual(train_path, train_root / "round_03")
            self.assertEqual(test_path, test_root / "round_03")

    def test_train_and_test_memory_directories_must_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_db = root / "memory_db"
            init_memory = memory_db / "target_agent_mem0_init_v07"
            init_memory.mkdir(parents=True)
            same_root = root / "target_memories"

            with self.assertRaisesRegex(ValueError, "must be different"):
                round_memory_paths(
                    str(memory_db),
                    init_memory.name,
                    run_name="run",
                    round_index=1,
                    train_memory_dir=str(same_root),
                    test_memory_dir=str(same_root),
                )

    def test_training_prepare_does_not_create_test_memory(self):
        args = SimpleNamespace(
            init_memory="target_agent_mem0_init_v07",
            memory_db_dir="memory_db",
        )
        with patch(
            "attack_agent.run_repeated_adaptive_attack.subprocess.run"
        ) as run:
            run_prepare_train_memory(
                args,
                train_mem_path=Path("memory_db/run/train/round_01"),
                overwrite=False,
            )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--copy_mode") + 1], "train")
        self.assertIn("--train_dst", command)
        self.assertNotIn("--test_dst", command)


if __name__ == "__main__":
    unittest.main()
