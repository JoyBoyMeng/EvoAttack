import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attack_agent.scripts import prepare_attack_train_test_memory


class PrepareMemoryCopyModeTests(unittest.TestCase):
    def run_prepare(self, *args: str) -> None:
        argv = ["prepare_attack_train_test_memory", *args]
        with patch.object(sys, "argv", argv):
            prepare_attack_train_test_memory.main()

    def test_train_mode_only_creates_train_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_memory = root / "init"
            init_memory.mkdir()
            (init_memory / "marker.txt").write_text("clean", encoding="utf-8")
            train_memory = root / "train"
            test_memory = root / "test"

            self.run_prepare(
                "--init_memory",
                str(init_memory),
                "--copy_mode",
                "train",
                "--train_dst",
                str(train_memory),
                "--test_dst",
                str(test_memory),
            )

            self.assertEqual((train_memory / "marker.txt").read_text(encoding="utf-8"), "clean")
            self.assertFalse(test_memory.exists())

    def test_test_mode_recopies_clean_init_without_touching_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_memory = root / "init"
            init_memory.mkdir()
            (init_memory / "marker.txt").write_text("clean", encoding="utf-8")
            train_memory = root / "train"
            train_memory.mkdir()
            (train_memory / "marker.txt").write_text("trained", encoding="utf-8")
            test_memory = root / "test"

            self.run_prepare(
                "--init_memory",
                str(init_memory),
                "--copy_mode",
                "test",
                "--train_dst",
                str(train_memory),
                "--test_dst",
                str(test_memory),
            )

            self.assertEqual((train_memory / "marker.txt").read_text(encoding="utf-8"), "trained")
            self.assertEqual((test_memory / "marker.txt").read_text(encoding="utf-8"), "clean")


if __name__ == "__main__":
    unittest.main()
