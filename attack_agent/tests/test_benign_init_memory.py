import tempfile
import unittest
from pathlib import Path

from attack_agent.build_benign_init_memory import reset_init_memory_dir


class BenignInitMemoryTests(unittest.TestCase):
    def test_existing_init_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_memory = Path(tmp) / "init"
            init_memory.mkdir()

            with self.assertRaises(FileExistsError):
                reset_init_memory_dir(init_memory, overwrite=False)

    def test_overwrite_removes_existing_init_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_memory = Path(tmp) / "init"
            init_memory.mkdir()
            (init_memory / "old.txt").write_text("old", encoding="utf-8")

            reset_init_memory_dir(init_memory, overwrite=True)

            self.assertFalse(init_memory.exists())


if __name__ == "__main__":
    unittest.main()
