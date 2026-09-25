import tempfile
import unittest
from pathlib import Path

import chromadb

from attack_agent.scripts.expand_init_memory import expand_init_memory


class ExpandInitMemoryTests(unittest.TestCase):
    def test_expands_each_clean_seed_and_preserves_tool_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = chromadb.PersistentClient(path=tmp)
            collection = client.create_collection("init")
            collection.add(
                ids=["one", "two"],
                embeddings=[[0.0, 0.0], [0.0, 0.0]],
                metadatas=[
                    {"attack_tool": "ProcessInjection", "memory_sequence_ns": 1},
                    {"attack_tool": "LogAlteration", "memory_sequence_ns": 2},
                ],
            )

            final_count = expand_init_memory(
                db_path=Path(tmp),
                collection_name="init",
                copies=5,
                expected_source_count=2,
            )

            self.assertEqual(final_count, 10)
            records = collection.get(include=["metadatas"])
            metadatas = records["metadatas"]
            self.assertEqual(
                sum(metadata["attack_tool"] == "ProcessInjection" for metadata in metadatas),
                5,
            )
            self.assertEqual(
                sum(metadata["attack_tool"] == "LogAlteration" for metadata in metadatas),
                5,
            )
            copy_indices = {metadata["init_copy_index"] for metadata in metadatas}
            self.assertEqual(copy_indices, {1, 2, 3, 4, 5})
            for copy_index in range(1, 6):
                self.assertEqual(
                    sum(metadata["init_copy_index"] == copy_index for metadata in metadatas),
                    2,
                )

    def test_rejects_an_unexpected_source_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = chromadb.PersistentClient(path=tmp)
            collection = client.create_collection("init")
            collection.add(
                ids=["one"],
                embeddings=[[0.0, 0.0]],
                metadatas=[{"attack_tool": "ProcessInjection"}],
            )

            with self.assertRaisesRegex(ValueError, "Expected exactly 2"):
                expand_init_memory(
                    db_path=Path(tmp),
                    collection_name="init",
                    copies=5,
                    expected_source_count=2,
                )


if __name__ == "__main__":
    unittest.main()
