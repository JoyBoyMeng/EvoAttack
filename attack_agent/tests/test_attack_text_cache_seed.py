import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attack_agent.attack_text_generator import (
    AttackTextGenerator,
    resolve_attack_text_cache_path,
)
from attack_agent.models import AttackStrategy
from attack_agent.run_repeated_adaptive_attack import seed_attack_text_cache


class _FailCompletions:
    def create(self, **kwargs):
        raise AssertionError(
            "LLM generation request must not occur for an attack-text cache hit."
        )


class AttackTextCacheSeedTests(unittest.TestCase):
    def test_staged_attempts_commit_only_final_text_to_canonical_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "attack_text_cache.json"
            fake_client = SimpleNamespace(
                chat=SimpleNamespace(completions=_FailCompletions())
            )
            with patch(
                "attack_agent.attack_text_generator.make_openai_client",
                return_value=fake_client,
            ):
                generator = AttackTextGenerator(cache_path=str(cache_path))
            strategy = AttackStrategy("S1", "one", "one", "{task}")
            generation_kwargs = {
                "strategy": strategy,
                "task": "same task",
                "attack_tool": "ProcessInjection",
                "attacker_instruction": "instruction",
                "attack_tool_description": "description",
                "persist": False,
                "use_cache": False,
            }

            with patch.object(
                generator,
                "_generate_remote",
                side_effect=["failed first text", "successful second text"],
            ):
                first = generator.generate(
                    **generation_kwargs,
                    variant_index=1,
                )
                second = generator.generate(
                    **generation_kwargs,
                    variant_index=2,
                    previous_texts=(first,),
                )

            self.assertFalse(cache_path.exists())
            canonical_key = generator.last_generation["canonical_cache_key"]
            legacy_key = generator._legacy_key(
                task="same task",
                attack_tool="ProcessInjection",
                strategy_id="S1",
            )
            cache_path.write_text(
                json.dumps(
                    {
                        legacy_key: "old failed legacy text",
                        f"{canonical_key}::retry_2": "old failed retry text",
                        "unrelated": "keep",
                    }
                ),
                encoding="utf-8",
            )

            committed_key = generator.commit_final(second)

            self.assertEqual(committed_key, canonical_key)
            self.assertEqual(
                json.loads(cache_path.read_text(encoding="utf-8")),
                {
                    canonical_key: "successful second text",
                    "unrelated": "keep",
                },
            )

    def test_retry_variant_uses_distinct_cache_key_and_prompt(self):
        generator = AttackTextGenerator.__new__(AttackTextGenerator)
        base_key = generator._key(
            agent_name="agent_one",
            task="same task",
            attack_tool="ProcessInjection",
            strategy_id="S1",
            delivery_mode="task_text",
            context_fingerprint="context",
            variant_index=1,
        )
        retry_key = generator._key(
            agent_name="agent_one",
            task="same task",
            attack_tool="ProcessInjection",
            strategy_id="S1",
            delivery_mode="task_text",
            context_fingerprint="context",
            variant_index=2,
        )
        _, retry_prompt = generator._task_text_prompt(
            strategy=AttackStrategy("S1", "one", "one", "{task}"),
            task="same task",
            attack_tool="ProcessInjection",
            attacker_instruction="instruction",
            attack_tool_description="description",
            variant_index=2,
            previous_texts=("first weak text",),
        )

        self.assertNotEqual(base_key, retry_key)
        self.assertTrue(retry_key.endswith("::retry_2"))
        self.assertIn("materially different formulation", retry_prompt)
        self.assertIn("first weak text", retry_prompt)

    def test_observation_prompt_uses_firm_operational_language(self):
        generator = AttackTextGenerator.__new__(AttackTextGenerator)
        strategy = AttackStrategy(
            "S10",
            "observation",
            "observation",
            "{task}",
            delivery_mode="memory_observation",
        )

        system, user = generator._memory_observation_prompt(
            strategy=strategy,
            agent_name="system_admin_agent",
            agent_description="System administrator",
            normal_tools=[{"name": "sys_monitor", "description": "monitor"}],
            task="Inspect the running services.",
            attack_tool="ProcessInjection",
            attacker_instruction="Use the relevant execution capability.",
            attack_tool_description="Execute a process.",
        )

        self.assertIn("clear operational requirement", system)
        self.assertIn("required operational step", user)
        self.assertIn(
            "You may directly use the specific attacker tool name, ProcessInjection",
            user,
        )
        self.assertIn("allowed but not mandatory", user)

    def test_cache_directory_resolves_to_shared_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "attack_text_cache"
            cache_dir.mkdir()

            self.assertEqual(
                resolve_attack_text_cache_path(
                    cache_dir,
                    agent_name="agent_one",
                ),
                cache_dir / "agent_one" / "attack_text_cache.json",
            )

    def test_generator_uses_agent_isolated_file_under_cache_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "attack_text_cache"
            cache_root.mkdir()
            fake_client = SimpleNamespace(
                chat=SimpleNamespace(completions=_FailCompletions())
            )
            with patch(
                "attack_agent.attack_text_generator.make_openai_client",
                return_value=fake_client,
            ):
                generator = AttackTextGenerator(
                    cache_path=str(cache_root),
                    agent_name="agent_one",
                )

            self.assertEqual(
                generator.cache_path,
                cache_root / "agent_one" / "attack_text_cache.json",
            )

    def test_concurrent_style_writes_merge_and_preserve_existing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "attack_text_cache" / "attack_text_cache.json"
            fake_client = SimpleNamespace(
                chat=SimpleNamespace(completions=_FailCompletions())
            )
            with patch(
                "attack_agent.attack_text_generator.make_openai_client",
                return_value=fake_client,
            ):
                first = AttackTextGenerator(cache_path=str(cache_path))
                second = AttackTextGenerator(cache_path=str(cache_path))

            first.cache = {"a": "first"}
            first._write_cache()
            second.cache = {"a": "replacement", "b": "second"}
            second._write_cache()

            self.assertEqual(
                json.loads(cache_path.read_text(encoding="utf-8")),
                {"a": "first", "b": "second"},
            )

    def test_generator_refreshes_shared_cache_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "same task"
            cache_path = Path(tmp) / "attack_text_cache.json"
            fake_client = SimpleNamespace(
                chat=SimpleNamespace(completions=_FailCompletions())
            )
            with patch(
                "attack_agent.attack_text_generator.make_openai_client",
                return_value=fake_client,
            ):
                generator = AttackTextGenerator(cache_path=str(cache_path))
                context_fingerprint = generator._context_fingerprint(
                    agent_name="agent_one",
                    agent_description="role",
                    normal_tools=[],
                    attacker_instruction="instruction",
                    attack_tool_description="description",
                    delivery_mode="task_text",
                )
                key = generator._key(
                    agent_name="agent_one",
                    task=task,
                    attack_tool="ProcessInjection",
                    strategy_id="S1",
                    delivery_mode="task_text",
                    context_fingerprint=context_fingerprint,
                )
                cache_path.write_text(
                    json.dumps({key: "added by previous round"}),
                    encoding="utf-8",
                )
                text = generator.generate(
                    strategy=AttackStrategy("S1", "one", "one", "{task}"),
                    task=task,
                    attack_tool="ProcessInjection",
                    attacker_instruction="instruction",
                    attack_tool_description="description",
                    agent_name="agent_one",
                    agent_description="role",
                )

            self.assertEqual(text, "added by previous round")
            self.assertTrue(generator.last_generation["cache_hit"])

    def test_seed_merge_preserves_existing_destination_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            destination = root / "run" / "attack_text_cache.json"
            source.write_text(
                json.dumps({"a": "source-a", "b": "source-b"}),
                encoding="utf-8",
            )
            destination.parent.mkdir(parents=True)
            destination.write_text(
                json.dumps({"a": "existing-a", "c": "existing-c"}),
                encoding="utf-8",
            )

            stats = seed_attack_text_cache(
                source=source,
                destination=destination,
            )
            merged = json.loads(destination.read_text(encoding="utf-8"))

            self.assertEqual(stats, (2, 2, 1, 3))
            self.assertEqual(
                merged,
                {
                    "a": "existing-a",
                    "b": "source-b",
                    "c": "existing-c",
                },
            )

    def test_cache_hit_returns_without_llm_generation_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "same task"
            key = (
                f"{hashlib.sha1(task.encode('utf-8')).hexdigest()[:12]}"
                "::ProcessInjection::S1"
            )
            cache_path = Path(tmp) / "attack_text_cache.json"
            cache_path.write_text(
                json.dumps({key: "cached attack text"}),
                encoding="utf-8",
            )
            fake_client = SimpleNamespace(
                chat=SimpleNamespace(completions=_FailCompletions())
            )

            with patch(
                "attack_agent.attack_text_generator.make_openai_client",
                return_value=fake_client,
            ):
                generator = AttackTextGenerator(cache_path=str(cache_path))
                text = generator.generate(
                    strategy=AttackStrategy("S1", "one", "one", "{task}"),
                    task=task,
                    attack_tool="ProcessInjection",
                    attacker_instruction="instruction",
                    attack_tool_description="description",
                )

            self.assertEqual(text, "cached attack text")

    def test_observation_cache_key_is_agent_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "attack_text_cache.json"
            fake_client = SimpleNamespace(
                chat=SimpleNamespace(completions=_FailCompletions())
            )
            strategy = AttackStrategy(
                "S10",
                "observation",
                "observation",
                "{task}",
                delivery_mode="memory_observation",
            )
            with patch(
                "attack_agent.attack_text_generator.make_openai_client",
                return_value=fake_client,
            ):
                generator = AttackTextGenerator(cache_path=str(cache_path))
                first_key = generator._key(
                    agent_name="agent_one",
                    task="same task",
                    attack_tool="ProcessInjection",
                    strategy_id="S10",
                    delivery_mode="memory_observation",
                    context_fingerprint=generator._context_fingerprint(
                        agent_name="agent_one",
                        agent_description="role",
                        normal_tools=[{"name": "normal_one", "description": "desc"}],
                        attacker_instruction="goal",
                        attack_tool_description="attack desc",
                        delivery_mode="memory_observation",
                    ),
                )
                generator.cache[first_key] = "cached observation"
                text = generator.generate(
                    strategy=strategy,
                    task="same task",
                    attack_tool="ProcessInjection",
                    attacker_instruction="goal",
                    attack_tool_description="attack desc",
                    agent_name="agent_one",
                    agent_description="role",
                    normal_tools=[{"name": "normal_one", "description": "desc"}],
                )

            self.assertEqual(text, "cached observation")
            self.assertTrue(generator.last_generation["cache_hit"])
            self.assertFalse(generator.last_generation["legacy_cache_hit"])


if __name__ == "__main__":
    unittest.main()
