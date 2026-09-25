# Adaptive attack experiment

The default experiment runs three consecutive attacks per isolated episode, read-only B1/B2/B3 tests, then `D1-D3 -> P3 -> D4-D5 -> P5`. Policy experience and the three final attack rewards are committed only after P5. CSV output contains three B1/B2/B3 test rows, associated with A1/A2/A3 by `attack_index`, plus explicit P1 (equal to B3), P3, and P5 rows. Every test/probe outcome uses `attack_success`, and fields that do not apply to a row contain `NULL`. See [MULTI_ROUND_EXPERIMENT_REQUIREMENTS.md](MULTI_ROUND_EXPERIMENT_REQUIREMENTS.md) for the exact contract.

Target-memory retrieval keeps the embedding Top-150 candidate search and
reserves exact `(task_hash, available_tools)` matches. It never fills the
shortfall with non-exact embedding candidates. Instead, the remaining prefix is
filled from the global-recent window up to ten total memories. Global and exact
blocks are each presented oldest-to-newest, with global first and exact last.

Run commands from `ASB_testv11` with the `aa` environment and configured `attack_agent/.env`.

## Offline tests

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate aa
python -m unittest discover -s attack_agent/tests -p 'test_*.py'
```

## One training process

```bash
python -u -m attack_agent.run_adaptive_attack \
  --target_agent system_admin_agent \
  --task_num 5 \
  --attack_tool_num 40 \
  --train_trajectories 1 \
  --train_round 1 \
  --attack_rounds 3 \
  --target_mem0_path memory_db/target_agent_mem0_with_embedding_init/target_agent_mem0_system_admin_agent_init \
  --episode_memory_root memory_db/run_time_memory/system_admin_agent_v10_round01 \
  --policy_memory_path logs/attack_agent/system_admin_agent_train_v10/policy_memory.jsonl \
  --attack_text_cache_path memory_db/attack_text_cache \
  --res_file logs/attack_agent/system_admin_agent_train_v10/round_01/train_results.csv \
  --selection_mode argmax \
  --seed 0 \
  --attack_text_attempts 1 \
  --state_retrieve_top_k 400 \
  --retrieve_top_k 200 \
  --no-cross_attack_round_retrieval \
  --no-probe_write_target_memory \
  --max_new_tokens 128
```

## Repeated training rounds

```bash
python -u -m attack_agent.run_repeated_adaptive_attack \
  --rounds 10 \
  --start_round 1 \
  --version v10 \
  --target_agent system_admin_agent \
  --init_memory memory_db/target_agent_mem0_with_embedding_init/target_agent_mem0_system_admin_agent_init \
  --task_num 5 \
  --attack_tool_num 40 \
  --train_trajectories 1 \
  --attack_rounds 3 \
  --selection_mode argmax \
  --seed 0 \
  --attack_text_attempts 1 \
  --state_retrieve_top_k 400 \
  --retrieve_top_k 200 \
  --no-cross_attack_round_retrieval \
  --no-probe_write_target_memory \
  --max_new_tokens 128
```

To retrieve policy experience from every inner attack position, replace `--no-cross_attack_round_retrieval` with `--cross_attack_round_retrieval`. This does not restrict or reset outer `train_round` history.

## Frozen final evaluation

```bash
python -u -m attack_agent.run_final_policy_test \
  --target_agent system_admin_agent \
  --task_num 5 \
  --attack_tool_num 40 \
  --attack_rounds 3 \
  --init_memory memory_db/target_agent_mem0_with_embedding_init/target_agent_mem0_system_admin_agent_init \
  --target_mem0_path memory_db/run_time_memory/system_admin_agent_v10_final_source \
  --episode_memory_root memory_db/run_time_memory/system_admin_agent_v10_final_episodes \
  --policy_memory_path logs/attack_agent/system_admin_agent_train_v10/policy_memory.jsonl \
  --res_file logs/attack_agent/system_admin_agent_train_v10/final_results.csv \
  --selection_mode argmax \
  --seed 0 \
  --state_retrieve_top_k 400 \
  --retrieve_top_k 200 \
  --no-cross_attack_round_retrieval \
  --no-probe_write_target_memory \
  --max_new_tokens 128
```

Final evaluation reads but does not update `policy_memory.jsonl`.
