# Adaptive attack experiment

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

## Repeated training rounds 10

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
