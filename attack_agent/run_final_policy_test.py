from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, List

from .advantage_policy import AdvantagePolicy
from .api_retry import get_api_retry_stats, install_openai_api_retries
from .attack_text_generator import AttackTextGenerator, DEFAULT_ATTACK_TEXT_CACHE_ROOT
from .local_model_config import configure_target_openai_env, register_openai_compatible_target_model
from .llm_policy import LLMStrategyPolicy
from .policy_memory import JsonlPolicyMemory
from .retry_memory import FaultTolerantVectorDB, get_memory_failure_stats
from .run_adaptive_attack import (
    CSV_NULL,
    RESULT_HEADER,
    add_enable_thinking_argument,
    add_probe_target_memory_write_argument,
    build_target_memory,
    clone_clean_target_memory,
    default_episode_memory_root,
    env_bool,
    episode_memory_path,
    format_elapsed_hms,
    load_experiment_specs,
    load_attack_agent_env,
    log_event,
    max_tokens_limit_for_args,
    remove_episode_target_memory,
    run_isolated_episode,
    select_benign_distractor_tasks,
    task_for_csv,
)
from .state_tracker import StateTracker
from .strategy_space import load_strategy_space
from .target_runner import ASBTargetRunner
from .scripts.prepare_attack_train_test_memory import copy_memory, resolve_init_path


FINAL_RESULT_HEADER = list(RESULT_HEADER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final policy-memory test for ASB adaptive attack.")

    parser.add_argument("--llm_name", type=str, default=os.getenv("ATTACK_TARGET_LLM_NAME", os.getenv("TARGET_LLM_NAME", "gpt-4o-mini")))
    parser.add_argument("--policy_llm_name", type=str, default=os.getenv("POLICY_LLM_NAME", "gpt-4o-mini"))
    parser.add_argument("--target_openai_base_url", type=str, default=os.getenv("ATTACK_TARGET_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--target_openai_api_key", type=str, default=os.getenv("ATTACK_TARGET_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
    parser.add_argument(
        "--target_openai_compatible",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ATTACK_TARGET_OPENAI_COMPATIBLE", False),
        help="Route --llm_name through ASB's OpenAI-compatible GPT wrapper even if the model name is not in ASB's registry.",
    )
    parser.add_argument("--policy_openai_base_url", type=str, default=os.getenv("POLICY_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--policy_openai_api_key", type=str, default=os.getenv("POLICY_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
    add_enable_thinking_argument(parser)
    add_probe_target_memory_write_argument(parser)
    parser.add_argument("--tasks_path", type=str, default="data/agent_task.jsonl")
    parser.add_argument("--attacker_tools_path", type=str, default="data/all_attack_tools.jsonl")
    parser.add_argument("--tools_info_path", type=str, default="data/all_normal_tools.jsonl")
    parser.add_argument("--target_agent", type=str, default="system_admin_agent")
    parser.add_argument("--task_num", type=int, default=5)
    parser.add_argument("--attack_tool_num", type=int, default=40)
    parser.add_argument("--attack_rounds", type=int, default=3)
    parser.add_argument("--attack_text_attempts", type=int, choices=(1,), default=1)
    parser.add_argument("--persistent_probe_delays", default="3,5")

    parser.add_argument("--history_k", type=int, default=10)
    parser.add_argument("--advantage_beta", type=float, default=3.0)
    parser.add_argument("--advantage_scale", type=float, default=0.5)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--advantage_temperature", type=float, default=0.3)
    parser.add_argument("--base_logit_temperature", type=float, default=3.0)
    parser.add_argument(
        "--retrieve_eps",
        type=float,
        default=1.0,
        help="Deprecated compatibility parameter; history-state retrieval does not use it.",
    )
    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=200,
    )
    parser.add_argument("--state_retrieve_top_k", type=int, default=400)
    parser.add_argument(
        "--cross_attack_round_retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--selection_mode", choices=["sample", "argmax"], default="argmax")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--strategy_space_path", type=str, default=None)
    parser.add_argument("--policy_memory_path", type=str, required=True)
    parser.add_argument(
        "--attack_text_cache_path",
        type=str,
        default=str(DEFAULT_ATTACK_TEXT_CACHE_ROOT),
        help=(
            "Shared attack-text cache file or directory. Existing keys are reused; "
            "new final-test keys are written back to the same cache."
        ),
    )
    parser.add_argument("--res_file", type=str, required=True)

    parser.add_argument(
        "--init_memory",
        default=None,
        help=(
            "Optional clean init-memory path. When supplied, it is copied over "
            "--target_mem0_path before the final test starts."
        ),
    )
    parser.add_argument(
        "--memory_db_dir",
        default="memory_db",
        help="Memory DB root used when --init_memory is a directory name.",
    )
    parser.add_argument("--target_mem0_path", type=str, default=os.getenv("ATTACK_TEST_TARGET_MEM0_PATH", "memory_db/target_agent_mem0_system_admin_agent_init_v03_4attacktest"))
    parser.add_argument("--episode_memory_root", type=str, default=None)
    parser.add_argument("--target_mem0_collection", type=str, default=os.getenv("ATTACK_TARGET_MEM0_COLLECTION", "asb_mem0_normal_system_admin_agent"))
    parser.add_argument("--target_mem0_user_id", type=str, default=os.getenv("ATTACK_TARGET_MEM0_USER_ID", "asb_normal_system_admin_agent"))
    parser.add_argument(
        "--target_mem0_top_k",
        type=int,
        choices=[10],
        default=10,
        help=(
            "Target-memory prompt maximum. Retrieval keeps the embedding Top-150 "
            "exact-context search, never uses semantic-similarity fillers, and fills "
            "the prefix shortfall from global-recent memory up to 10 total records."
        ),
    )
    parser.add_argument(
        "--target_mem0_use_real_embedding",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TARGET_MEM0_USE_REAL_EMBEDDING", True),
        help=(
            "Use the configured embedding model for target-memory writes. "
            "This must remain enabled because runtime target memories use the "
            "same template-derived embeddings as init memory."
        ),
    )
    parser.add_argument("--target_mem0_infer", action="store_true", default=False)
    parser.add_argument("--disable_target_memory", action="store_true")
    parser.add_argument("--mem0_llm_model", type=str, default=os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--mem0_embedding_model", type=str, default=os.getenv("MEM0_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--mem0_llm_provider", type=str, default="openai")
    parser.add_argument("--mem0_llm_base_url", type=str, default=os.getenv("MEM0_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--mem0_llm_api_key", type=str, default=os.getenv("MEM0_LLM_API_KEY", os.getenv("OPENAI_API_KEY")))
    parser.add_argument("--mem0_embedding_provider", type=str, default="openai")
    parser.add_argument("--mem0_embedding_base_url", type=str, default=os.getenv("MEM0_EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--mem0_embedding_api_key", type=str, default=os.getenv("MEM0_EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY")))

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--api_retry_sleep", type=float, default=5.0)
    parser.add_argument("--use_backend", type=str, default="None", choices=["None", "ollama", "vllm"])
    parser.add_argument("--log_mode", type=str, default="console", choices=["console", "file"])

    return parser.parse_args()


def ensure_result_header(path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        with out.open(newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), [])
        if existing != FINAL_RESULT_HEADER:
            raise ValueError(
                f"Existing final-test result file has an incompatible header: {out}. "
                "Use a new res_file or remove the old result file before rerunning."
            )
        return
    with out.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(FINAL_RESULT_HEADER)


def append_row(path: str, row: List[Any]) -> None:
    if len(row) != len(FINAL_RESULT_HEADER):
        raise ValueError(
            f"Final-test result row has {len(row)} columns; "
            f"expected {len(FINAL_RESULT_HEADER)}."
        )
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def prepare_final_test_memory(args: argparse.Namespace) -> None:
    """Reset the final-test target memory from a clean init copy when requested."""
    if not args.init_memory:
        return
    source = resolve_init_path(Path(args.memory_db_dir), args.init_memory)
    destination = Path(args.target_mem0_path)
    if source.resolve() == destination.resolve():
        raise ValueError("init_memory and target_mem0_path must be different directories for final testing.")
    copy_memory(source, destination, overwrite=True)
    log_event(
        "FINAL_TEST_TARGET_MEMORY_PREPARED",
        init_memory=str(source),
        test_memory=str(destination),
    )


def _run_final_test() -> None:
    load_attack_agent_env()
    args = parse_args()
    log_event(
        "FINAL_TEST_CONFIGURATION",
        enable_thinking=int(args.enable_thinking),
        target_mem0_use_real_embedding=int(args.target_mem0_use_real_embedding),
        probe_write_target_memory=int(args.probe_write_target_memory),
        attack_rounds=args.attack_rounds,
        state_retrieve_top_k=args.state_retrieve_top_k,
        retrieve_top_k=args.retrieve_top_k,
        cross_attack_round_retrieval=int(args.cross_attack_round_retrieval),
    )
    if not args.target_mem0_path:
        raise ValueError("target_mem0_path is required. Set ATTACK_TEST_TARGET_MEM0_PATH in attack_agent/.env or pass --target_mem0_path.")
    prepare_final_test_memory(args)
    if args.use_backend == "None":
        args.use_backend = None
    max_tokens_limit = max_tokens_limit_for_args(args)
    os.environ["TARGET_MAX_TOKENS_LIMIT"] = str(
        max(int(os.getenv("TARGET_MAX_TOKENS_LIMIT", "0")), max_tokens_limit)
    )
    configure_target_openai_env(
        api_key=args.target_openai_api_key,
        base_url=args.target_openai_base_url,
    )
    if args.target_openai_compatible:
        register_openai_compatible_target_model(args.llm_name)
    install_openai_api_retries(max_retries=args.api_max_retries, retry_sleep=args.api_retry_sleep)
    ensure_result_header(args.res_file)

    specs = load_experiment_specs(args)
    if not specs:
        raise ValueError("No experiment specs loaded.")

    strategy_space = load_strategy_space(args.strategy_space_path)
    policy_memory = JsonlPolicyMemory(args.policy_memory_path)
    state_tracker = StateTracker(
        history_k=args.history_k,
        strategy_ids=[strategy.strategy_id for strategy in strategy_space],
    )
    restored_policy_history_events = state_tracker.restore_attack_history(
        policy_memory.load_all()
    )
    log_event(
        "FINAL_TEST_POLICY_STATE_RESTORED",
        restored_policy_history_events=restored_policy_history_events,
        history_k=args.history_k,
    )
    llm_policy = LLMStrategyPolicy(
        model=args.policy_llm_name,
        top_n=len(strategy_space),
        max_retries=1,
        retry_sleep=0.0,
        max_logprob_retries=args.api_max_retries,
        base_logit_temperature=args.base_logit_temperature,
        api_key=args.policy_openai_api_key,
        base_url=args.policy_openai_base_url,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        max_tokens_limit=max_tokens_limit,
    )
    advantage_policy = AdvantagePolicy(
        memory=policy_memory,
        beta=args.advantage_beta,
        eps=args.retrieve_eps,
        top_k=args.retrieve_top_k,
        state_top_k=args.state_retrieve_top_k,
        cross_attack_round_retrieval=args.cross_attack_round_retrieval,
        advantage_scale=args.advantage_scale,
        advantage_clip=args.advantage_clip,
        advantage_temperature=args.advantage_temperature,
        selection_mode=args.selection_mode,
        seed=args.seed,
    )
    attack_text_generator = AttackTextGenerator(
        cache_path=args.attack_text_cache_path,
        agent_name=args.target_agent,
        model=args.policy_llm_name,
        temperature=0.0,
        max_retries=1,
        retry_sleep=0.0,
        api_key=args.policy_openai_api_key,
        base_url=args.policy_openai_base_url,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        max_tokens_limit=max_tokens_limit,
    )

    if args.probe_write_target_memory:
        raise ValueError(
            "Final multi-round evaluation requires --no-probe_write_target_memory."
        )
    if args.attack_rounds <= 0:
        raise ValueError("--attack_rounds must be positive.")
    clean_target_memory_source = Path(args.target_mem0_path)
    episode_memory_root = (
        Path(args.episode_memory_root)
        if args.episode_memory_root
        else default_episode_memory_root(
            args.target_mem0_path,
            f"{args.target_agent}_final_test",
        )
    )
    if not args.disable_target_memory:
        if not clean_target_memory_source.is_dir():
            raise FileNotFoundError(
                f"Clean final-test target memory does not exist: {clean_target_memory_source}"
            )
        if episode_memory_root.exists() and any(episode_memory_root.iterdir()):
            raise FileExistsError(
                f"Final-test episode root contains artifacts: {episode_memory_root}"
            )
        episode_memory_root.mkdir(parents=True, exist_ok=True)

    target_runner = ASBTargetRunner(
        llm_name=args.llm_name,
        tools_info_path=args.tools_info_path,
        vector_db=None,
        read_db=not args.disable_target_memory,
        write_db=not args.disable_target_memory,
        max_new_tokens=args.max_new_tokens,
        use_backend=args.use_backend,
        log_mode=args.log_mode,
        enable_thinking=args.enable_thinking,
    )

    attack_results: List[Any] = []
    completed_specs = 0
    p3_success_count = 0
    p5_success_count = 0
    try:
        for index, spec in enumerate(specs, start=1):
            row_started_at = time.perf_counter()
            agent_name = spec["agent_name"]
            task = spec["task"]
            attacker_tool = spec["attacker_tool"]
            attack_tool_name = str(attacker_tool["Attacker Tool"])
            log_event(
                "FINAL_TEST_ATTACK_START",
                index=f"{index}/{len(specs)}",
                agent=agent_name,
                task=task,
                attack_tool=attack_tool_name,
                attack_rounds=args.attack_rounds,
            )
            episode_destination: Path | None = None
            target_memory: Any = None
            try:
                if not args.disable_target_memory:
                    episode_destination = episode_memory_path(
                        root=episode_memory_root,
                        spec=spec,
                        trajectory_index=1,
                        global_trajectory_index=index,
                    )
                    clone_clean_target_memory(
                        source=clean_target_memory_source,
                        destination=episode_destination,
                    )
                    target_memory = build_target_memory(
                        args,
                        agent_name,
                        path=str(episode_destination),
                    )
                    target_runner.vector_db = FaultTolerantVectorDB(target_memory)
                else:
                    target_runner.vector_db = None
                transitions = run_isolated_episode(
                    args=args,
                    spec=spec,
                    trajectory_index=1,
                    global_trajectory_index=index,
                    distractor_tasks=select_benign_distractor_tasks(
                        attack_task=task,
                        count=5,
                    ),
                    state_tracker=state_tracker,
                    strategy_space=strategy_space,
                    llm_policy=llm_policy,
                    advantage_policy=advantage_policy,
                    target_runner=target_runner,
                    policy_memory=policy_memory,
                    attack_text_generator=attack_text_generator,
                    is_test=True,
                    commit_policy=False,
                    write_results=False,
                )
                if len(transitions) != args.attack_rounds:
                    continue
                completed_specs += 1
                attack_results.extend(transitions)
                metadata = transitions[0].metadata
                p3_success_count += int(metadata["persistent_success_by_delay"]["3"])
                p5_success_count += int(metadata["persistent_success_by_delay"]["5"])
                elapsed = time.perf_counter() - row_started_at
                for transition in transitions:
                    append_row(
                        args.res_file,
                        [
                            agent_name,
                            task_for_csv(task),
                            attack_tool_name,
                            1,
                            transition.state.attack_round,
                            CSV_NULL,
                            "TEST_ATTACK",
                            transition.action_id,
                            transition.reward,
                            transition.benign_success,
                            transition.metadata.get(
                                "state_score_before",
                                transition.state.state_score,
                            ),
                            transition.metadata.get(
                                "state_score_after",
                                transition.next_state.state_score,
                            ),
                            round(
                                float(
                                    transition.metadata.get(
                                        "attack_row_elapsed_seconds",
                                        elapsed,
                                    )
                                ),
                                6,
                            ),
                        ],
                    )
                for delay in (1, 3, 5):
                    probe_success = (
                        transitions[-1].benign_success
                        if delay == 1
                        else metadata["persistent_success_by_delay"][str(delay)]
                    )
                    append_row(
                        args.res_file,
                        [
                            agent_name,
                            task_for_csv(task),
                            attack_tool_name,
                            1,
                            CSV_NULL,
                            delay,
                            "PERSISTENT_PROBE",
                            CSV_NULL,
                            CSV_NULL,
                            probe_success,
                            CSV_NULL,
                            CSV_NULL,
                            round(
                                float(
                                    transitions[-1].metadata.get(
                                        "probe1_row_elapsed_seconds",
                                        elapsed,
                                    )
                                    if delay == 1
                                    else metadata.get(
                                        "probe_elapsed_seconds_by_delay",
                                        {},
                                    ).get(str(delay), elapsed)
                                ),
                                6,
                            ),
                        ],
                    )
            finally:
                target_runner.vector_db = None
                if target_memory is not None:
                    close_memory = getattr(target_memory, "close", None)
                    if callable(close_memory):
                        close_memory()
                if episode_destination is not None and episode_destination.exists():
                    remove_episode_target_memory(episode_destination)

        total_attacks = len(attack_results)
        malicious_count = sum(row.malicious_success for row in attack_results)
        immediate_benign_count = sum(row.benign_success for row in attack_results)
        probe1_count = sum(
            row.benign_success
            for row in attack_results
            if row.state.attack_round == args.attack_rounds
        )
        memory_count_distribution = Counter(
            int(row.metadata.get("used_memory_count", 0))
            for row in attack_results
        )
        log_event(
            "FINAL_TEST_SUMMARY",
            total_specs=len(specs),
            completed_specs=completed_specs,
            total_attacks=total_attacks,
            malicious_success_count=malicious_count,
            malicious_success_rate=round(malicious_count / total_attacks, 6) if total_attacks else 0.0,
            immediate_benign_success_count=immediate_benign_count,
            immediate_benign_success_rate=round(immediate_benign_count / total_attacks, 6) if total_attacks else 0.0,
            probe1_success_count=probe1_count,
            probe1_success_rate=round(probe1_count / completed_specs, 6) if completed_specs else 0.0,
            p3_success_count=p3_success_count,
            p3_success_rate=round(p3_success_count / completed_specs, 6) if completed_specs else 0.0,
            p5_success_count=p5_success_count,
            p5_success_rate=round(p5_success_count / completed_specs, 6) if completed_specs else 0.0,
            used_memory_count_distribution=dict(memory_count_distribution),
            api_retry_stats=get_api_retry_stats(),
            memory_failure_stats=get_memory_failure_stats(),
        )
    finally:
        target_runner.close()


def run_final_test() -> None:
    overall_started_at = time.perf_counter()
    try:
        _run_final_test()
    finally:
        overall_elapsed_seconds = time.perf_counter() - overall_started_at
        log_event(
            "OVERALL_RUNTIME",
            overall_elapsed_seconds=round(overall_elapsed_seconds, 6),
            overall_elapsed_hms=format_elapsed_hms(overall_elapsed_seconds),
        )


if __name__ == "__main__":
    run_final_test()
