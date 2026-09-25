from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .api_retry import install_openai_api_retries
from .local_model_config import configure_target_openai_env, register_openai_compatible_target_model
from .retry_memory import FaultTolerantVectorDB
from .run_adaptive_attack import (
    add_enable_thinking_argument,
    build_target_memory,
    env_bool,
    format_elapsed_hms,
    is_api_failed,
    load_attack_agent_env,
    load_experiment_specs,
    log_event,
    max_tokens_limit_for_args,
    run_target_with_failure_guard,
)
from .target_runner import ASBTargetRunner


RESULT_HEADER = [
    "agent",
    "task_index",
    "attack_tool_index",
    "attack_tool",
    "original_success",
    "called_attack_tool",
    "api_failed",
    "row_elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean target-memory init by running one benign probe for every "
            "(task, attack-tool) spec. Memory writes are enabled but memory reads are disabled."
        )
    )
    parser.add_argument("--llm_name", default=os.getenv("ATTACK_TARGET_LLM_NAME", os.getenv("TARGET_LLM_NAME", "gpt-4o-mini")))
    parser.add_argument("--target_openai_base_url", default=os.getenv("ATTACK_TARGET_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--target_openai_api_key", default=os.getenv("ATTACK_TARGET_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
    parser.add_argument(
        "--target_openai_compatible",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ATTACK_TARGET_OPENAI_COMPATIBLE", False),
    )
    add_enable_thinking_argument(parser)
    parser.add_argument("--tasks_path", default="data/agent_task.jsonl")
    parser.add_argument("--attacker_tools_path", default="data/all_attack_tools.jsonl")
    parser.add_argument("--tools_info_path", default="data/all_normal_tools.jsonl")
    parser.add_argument("--target_agent", default="system_admin_agent")
    parser.add_argument("--task_num", type=int, default=1)
    parser.add_argument("--attack_tool_num", type=int, default=40)

    parser.add_argument("--target_mem0_path", required=True)
    parser.add_argument("--target_mem0_collection", default=None)
    parser.add_argument("--target_mem0_user_id", default=None)
    parser.add_argument("--target_mem0_top_k", type=int, default=10)
    parser.add_argument(
        "--target_mem0_use_real_embedding",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TARGET_MEM0_USE_REAL_EMBEDDING", False),
    )
    parser.add_argument("--target_mem0_infer", action="store_true", default=False)
    parser.add_argument("--mem0_llm_model", default=os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--mem0_embedding_model", default=os.getenv("MEM0_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--mem0_llm_provider", default="openai")
    parser.add_argument("--mem0_llm_base_url", default=os.getenv("MEM0_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--mem0_llm_api_key", default=os.getenv("MEM0_LLM_API_KEY", os.getenv("OPENAI_API_KEY")))
    parser.add_argument("--mem0_embedding_provider", default="openai")
    parser.add_argument("--mem0_embedding_base_url", default=os.getenv("MEM0_EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--mem0_embedding_api_key", default=os.getenv("MEM0_EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY")))

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--api_retry_sleep", type=float, default=5.0)
    parser.add_argument("--use_backend", default="None", choices=["None", "ollama", "vllm"])
    parser.add_argument("--log_mode", default="console", choices=["console", "file"])
    parser.add_argument("--res_file", required=True)
    parser.add_argument(
        "--overwrite_init_memory",
        action="store_true",
        help="Delete an existing --target_mem0_path before building this init memory.",
    )
    return parser.parse_args()


def reset_init_memory_dir(path: Path, *, overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("Refusing to use the filesystem root as target_mem0_path.")
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"target_mem0_path is not a directory: {path}")
    if not overwrite:
        raise FileExistsError(
            f"Init memory already exists: {path}. Pass --overwrite_init_memory to rebuild it."
        )
    shutil.rmtree(path)


def ensure_result_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow(RESULT_HEADER)


def append_result(path: Path, row: list[Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow(row)


def main() -> None:
    load_attack_agent_env()
    args = parse_args()
    if args.task_num != 1:
        raise ValueError("This init builder requires --task_num 1 so each init belongs to one agent and one task.")
    if args.attack_tool_num <= 0:
        raise ValueError("--attack_tool_num must be positive.")
    if args.use_backend == "None":
        args.use_backend = None
    if args.target_mem0_collection is None:
        args.target_mem0_collection = f"asb_mem0_normal_{args.target_agent}"
    if args.target_mem0_user_id is None:
        args.target_mem0_user_id = f"asb_normal_{args.target_agent}"
    args.disable_target_memory = False

    output_path = Path(args.target_mem0_path)
    reset_init_memory_dir(output_path, overwrite=args.overwrite_init_memory)
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
    install_openai_api_retries(
        max_retries=args.api_max_retries,
        retry_sleep=args.api_retry_sleep,
    )

    specs = load_experiment_specs(args)
    if not specs:
        raise ValueError("No benign init specs found.")
    ensure_result_header(Path(args.res_file))
    agent_name = specs[0]["agent_name"]
    target_memory = FaultTolerantVectorDB(build_target_memory(args, agent_name))
    target_runner = ASBTargetRunner(
        llm_name=args.llm_name,
        tools_info_path=args.tools_info_path,
        vector_db=target_memory,
        read_db=False,
        write_db=True,
        max_new_tokens=args.max_new_tokens,
        use_backend=args.use_backend,
        log_mode=args.log_mode,
        enable_thinking=args.enable_thinking,
    )
    failures = 0
    try:
        log_event(
            "BENIGN_INIT_START",
            init_memory=output_path,
            agent=agent_name,
            task_count=1,
            attack_tool_count=len(specs),
            read_target_memory=0,
            write_target_memory=1,
        )
        for index, spec in enumerate(specs, start=1):
            row_started_at = time.perf_counter()
            tool_name = str(spec["attacker_tool"]["Attacker Tool"])
            task_hash = hashlib.sha256(spec["task"].encode("utf-8")).hexdigest()[:16]
            observation = run_target_with_failure_guard(
                args=args,
                target_runner=target_runner,
                agent_path=spec["agent_path"],
                agent_name=spec["agent_name"],
                task=spec["task"],
                # `attack_tool_num` only supplies the 40 seed labels.  Do not
                # pass an attacker tool to the target agent: the init memory
                # must be produced with normal tools only.
                attacker_tool=None,
                attacked=False,
                strategy=None,
                context=[],
                write_target_memory=True,
                memory_metadata={
                    "source": "init_benign_seed",
                    "phase": "benign_bootstrap",
                    # This is retrieval provenance only. The attack tool was
                    # not exposed to the target agent during this run.
                    "attack_tool": tool_name,
                    "task_hash": task_hash,
                    "benign_seed_index": index,
                },
                call_site="init_benign_seed",
            )
            api_failed = is_api_failed(observation)
            row_elapsed_seconds = time.perf_counter() - row_started_at
            failures += int(api_failed)
            append_result(
                Path(args.res_file),
                [
                    spec["agent_name"],
                    1,
                    index,
                    tool_name,
                    int(observation.original_success),
                    int(observation.called_attack_tool),
                    int(api_failed),
                    round(row_elapsed_seconds, 6),
                ],
            )
            log_event(
                "BENIGN_INIT_STEP",
                index=f"{index}/{len(specs)}",
                attack_tool=tool_name,
                original_success=int(observation.original_success),
                called_attack_tool=int(observation.called_attack_tool),
                api_failed=int(api_failed),
                row_elapsed_seconds=round(row_elapsed_seconds, 6),
            )
    finally:
        target_runner.close()

    log_event(
        "BENIGN_INIT_END",
        init_memory=output_path,
        requested_records=len(specs),
        failed_records=failures,
    )
    if failures:
        raise RuntimeError(
            f"Benign init memory is incomplete: {failures}/{len(specs)} target calls failed. "
            "Rerun with --overwrite_init_memory."
        )


if __name__ == "__main__":
    main()
