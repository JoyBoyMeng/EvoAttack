from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from .attack_text_generator import (
    DEFAULT_ATTACK_TEXT_CACHE_ROOT,
    resolve_attack_text_cache_path,
)
from .run_adaptive_attack import (
    add_enable_thinking_argument,
    add_probe_target_memory_write_argument,
    format_elapsed_hms,
    load_attack_agent_env,
)
from .scripts.prepare_attack_train_test_memory import resolve_init_path


ASB_ROOT = Path(__file__).resolve().parents[1]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run adaptive attack training for multiple rounds. Each round copies a fresh "
            "target train memory from the init memory, while keeping the attack policy "
            "memory and attack text cache."
        )
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of sequential training rounds.")
    parser.add_argument(
        "--start_round",
        type=int,
        default=1,
        help=(
            "First round index to run, inclusive. Use 2 with --rounds 10 to resume "
            "a completed round 1 without overwriting its artifacts."
        ),
    )
    parser.add_argument(
        "--init_memory",
        default=os.getenv("ATTACK_INIT_MEMORY_NAME", "target_agent_mem0_system_admin_agent_init_v03"),
        help="Init memory name under memory_db, or a full/relative path.",
    )
    parser.add_argument("--memory_db_dir", default="memory_db")
    parser.add_argument(
        "--runtime_memory_dir",
        default="memory_db/run_time_memory",
        help=(
            "Root for ephemeral per-episode target-memory copies. Each run and "
            "round receives an isolated subdirectory below this root."
        ),
    )
    parser.add_argument(
        "--train_memory_dir",
        default=None,
        help=(
            "Directory that stores per-round train memories. Each round is copied to "
            "<train_memory_dir>/round_NN. Defaults to "
            "<runtime_memory_dir>/<actual-run-name>/train."
        ),
    )
    parser.add_argument(
        "--test_memory_dir",
        default=None,
        help=(
            "Directory reserved for per-round test memories copied from init before "
            "testing. Defaults to "
            "<runtime_memory_dir>/<actual-run-name>/test."
        ),
    )
    parser.add_argument("--target_agent", default="system_admin_agent")
    parser.add_argument("--version", default="v01")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument(
        "--attack_text_cache_path",
        default=str(DEFAULT_ATTACK_TEXT_CACHE_ROOT),
        help=(
            "Shared attack-text cache file or directory. Every round reads and "
            "updates this cache; default: memory_db/attack_text_cache."
        ),
    )
    parser.add_argument(
        "--initial_attack_text_cache",
        default=None,
        help=(
            "Optional existing attack_text_cache.json merged into the shared cache. "
            "Existing shared-cache values are preserved."
        ),
    )
    parser.add_argument(
        "--unique_run_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If run_dir already has previous run artifacts, create a numeric-suffixed run directory.",
    )
    parser.add_argument(
        "--unique_memory_dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy target train memories into fresh per-round directories instead of overwriting a fixed destination.",
    )
    parser.add_argument(
        "--overwrite_memory",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ATTACK_ROUND_OVERWRITE_MEMORY", True),
        help="Overwrite the fixed *_4attacktrain destination when --no-unique_memory_dirs is used.",
    )
    parser.add_argument("--task_num", type=int, default=5)
    parser.add_argument("--attack_tool_num", type=int, default=40)
    parser.add_argument("--train_trajectories", type=int, default=1)
    parser.add_argument("--attack_rounds", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument(
        "--attack_text_attempts",
        type=int,
        choices=(1, 2, 3),
        default=1,
    )
    parser.add_argument(
        "--persistent_T",
        type=int,
        default=None,
        help="Deprecated. Forwarded for old commands; every completed trajectory is probed after each training run.",
    )
    parser.add_argument("--advantage_beta", type=float, default=3.0)
    parser.add_argument("--advantage_scale", type=float, default=0.5)
    parser.add_argument("--advantage_temperature", type=float, default=0.3)
    parser.add_argument("--state_retrieve_top_k", type=int, default=400)
    parser.add_argument("--retrieve_top_k", type=int, default=200)
    parser.add_argument(
        "--cross_attack_round_retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--api_retry_sleep", type=float, default=5.0)
    add_enable_thinking_argument(parser)
    add_probe_target_memory_write_argument(parser)
    return parser.parse_known_args()


def has_run_artifacts(path: Path) -> bool:
    if not path.exists():
        return False
    ignored = {"repeated_train.log"}
    return any(child.name not in ignored for child in path.iterdir())


def unique_path(path: Path) -> Path:
    if not has_run_artifacts(path):
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not has_run_artifacts(candidate):
            return candidate
    raise RuntimeError(f"Unable to find an unused run directory for {path}")


def load_attack_text_cache(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Attack-text cache must be a JSON object: {path}")
    return {str(key): str(value) for key, value in data.items()}


def seed_attack_text_cache(
    *,
    source: Path,
    destination: Path,
) -> tuple[int, int, int, int]:
    if not source.is_file():
        raise FileNotFoundError(f"Initial attack-text cache does not exist: {source}")
    source_cache = load_attack_text_cache(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f"{destination.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            existing_cache = (
                load_attack_text_cache(destination)
                if destination.is_file()
                else {}
            )
            merged_cache = dict(source_cache)
            merged_cache.update(existing_cache)
            temp_path = destination.with_name(
                f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            try:
                with temp_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        merged_cache,
                        f,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                os.replace(temp_path, destination)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    added_count = len(set(source_cache) - set(existing_cache))
    return (
        len(source_cache),
        len(existing_cache),
        added_count,
        len(merged_cache),
    )


def path_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def runtime_episode_memory_root(
    runtime_memory_dir: str,
    *,
    run_dir: Path,
    round_index: int,
) -> Path:
    root = Path(runtime_memory_dir)
    if not root.is_absolute():
        root = ASB_ROOT / root
    return (
        root
        / path_token(run_dir.name)
        / "episodes"
        / f"round_{round_index:02d}"
    )


def runtime_run_memory_root(runtime_memory_dir: str, *, run_dir: Path) -> Path:
    root = Path(runtime_memory_dir)
    if not root.is_absolute():
        root = ASB_ROOT / root
    return root / path_token(run_dir.name)


def derived_memory_paths(
    memory_db_dir: str,
    init_memory: str,
    *,
    suffix: str = "",
    train_memory_dir: str | None = None,
    test_memory_dir: str | None = None,
) -> tuple[Path, Path]:
    src = resolve_init_path(Path(memory_db_dir), init_memory)
    suffix_part = f"_{path_token(suffix)}" if suffix else ""
    train_path = (
        Path(train_memory_dir)
        if train_memory_dir
        else src.parent / f"{src.name}_4attacktrain{suffix_part}"
    )
    test_path = (
        Path(test_memory_dir)
        if test_memory_dir
        else src.parent / f"{src.name}_4attacktest{suffix_part}"
    )
    validate_memory_roots(src, train_path, test_path)
    return train_path, test_path


def validate_memory_roots(src: Path, train_root: Path, test_root: Path) -> None:
    resolved_src = src.resolve()
    resolved_train = train_root.resolve()
    resolved_test = test_root.resolve()
    if resolved_train == resolved_test:
        raise ValueError("train_memory_dir and test_memory_dir must be different directories.")
    if resolved_src in {resolved_train, resolved_test}:
        raise ValueError("init_memory must be different from train_memory_dir and test_memory_dir.")


def round_memory_paths(
    memory_db_dir: str,
    init_memory: str,
    *,
    run_name: str,
    round_index: int,
    collision_index: int = 0,
    train_memory_dir: str | None = None,
    test_memory_dir: str | None = None,
) -> tuple[Path, Path]:
    src = resolve_init_path(Path(memory_db_dir), init_memory)
    memory_run_dir = src.parent / path_token(run_name)
    train_root = Path(train_memory_dir) if train_memory_dir else memory_run_dir / "train"
    test_root = Path(test_memory_dir) if test_memory_dir else memory_run_dir / "test"
    validate_memory_roots(src, train_root, test_root)
    round_name = f"round_{round_index:02d}"
    if collision_index:
        round_name = f"{round_name}_{collision_index:02d}"
    return train_root / round_name, test_root / round_name


def unique_round_memory_paths(
    memory_db_dir: str,
    init_memory: str,
    *,
    run_name: str,
    round_index: int,
    train_memory_dir: str | None = None,
    test_memory_dir: str | None = None,
) -> tuple[Path, Path]:
    train_path, test_path = round_memory_paths(
        memory_db_dir,
        init_memory,
        run_name=run_name,
        round_index=round_index,
        train_memory_dir=train_memory_dir,
        test_memory_dir=test_memory_dir,
    )
    if not train_path.exists() and not test_path.exists():
        return train_path, test_path
    for index in range(1, 1000):
        train_path, test_path = round_memory_paths(
            memory_db_dir,
            init_memory,
            run_name=run_name,
            round_index=round_index,
            collision_index=index,
            train_memory_dir=train_memory_dir,
            test_memory_dir=test_memory_dir,
        )
        if not train_path.exists() and not test_path.exists():
            return train_path, test_path
    raise RuntimeError(
        "Unable to find unused target memory directories for "
        f"run={run_name}, round={round_index:02d}"
    )


def run_prepare_train_memory(
    args: argparse.Namespace,
    *,
    train_mem_path: Path,
    overwrite: bool,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "attack_agent.scripts.prepare_attack_train_test_memory",
        "--init_memory",
        args.init_memory,
        "--memory_db_dir",
        args.memory_db_dir,
        "--copy_mode",
        "train",
        "--train_dst",
        str(train_mem_path),
    ]
    if overwrite:
        cmd.append("--overwrite")
    print(f"[RepeatedAdaptiveAttack] prepare memory: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ASB_ROOT, check=True)


def run_training_round(
    *,
    args: argparse.Namespace,
    extra_args: List[str],
    round_index: int,
    run_dir: Path,
    train_mem_path: Path,
    attack_text_cache_path: Path,
) -> None:
    round_dir = run_dir / f"round_{round_index:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    policy_memory_path = run_dir / "policy_memory.jsonl"
    res_file = round_dir / "train_results.csv"
    log_file = round_dir / "train.log"
    episode_memory_root = runtime_episode_memory_root(
        args.runtime_memory_dir,
        run_dir=run_dir,
        round_index=round_index,
    )

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "attack_agent.run_adaptive_attack",
        "--target_agent",
        args.target_agent,
        "--task_num",
        str(args.task_num),
        "--attack_tool_num",
        str(args.attack_tool_num),
        "--train_trajectories",
        str(args.train_trajectories),
        "--train_round",
        str(round_index),
        "--attack_rounds",
        str(args.attack_rounds),
        "--max_steps",
        str(args.max_steps),
        "--attack_text_attempts",
        str(args.attack_text_attempts),
        "--advantage_beta",
        str(args.advantage_beta),
        "--advantage_scale",
        str(args.advantage_scale),
        "--advantage_temperature",
        str(args.advantage_temperature),
        "--state_retrieve_top_k",
        str(args.state_retrieve_top_k),
        "--retrieve_top_k",
        str(args.retrieve_top_k),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--target_mem0_path",
        str(train_mem_path),
        "--episode_memory_root",
        str(episode_memory_root),
        "--api_max_retries",
        str(args.api_max_retries),
        "--api_retry_sleep",
        str(args.api_retry_sleep),
        # A repeated-training child is exactly one round. Keep cumulative
        # statistics at round granularity instead of logging them after every
        # attack trajectory.
        "--no-trajectory_summary",
        "--enable_thinking" if args.enable_thinking else "--no-enable_thinking",
        (
            "--probe_write_target_memory"
            if args.probe_write_target_memory
            else "--no-probe_write_target_memory"
        ),
        (
            "--cross_attack_round_retrieval"
            if args.cross_attack_round_retrieval
            else "--no-cross_attack_round_retrieval"
        ),
        "--policy_memory_path",
        str(policy_memory_path),
        "--attack_text_cache_path",
        str(attack_text_cache_path),
        "--res_file",
        str(res_file),
        *extra_args,
    ]
    if args.persistent_T is not None:
        cmd.extend(["--persistent_T", str(args.persistent_T)])

    print(f"[RepeatedAdaptiveAttack] round {round_index} start", flush=True)
    print(f"[RepeatedAdaptiveAttack] log: {log_file}", flush=True)
    print(
        f"[RepeatedAdaptiveAttack] runtime_memory: {episode_memory_root}",
        flush=True,
    )
    with log_file.open("w", encoding="utf-8") as out:
        subprocess.run(cmd, cwd=ASB_ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)
    for source in (policy_memory_path, attack_text_cache_path):
        if not source.is_file():
            raise FileNotFoundError(f"Round artifact was not created: {source}")
        destination = round_dir / source.name
        shutil.copy2(source, destination)
        print(
            f"[RepeatedAdaptiveAttack] round {round_index} snapshot: {destination}",
            flush=True,
        )
    print(f"[RepeatedAdaptiveAttack] round {round_index} done", flush=True)


def _run_repeated_training() -> None:
    load_attack_agent_env()
    args, extra_args = parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")
    if args.start_round <= 0:
        raise ValueError("--start_round must be positive.")
    if args.start_round > args.rounds:
        raise ValueError("--start_round must not exceed --rounds.")
    if any(
        item == "--episode_memory_root"
        or item.startswith("--episode_memory_root=")
        for item in extra_args
    ):
        raise ValueError(
            "Repeated training owns --episode_memory_root. Use "
            "--runtime_memory_dir to place all temporary memories under the "
            "configured runtime root."
        )
    if any(
        item in {"--trajectory_summary", "--no-trajectory_summary"}
        or item.startswith("--trajectory_summary=")
        or item.startswith("--no-trajectory_summary=")
        for item in extra_args
    ):
        raise ValueError(
            "Repeated training fixes trajectory summaries off: statistics are "
            "logged once at the end of each round."
        )

    requested_run_dir = (
        Path(args.run_dir) if args.run_dir else Path("logs/attack_agent") / f"{args.target_agent}_train_{args.version}"
    )
    run_dir = unique_path(requested_run_dir) if args.unique_run_dir else requested_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_run_root = runtime_run_memory_root(
        args.runtime_memory_dir,
        run_dir=run_dir,
    )
    effective_train_memory_dir = (
        Path(args.train_memory_dir)
        if args.train_memory_dir
        else runtime_run_root / "train"
    )
    effective_test_memory_dir = (
        Path(args.test_memory_dir)
        if args.test_memory_dir
        else runtime_run_root / "test"
    )
    attack_text_cache_path = resolve_attack_text_cache_path(
        args.attack_text_cache_path,
        agent_name=args.target_agent,
    )
    if not attack_text_cache_path.is_absolute():
        attack_text_cache_path = ASB_ROOT / attack_text_cache_path
    attack_text_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_seed_stats = None
    if args.initial_attack_text_cache:
        source_cache_path = Path(args.initial_attack_text_cache).expanduser()
        if not source_cache_path.is_absolute():
            source_cache_path = ASB_ROOT / source_cache_path
        cache_seed_stats = seed_attack_text_cache(
            source=source_cache_path,
            destination=attack_text_cache_path,
        )

    print("[RepeatedAdaptiveAttack] configuration", flush=True)
    print(f"  rounds: {args.rounds}", flush=True)
    print(f"  start_round: {args.start_round}", flush=True)
    print(f"  requested_run_dir: {requested_run_dir}", flush=True)
    print(f"  run_dir: {run_dir}", flush=True)
    print(f"  unique_run_dir: {args.unique_run_dir}", flush=True)
    print(f"  unique_memory_dirs: {args.unique_memory_dirs}", flush=True)
    print(f"  runtime_memory_dir: {args.runtime_memory_dir}", flush=True)
    print(f"  train_memory_dir: {effective_train_memory_dir}", flush=True)
    print(f"  test_memory_dir: {effective_test_memory_dir}", flush=True)
    print(f"  enable_thinking: {args.enable_thinking}", flush=True)
    print(f"  probe_write_target_memory: {args.probe_write_target_memory}", flush=True)
    print(f"  attack_rounds: {args.attack_rounds}", flush=True)
    print(f"  state_retrieve_top_k: {args.state_retrieve_top_k}", flush=True)
    print(f"  retrieve_top_k: {args.retrieve_top_k}", flush=True)
    print(
        f"  cross_attack_round_retrieval: {args.cross_attack_round_retrieval}",
        flush=True,
    )
    print("  statistics_logging: once_per_round", flush=True)
    print(f"  policy_memory: {run_dir / 'policy_memory.jsonl'}", flush=True)
    print(f"  attack_text_cache: {attack_text_cache_path}", flush=True)
    if cache_seed_stats is not None:
        source_count, existing_count, added_count, merged_count = cache_seed_stats
        print(f"  initial_attack_text_cache: {args.initial_attack_text_cache}", flush=True)
        print(f"  cache_seed_source_count: {source_count}", flush=True)
        print(f"  cache_seed_existing_count: {existing_count}", flush=True)
        print(f"  cache_seed_added_count: {added_count}", flush=True)
        print(f"  cache_seed_merged_count: {merged_count}", flush=True)

    for round_index in range(args.start_round, args.rounds + 1):
        if args.unique_memory_dirs:
            train_mem_path, test_mem_path = unique_round_memory_paths(
                args.memory_db_dir,
                args.init_memory,
                run_name=run_dir.name,
                round_index=round_index,
                train_memory_dir=str(effective_train_memory_dir),
                test_memory_dir=str(effective_test_memory_dir),
            )
            overwrite_memory = False
        else:
            train_mem_path, test_mem_path = derived_memory_paths(
                args.memory_db_dir,
                args.init_memory,
                train_memory_dir=str(effective_train_memory_dir),
                test_memory_dir=str(effective_test_memory_dir),
            )
            overwrite_memory = args.overwrite_memory
        print(f"[RepeatedAdaptiveAttack] round {round_index} train_memory: {train_mem_path}", flush=True)
        print(
            f"[RepeatedAdaptiveAttack] round {round_index} test_memory_destination: "
            f"{test_mem_path} (copy from init before testing)",
            flush=True,
        )
        run_prepare_train_memory(
            args,
            train_mem_path=train_mem_path,
            overwrite=overwrite_memory,
        )
        run_training_round(
            args=args,
            extra_args=extra_args,
            round_index=round_index,
            run_dir=run_dir,
            train_mem_path=train_mem_path,
            attack_text_cache_path=attack_text_cache_path,
        )

    print("[RepeatedAdaptiveAttack] all rounds completed", flush=True)


def main() -> None:
    overall_started_at = time.perf_counter()
    try:
        _run_repeated_training()
    finally:
        overall_elapsed_seconds = time.perf_counter() - overall_started_at
        print("[RepeatedAdaptiveAttack] OVERALL_RUNTIME", flush=True)
        print(
            f"overall_elapsed_seconds: {round(overall_elapsed_seconds, 6)}",
            flush=True,
        )
        print(
            f"overall_elapsed_hms: {format_elapsed_hms(overall_elapsed_seconds)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
