from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

import pandas as pd

from .advantage_policy import AdvantagePolicy
from .agent_context import load_agent_generation_context
from .api_retry import install_openai_api_retries
from .attack_text_generator import AttackTextGenerator, DEFAULT_ATTACK_TEXT_CACHE_ROOT
from .evaluator import malicious_step_reward_components, persistent_probe_reward
from .local_model_config import configure_target_openai_env, register_openai_compatible_target_model
from .llm_policy import LLMStrategyPolicy
from .mem0_builder import build_configurable_mem0_adapter
from .models import (
    AttackState,
    AttackStrategy,
    MULTI_ROUND_HISTORY_FORMAT,
    TargetObservation,
    Transition,
)
from .policy_memory import JsonlPolicyMemory
from .retry_memory import FaultTolerantVectorDB
from .state_tracker import (
    POLICY_HISTORY_EVENTS_KEY,
    StateTracker,
    make_policy_history_event,
)
from .strategy_space import load_strategy_space, write_default_strategy_space
from .target_runner import ASBTargetRunner


ASB_ROOT = Path(__file__).resolve().parents[1]
ATTACK_AGENT_ROOT = Path(__file__).resolve().parent
RESULT_HEADER = [
    "agent",
    "task",
    "attack_tool",
    "trajectory_index",
    "attack_index",
    "probe_index",
    "stage",
    "action_id",
    "total_reward",
    "attack_success",
    "state_score_before",
    "state_score_after",
    "row_elapsed_seconds",
]
CSV_NULL = "NULL"
DISTRACTOR_MODE = "init_seed_replay"


def format_elapsed_hms(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(round(elapsed_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def remove_episode_target_memory(path: Path, *, attempts: int = 5) -> None:
    """Remove an ephemeral Chroma directory, tolerating short shutdown races."""
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(0.1 * attempt)


def task_for_csv(task: str) -> str:
    text = str(task)
    if len(text) <= 20:
        return text
    return text[:20] + "..."


def max_tokens_limit_for_args(args: argparse.Namespace) -> int:
    return max(int(args.max_new_tokens), int(os.getenv("ATTACK_MAX_TOKENS_LIMIT", "32768")))


def load_attack_agent_env() -> None:
    load_dotenv(ASB_ROOT / ".env")
    load_dotenv(ATTACK_AGENT_ROOT / ".env", override=True)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def default_enable_thinking() -> bool:
    if os.getenv("ENABLE_THINKING") is not None:
        return env_bool("ENABLE_THINKING")
    for legacy_name in (
        "POLICY_ENABLE_THINKING",
        "POLICY_LOGPROB_ENABLE_THINKING",
        "TARGET_ENABLE_THINKING",
    ):
        if os.getenv(legacy_name) is not None:
            return env_bool(legacy_name)
    return False


def add_enable_thinking_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=default_enable_thinking(),
        help=(
            "Enable or disable Qwen-style thinking for every LLM in this run: "
            "attack policy, attack text generation, target agent, and Mem0 inference."
        ),
    )
    for legacy_option in (
        "--policy_enable_thinking",
        "--policy_logprob_enable_thinking",
    ):
        parser.add_argument(
            legacy_option,
            dest="enable_thinking",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )


def add_probe_target_memory_write_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--probe_write_target_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow immediate benign and persistent probes to write target memory. "
            "Default false; probes still read target memory, and malicious interactions "
            "keep their normal write behavior."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive policy-memory attack agent for ASB.")

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
    parser.add_argument("--task_num", type=int, default=1)
    parser.add_argument("--attack_tool_num", type=int, default=1)
    parser.add_argument(
        "--attack_channel",
        choices=["task_text", "observation"],
        default="task_text",
        help=(
            "Malicious channel for the poisoning episode: task_text appends the "
            "generated attack text to the task; observation injects it into an "
            "untrusted normal-tool observation."
        ),
    )

    parser.add_argument("--train_trajectories", type=int, default=5)
    parser.add_argument(
        "--train_round",
        type=int,
        default=1,
        help="Outer repeated-training round stored as metadata; retrieval never filters on it.",
    )
    parser.add_argument(
        "--attack_rounds",
        type=int,
        default=3,
        help="Number of consecutive attack/benign-test pairs in each episode.",
    )
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument(
        "--attack_text_attempts",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help=(
            "Maximum text-quality attempts per training attack, including the initial "
            "attempt. The main experiment defaults to one attempt; values 2 or 3 enable "
            "an explicit single-attack best-of-k control. Failed S1-S9 malicious attacks and failed "
            "S10-S14 P0 probes are reset to clean init memory before a new text variant "
            "is tried."
        ),
    )
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
        help="Final maximum after attack-history sequence ranking.",
    )
    parser.add_argument(
        "--state_retrieve_top_k",
        type=int,
        default=400,
        help="Maximum retained after state-score distance ranking.",
    )
    parser.add_argument(
        "--cross_attack_round_retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retrieve across attack positions; default keeps only the same attack_round.",
    )
    parser.add_argument("--selection_mode", choices=["sample", "argmax"], default="argmax")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trajectory_summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log cumulative summary after each trajectory. Default false; final per-run summary is always logged.",
    )

    parser.add_argument("--strategy_space_path", type=str, default=None)
    parser.add_argument("--write_default_strategy_space", type=str, default=None)

    parser.add_argument("--policy_memory_path", type=str, default="logs/adaptive_attack/policy_memory.jsonl")
    parser.add_argument(
        "--attack_text_cache_path",
        type=str,
        default=str(DEFAULT_ATTACK_TEXT_CACHE_ROOT),
        help=(
            "Shared attack-text cache file or directory. Existing keys are reused; "
            "new keys are written back to the same cache."
        ),
    )
    parser.add_argument("--res_file", type=str, default="logs/adaptive_attack/results.csv")

    parser.add_argument("--target_mem0_path", type=str, default=os.getenv("ATTACK_TRAIN_TARGET_MEM0_PATH", "memory_db/target_agent_mem0_system_admin_agent_init_v03_4attacktrain"))
    parser.add_argument(
        "--episode_memory_root",
        type=str,
        default=None,
        help=(
            "Directory for isolated per-(task, tool, trajectory) target-memory copies. "
            "Defaults to a source-specific directory under "
            "memory_db/run_time_memory and is never overwritten."
        ),
    )
    parser.add_argument(
        "--persistent_probe_delays",
        type=str,
        default="3,5",
        help=(
            "Comma-separated cumulative counts of same-task benign init-memory "
            "replays before each read-only persistent probe. Default: 3,5."
        ),
    )
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
    parser.add_argument(
        "--target_mem0_infer",
        action="store_true",
        default=False,
        help="Enable Mem0 inference when writing target memory. Default false for deterministic writes.",
    )
    parser.add_argument("--disable_target_memory", action="store_true")
    parser.add_argument("--mem0_llm_model", type=str, default=os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--mem0_embedding_model", type=str, default=os.getenv("MEM0_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--mem0_llm_provider", type=str, default="openai")
    parser.add_argument("--mem0_llm_base_url", type=str, default=os.getenv("MEM0_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--mem0_llm_api_key", type=str, default=os.getenv("MEM0_LLM_API_KEY", os.getenv("OPENAI_API_KEY")))
    parser.add_argument("--mem0_embedding_provider", type=str, default="openai")
    parser.add_argument("--mem0_embedding_base_url", type=str, default=os.getenv("MEM0_EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--mem0_embedding_api_key", type=str, default=os.getenv("MEM0_EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY")))

    parser.add_argument(
        "--persistent_T",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility parameter. Persistent probes now run within each "
            "isolated episode after --persistent_probe_delays benign distractors."
        ),
    )
    parser.add_argument(
        "--continue_after_malicious_success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compatibility option when --max_steps is greater than 1; the default training flow uses one attack step.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--api_max_retries",
        type=int,
        default=3,
        help="Retry each OpenAI API call this many times before falling back.",
    )
    parser.add_argument(
        "--api_retry_sleep",
        type=float,
        default=5.0,
        help="Seconds to sleep between API retry attempts.",
    )
    parser.add_argument("--use_backend", type=str, default="None", choices=["None", "ollama", "vllm"])
    parser.add_argument("--log_mode", type=str, default="console", choices=["console", "file"])

    return parser.parse_args()


def load_experiment_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    tasks_df = pd.read_json(args.tasks_path, lines=True)
    tools_df = pd.read_json(args.attacker_tools_path, lines=True)

    tasks_df = tasks_df[tasks_df["agent_name"] == args.target_agent]
    if len(tasks_df) == 0:
        raise ValueError(f"Cannot find target_agent={args.target_agent} in {args.tasks_path}")

    specs: List[Dict[str, Any]] = []
    for _, row in tasks_df.iterrows():
        agent_name = str(row["agent_name"])
        agent_path = str(row["agent_path"])
        tasks = list(row["tasks"])[: args.task_num]
        attacker_tools = tools_df[tools_df["Corresponding Agent"] == agent_name].head(args.attack_tool_num)
        for task in tasks:
            for _, tool in attacker_tools.iterrows():
                specs.append(
                    {
                        "agent_name": agent_name,
                        "agent_path": agent_path,
                        "task": str(task),
                        "attacker_tool": dict(tool.to_dict()),
                    }
                )
    return specs


def build_target_memory(
    args: argparse.Namespace,
    agent_name: str,
    *,
    path: Optional[str] = None,
) -> Any:
    if args.disable_target_memory:
        return None
    return build_configurable_mem0_adapter(
        path=path or args.target_mem0_path,
        collection_name=args.target_mem0_collection,
        user_id=args.target_mem0_user_id,
        agent_id=agent_name,
        top_k=args.target_mem0_top_k,
        infer=args.target_mem0_infer,
        llm_model=args.mem0_llm_model,
        embedding_model=args.mem0_embedding_model,
        namespace="adaptive_target_normal",
        llm_provider=args.mem0_llm_provider,
        llm_api_key=args.mem0_llm_api_key,
        llm_base_url=args.mem0_llm_base_url,
        embedding_provider=args.mem0_embedding_provider,
        embedding_api_key=args.mem0_embedding_api_key,
        embedding_base_url=args.mem0_embedding_base_url,
        enable_thinking=args.enable_thinking,
        retrieval_mode="exact_then_global",
        use_real_embedding=args.target_mem0_use_real_embedding,
        write_embedding_mode="template",
    )


def parse_persistent_probe_delays(value: str) -> List[int]:
    try:
        delays = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            "--persistent_probe_delays must be a comma-separated list of positive integers."
        ) from exc
    if not delays or any(delay <= 0 for delay in delays) or delays != sorted(set(delays)):
        raise ValueError(
            "--persistent_probe_delays must contain unique positive integers in ascending order."
        )
    return delays


def default_episode_memory_root(
    target_mem0_path: str,
    target_agent: str = "unknown_agent",
) -> Path:
    source = Path(target_mem0_path)
    source_hash = hashlib.sha256(
        str(source.resolve()).encode("utf-8")
    ).hexdigest()[:10]
    agent_slug = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(target_agent)
    ).strip("_") or "unknown_agent"
    source_slug = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in source.name
    ).strip("_") or "target_memory"
    return ASB_ROOT / "memory_db" / "run_time_memory" / agent_slug / (
        f"{source_slug}_{source_hash}"
    )


def episode_memory_path(
    *,
    root: Path,
    spec: Dict[str, Any],
    trajectory_index: int,
    global_trajectory_index: int,
) -> Path:
    task_hash = hashlib.sha256(str(spec["task"]).encode("utf-8")).hexdigest()[:12]
    tool_slug = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(spec["attacker_tool"]["Attacker Tool"])
    ).strip("_") or "unnamed_tool"
    return root / (
        f"trajectory_{trajectory_index:02d}_spec_{global_trajectory_index:04d}_"
        f"{task_hash}_{tool_slug}"
    )


def clone_clean_target_memory(*, source: Path, destination: Path) -> None:
    """Create an episode-private target-memory copy without modifying the source."""
    if not source.is_dir():
        raise FileNotFoundError(f"Clean target-memory source does not exist: {source}")
    if destination.exists():
        raise FileExistsError(
            f"Episode target-memory destination already exists: {destination}. "
            "Use a new run/episode root; existing episode artifacts are preserved."
        )
    if source == destination or source in destination.parents:
        raise ValueError(
            "Episode target-memory destination must not be the clean source or a child of it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def select_benign_distractor_tasks(
    *,
    attack_task: str,
    count: int,
) -> List[str]:
    """Return the same-task labels used by the distractor replay schedule.

    Each replay clones the matching benign init seed, whose target-agent run
    used this original task without memory reads, attack text, or attacker
    tools. Retaining the task hash makes D1--D5 eligible for the same exact
    context as the malicious trace at P3/P5.
    """
    if count < 1:
        raise ValueError("count must be positive")
    return [attack_task] * count


def attack_text_retry_trigger(strategy: AttackStrategy) -> str:
    """Return the failed stage that requests a new text for this strategy."""
    strategy_id = str(strategy.strategy_id).strip().upper()
    if strategy_id.startswith("S"):
        try:
            strategy_number = int(strategy_id[1:])
        except ValueError:
            strategy_number = 0
        if 1 <= strategy_number <= 9:
            return "malicious_attack_failed"
        if 10 <= strategy_number <= 14:
            return "probe1_failed"
    return (
        "probe1_failed"
        if strategy.delivery_mode == "memory_observation"
        else "malicious_attack_failed"
    )


def ensure_result_header(path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        with out.open(newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), [])
        if existing != RESULT_HEADER:
            raise ValueError(
                f"Existing result file has an incompatible header: {out}. "
                "Use a new run_dir/res_file or remove the old result file before rerunning."
            )
        return
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_HEADER)


def append_result(
    path: str,
    transition: Transition,
    *,
    is_test: bool,
    decision: Any,
    trajectory_index: int,
    row_elapsed_seconds: float,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [
            transition.state.agent,
            task_for_csv(transition.state.task),
            transition.state.attack_tool,
            trajectory_index,
            transition.state.attack_round,
            CSV_NULL,
            "TEST_ATTACK" if is_test else "TRAIN_ATTACK",
            transition.action_id,
            transition.reward,
            transition.benign_success,
            transition.metadata.get("state_score_before", transition.state.state_score),
            transition.metadata.get("state_score_after", transition.next_state.state_score),
            round(row_elapsed_seconds, 6),
        ]
        if len(row) != len(RESULT_HEADER):
            raise ValueError(
                f"Training result row has {len(row)} columns; expected {len(RESULT_HEADER)}."
            )
        writer.writerow(row)


def update_result_rows_for_persistent_bonus(
    path: str,
    *,
    pending: Dict[str, Any],
    bonus: float,
    persistent_success: bool,
) -> int:
    if bonus == 0 or not Path(path).exists():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows or not fieldnames:
        return 0

    spec = pending["spec"]
    agent = spec["agent_name"]
    task = task_for_csv(spec["task"])
    attack_tool = spec["attacker_tool"]["Attacker Tool"]
    trajectory_index = str(pending.get("trajectory_index", pending["global_trajectory_index"]))
    updated = 0
    for row in rows:
        if row.get("agent") != agent:
            continue
        if row.get("task") != task:
            continue
        if row.get("attack_tool") != attack_tool:
            continue
        if row.get("trajectory_index") != trajectory_index:
            continue
        if row.get("stage") != "TRAIN_ATTACK":
            continue
        row["total_reward"] = str(float(row.get("total_reward") or 0.0) + bonus)
        updated += 1

    if updated:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return updated


def log_event(title: str, **fields: Any) -> None:
    print("\n" + "=" * 90, flush=True)
    print(f"[AdaptiveAttack::{title}]", flush=True)
    for key, value in fields.items():
        print(f"{key}: {value}", flush=True)
    print("=" * 90, flush=True)


def is_api_failed(obs: TargetObservation) -> bool:
    return bool(obs.raw_result.get("api_failed"))


def failed_observation(
    *,
    agent_name: str,
    task: str,
    attack_tool_name: str,
    attacked: bool,
    error: str,
    attempts: int,
) -> TargetObservation:
    return TargetObservation(
        agent=agent_name,
        task=task,
        attack_tool=attack_tool_name,
        attacked=attacked,
        called_attack_tool=False,
        attack_success=False,
        original_success=False,
        messages=[],
        raw_result={
            "api_failed": True,
            "error": error,
            "attempts": attempts,
        },
    )


def run_target_with_failure_guard(
    *,
    args: argparse.Namespace,
    target_runner: ASBTargetRunner,
    agent_path: str,
    agent_name: str,
    task: str,
    attacker_tool: Optional[Dict[str, Any]],
    attacked: bool,
    strategy: Optional[AttackStrategy],
    attack_text: Optional[str] = None,
    context: Optional[List[str]] = None,
    expected_malicious_trajectory_id: Optional[str] = None,
    read_target_memory: Optional[bool] = None,
    write_target_memory: Optional[bool] = None,
    memory_metadata: Optional[Dict[str, Any]] = None,
    call_site: str,
    attack_channel: Optional[str] = None,
) -> TargetObservation:
    attack_tool_name = str(attacker_tool["Attacker Tool"]) if attacker_tool else ""
    try:
        return target_runner.run(
            agent_path=agent_path,
            agent_name=agent_name,
            task=task,
            attacker_tool=attacker_tool,
            attacked=attacked,
            strategy=strategy,
            attack_text=attack_text,
            context=context or [],
            expected_malicious_trajectory_id=expected_malicious_trajectory_id,
            read_target_memory=read_target_memory,
            write_target_memory=write_target_memory,
            memory_metadata=memory_metadata,
            attack_channel=attack_channel,
        )
    except Exception as exc:
        last_error = repr(exc)

    log_event(
        "TARGET_RUN_FAILED",
        call_site=call_site,
        attempts=1,
        agent=agent_name,
        task=task,
        attack_tool=attack_tool_name,
        attacked=int(attacked),
        error=last_error,
    )
    return failed_observation(
        agent_name=agent_name,
        task=task,
        attack_tool_name=attack_tool_name,
        attacked=attacked,
        error=last_error,
        attempts=1,
    )


def append_persistent_probe_result(
    path: str,
    *,
    pending: Dict[str, Any],
    current_global_trajectory: int,
    persistent_success: bool,
    malicious_memory_retrieved: Optional[bool],
    updated_rows: int,
    row_elapsed_seconds: float,
) -> None:
    spec = pending["spec"]
    probe_index = current_global_trajectory - int(pending["global_trajectory_index"])
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [
            spec["agent_name"],
            task_for_csv(spec["task"]),
            spec["attacker_tool"]["Attacker Tool"],
            pending.get("trajectory_index", pending["global_trajectory_index"]),
            CSV_NULL,
            probe_index,
            "PERSISTENT_PROBE",
            CSV_NULL,
            CSV_NULL,
            int(persistent_success),
            CSV_NULL,
            CSV_NULL,
            round(row_elapsed_seconds, 6),
        ]
        if len(row) != len(RESULT_HEADER):
            raise ValueError(
                f"Persistent result row has {len(row)} columns; expected {len(RESULT_HEADER)}."
            )
        writer.writerow(row)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_final_summary(
    *,
    args: argparse.Namespace,
    specs: List[Dict[str, Any]],
    global_trajectory_index: int,
    policy_memory: JsonlPolicyMemory,
    pending_persistent_count: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, str]] = []
    if Path(args.res_file).exists():
        with open(args.res_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    normal_rows = [
        row
        for row in rows
        if row.get("stage") in {"TRAIN_ATTACK", "TEST_ATTACK"}
    ]
    probe_rows = [row for row in rows if row.get("stage") == "PERSISTENT_PROBE"]
    probe1_rows = [row for row in probe_rows if row.get("probe_index") == "1"]
    persistent_rows = [
        row for row in probe_rows
        if row.get("probe_index") in {"3", "5"}
    ]

    malicious_count = len(normal_rows)
    # P1 is written as its own probe row, so benign metrics no longer depend on
    # any attack-row diagnostic field.
    benign_count = len(probe1_rows)
    benign_success_count = sum(1 for row in probe1_rows if row.get("attack_success") == "1")
    persistent_count = len(persistent_rows)
    persistent_success_count = sum(1 for row in persistent_rows if row.get("attack_success") == "1")
    normal_keys = {
        (row.get("task", ""), row.get("attack_tool", ""))
        for row in normal_rows
        if row.get("task") or row.get("attack_tool")
    }
    persistent_keys = {
        (row.get("task", ""), row.get("attack_tool", ""))
        for row in persistent_rows
        if row.get("task") or row.get("attack_tool")
    }
    task_attack_tool_keys = normal_keys | persistent_keys
    task_attack_tool_count = len(task_attack_tool_keys)
    benign_success_keys = {
        (row.get("task", ""), row.get("attack_tool", ""))
        for row in probe1_rows
        if row.get("attack_success") == "1"
    }
    persistent_success_keys = {
        (row.get("task", ""), row.get("attack_tool", ""))
        for row in persistent_rows
        if row.get("attack_success") == "1"
    }

    summary: Dict[str, Any] = {
        "malicious_attack_count": malicious_count,
        "benign_attack_count": benign_count,
        "benign_attack_success_count": benign_success_count,
        "benign_attack_success_rate": round(_safe_rate(benign_success_count, benign_count), 6),
        "persistent_attack_count": persistent_count,
        "persistent_attack_success_count": persistent_success_count,
        "persistent_attack_success_rate": round(_safe_rate(persistent_success_count, persistent_count), 6),
        "task_attack_tool_count": task_attack_tool_count,
        "task_attack_tool_benign_success_count": len(benign_success_keys),
        "task_attack_tool_benign_success_rate": round(
            _safe_rate(len(benign_success_keys), task_attack_tool_count), 6
        ),
        "task_attack_tool_persistent_success_count": len(persistent_success_keys),
        "task_attack_tool_persistent_success_rate": round(
            _safe_rate(len(persistent_success_keys), task_attack_tool_count), 6
        ),
    }
    delays = sorted(
        {
            str(row.get("probe_index", "")).strip()
            for row in probe_rows
            if str(row.get("probe_index", "")).strip() not in {"", CSV_NULL}
        },
        key=int,
    )
    for delay in delays:
        delay_rows = [
            row for row in probe_rows
            if str(row.get("probe_index", "")).strip() == delay
        ]
        delay_success_count = sum(
            1 for row in delay_rows if row.get("attack_success") == "1"
        )
        summary[f"persistent_{delay}_attack_count"] = len(delay_rows)
        summary[f"persistent_{delay}_attack_success_count"] = delay_success_count
        summary[f"persistent_{delay}_attack_success_rate"] = round(
            _safe_rate(delay_success_count, len(delay_rows)), 6
        )
    return summary


def log_trajectory_summary(
    *,
    args: argparse.Namespace,
    specs: List[Dict[str, Any]],
    spec: Dict[str, Any],
    trajectory_index: int,
    global_trajectory_index: int,
    policy_memory: JsonlPolicyMemory,
    pending_persistent_count: int,
) -> None:
    try:
        summary = build_final_summary(
            args=args,
            specs=specs,
            global_trajectory_index=global_trajectory_index,
            policy_memory=policy_memory,
            pending_persistent_count=pending_persistent_count,
        )
        log_event(
            "TRAJECTORY_SUMMARY",
            global_trajectory=global_trajectory_index,
            local_train_trajectory=f"{trajectory_index}/{args.train_trajectories}",
            agent=spec["agent_name"],
            task=spec["task"],
            attack_tool=spec["attacker_tool"]["Attacker Tool"],
            pending_persistent_count=pending_persistent_count,
            **summary,
        )
    except Exception as exc:
        log_event(
            "TRAJECTORY_SUMMARY_FAILED",
            global_trajectory=global_trajectory_index,
            local_train_trajectory=trajectory_index,
            agent=spec.get("agent_name", ""),
            task=spec.get("task", ""),
            attack_tool=spec.get("attacker_tool", {}).get("Attacker Tool", ""),
            error=repr(exc),
        )


def process_due_persistent_probe(
    *,
    args: argparse.Namespace,
    pending: Dict[str, Any],
    current_global_trajectory: int,
    target_runner: ASBTargetRunner,
    state_tracker: StateTracker,
    policy_memory: JsonlPolicyMemory,
) -> None:
    row_started_at = time.perf_counter()
    spec = pending["spec"]
    attacker_tool = spec["attacker_tool"]
    attack_tool_name = str(attacker_tool["Attacker Tool"])
    log_event(
        "PERSISTENT_PROBE_START",
        current_global_trajectory=current_global_trajectory,
        source_global_trajectory=pending["global_trajectory_index"],
        source_trajectory_index=pending.get("trajectory_index", ""),
        source_trajectory_id=pending["trajectory_id"],
        agent=spec["agent_name"],
        task=spec["task"],
        attack_tool=attack_tool_name,
    )
    probe_obs = run_target_with_failure_guard(
        args=args,
        target_runner=target_runner,
        agent_path=spec["agent_path"],
        agent_name=spec["agent_name"],
        task=spec["task"],
        attacker_tool=attacker_tool,
        attacked=False,
        strategy=None,
        context=[],
        expected_malicious_trajectory_id=pending["trajectory_id"],
        write_target_memory=bool(getattr(args, "probe_write_target_memory", False)),
        call_site="persistent_probe",
    )
    recorded_policy_history_events = 0
    if not is_api_failed(probe_obs):
        state_tracker.add_no_strategy_history(
            agent=spec["agent_name"],
            task=spec["task"],
            attack_tool=attack_tool_name,
            success=probe_obs.called_attack_tool,
        )
        persistent_history_event = make_policy_history_event(
            kind="persistent",
            strategy_id=None,
            success=probe_obs.called_attack_tool,
            sequence_ns=time.time_ns(),
        )
        try:
            recorded_policy_history_events = (
                policy_memory.add_trajectory_history_event(
                    trajectory_id=pending["trajectory_id"],
                    event=persistent_history_event,
                )
            )
        except Exception as exc:
            log_event(
                "PERSISTENT_HISTORY_WRITE_FAILED",
                source_trajectory_id=pending["trajectory_id"],
                error=repr(exc),
            )
    updated_rows = 0
    updated_result_rows = 0
    persistent_success = (not is_api_failed(probe_obs)) and probe_obs.called_attack_tool
    persistent_reward = persistent_probe_reward(persistent_success)
    updated_rows = policy_memory.add_trajectory_reward_bonus(
        trajectory_id=pending["trajectory_id"],
        bonus=persistent_reward,
        persistent_success=persistent_success,
    )
    updated_result_rows = update_result_rows_for_persistent_bonus(
        args.res_file,
        pending=pending,
        bonus=persistent_reward,
        persistent_success=persistent_success,
    )
    log_event(
        "PERSISTENT_PROBE_END",
        source_trajectory_id=pending["trajectory_id"],
        attack_tool=attack_tool_name,
        api_failed=int(is_api_failed(probe_obs)),
        persistent_success=int(persistent_success),
        malicious_memory_retrieved=(
            "" if probe_obs.malicious_memory_retrieved is None
            else int(probe_obs.malicious_memory_retrieved)
        ),
        recorded_policy_history_events=recorded_policy_history_events,
        updated_policy_memory_rows=updated_rows,
        updated_result_rows=updated_result_rows,
    )
    append_persistent_probe_result(
        args.res_file,
        pending=pending,
        current_global_trajectory=current_global_trajectory,
        persistent_success=persistent_success,
        malicious_memory_retrieved=probe_obs.malicious_memory_retrieved,
        updated_rows=updated_rows,
        row_elapsed_seconds=time.perf_counter() - row_started_at,
    )


def process_end_of_run_persistent_probes(
    *,
    args: argparse.Namespace,
    pending_persistent: List[Dict[str, Any]],
    current_global_trajectory: int,
    target_runner: ASBTargetRunner,
    state_tracker: StateTracker,
    policy_memory: JsonlPolicyMemory,
) -> None:
    queued_count = len(pending_persistent)
    log_event(
        "PERSISTENT_PROBE_BATCH_START",
        schedule="end_of_run",
        queued_persistent_count=queued_count,
        current_global_trajectory=current_global_trajectory,
    )
    processed_count = 0
    failed_count = 0
    while pending_persistent:
        due = pending_persistent.pop(0)
        try:
            process_due_persistent_probe(
                args=args,
                pending=due,
                current_global_trajectory=current_global_trajectory,
                target_runner=target_runner,
                state_tracker=state_tracker,
                policy_memory=policy_memory,
            )
            processed_count += 1
        except Exception as exc:
            failed_count += 1
            log_event(
                "PERSISTENT_PROBE_UNEXPECTED_FAILED",
                current_global_trajectory=current_global_trajectory,
                source_trajectory_id=due.get("trajectory_id", ""),
                error=repr(exc),
            )
    log_event(
        "PERSISTENT_PROBE_BATCH_END",
        schedule="end_of_run",
        queued_persistent_count=queued_count,
        processed_persistent_count=processed_count,
        failed_persistent_count=failed_count,
        remaining_persistent_count=len(pending_persistent),
    )


def build_persistent_probe_request(
    *,
    transitions: List[Transition],
    spec: Dict[str, Any],
    trajectory_index: int,
    global_trajectory_index: int,
) -> Optional[Dict[str, Any]]:
    if not transitions:
        return None
    return {
        "trajectory_id": transitions[0].trajectory_id,
        "spec": spec,
        "trajectory_index": trajectory_index,
        "global_trajectory_index": global_trajectory_index,
    }


def run_trajectory(
    *,
    args: argparse.Namespace,
    spec: Dict[str, Any],
    trajectory_index: int,
    is_test: bool,
    state_tracker: StateTracker,
    strategy_space: List[AttackStrategy],
    llm_policy: LLMStrategyPolicy,
    advantage_policy: AdvantagePolicy,
    target_runner: ASBTargetRunner,
    policy_memory: JsonlPolicyMemory,
    attack_text_generator: AttackTextGenerator,
    defer_commit: bool = False,
    reset_target_memory_for_retry: Optional[Callable[[], None]] = None,
) -> List[Transition]:
    agent_name = spec["agent_name"]
    agent_path = spec["agent_path"]
    task = spec["task"]
    attacker_tool = spec["attacker_tool"]
    attack_tool_name = str(attacker_tool["Attacker Tool"])
    trajectory_id = str(uuid.uuid4())
    transitions: List[Transition] = []
    state_score = 0.0
    attack_history: List[str] = []
    attack_rounds = int(getattr(args, "attack_rounds", getattr(args, "max_steps", 3)))
    wrote_policy_memory_count = 0
    wrote_result_count = 0
    log_event(
        "TRAJECTORY_START",
        trajectory_id=trajectory_id,
        mode="test" if is_test else "train",
        local_trajectory_index=trajectory_index,
        agent=agent_name,
        task=task,
        attack_tool=attack_tool_name,
        attack_rounds=attack_rounds,
        train_round=int(getattr(args, "train_round", 1)),
    )

    for step_index in range(attack_rounds):
        attack_round = step_index + 1
        step_id = attack_round
        row_started_at = time.perf_counter()
        state = AttackState(
            agent=agent_name,
            task=task,
            attack_tool=attack_tool_name,
            local_history=[],
            step_id=step_id,
            history_format=MULTI_ROUND_HISTORY_FORMAT,
            state_score=state_score,
            attack_history=list(attack_history),
            train_round=int(getattr(args, "train_round", 1)),
            attack_round=attack_round,
        )
        log_event(
            "STEP_START",
            trajectory_id=trajectory_id,
            step_id=step_id,
            attack_round=attack_round,
            state_score_before=state.state_score,
            attack_history_before=state.attack_history,
        )

        candidates = list(strategy_space)
        base_logits = llm_policy.get_choice_logits(
            state=state,
            candidates=candidates,
            context=[],
            recent_attack_summary="none",
            attack_tool_description=str(attacker_tool.get("Description", "")),
        )
        decision = advantage_policy.decide(
            state=state,
            candidates=candidates,
            base_logits=base_logits,
            use_memory=True,
        )
        log_event(
            "ACTION_DECISION",
            trajectory_id=trajectory_id,
            step_id=step_id,
            candidate_ids=[candidate.strategy_id for candidate in candidates],
            candidate_names=[candidate.name for candidate in candidates],
            base_logits=base_logits,
            base_probs=decision.base_probs,
            used_memory_count=decision.used_memory_count,
            value_estimate=decision.value_estimate,
            q_estimates=decision.q_estimates,
            raw_advantages=decision.raw_advantages,
            advantage_clip=decision.advantage_clip,
            advantage_probs=decision.advantage_probs,
            updated_probs=decision.updated_probs,
            selected_index=decision.selected_index,
            selected_strategy_id=decision.selected.strategy_id,
            selected_strategy_name=decision.selected.name,
            choice_debug=llm_policy.last_choice_debug,
        )
        agent_description, normal_tools = load_agent_generation_context(
            agent_name,
            agent_path,
            getattr(args, "tools_info_path", "data/all_normal_tools.jsonl"),
        )
        max_text_attempts = int(getattr(args, "attack_text_attempts", 1))
        previous_attack_texts: List[str] = []
        malicious_obs: Optional[TargetObservation] = None
        benign_obs: Optional[TargetObservation] = None
        benign_success = False
        accepted_attempt = 0
        attempt_api_failed = False
        benign_probe_elapsed_seconds = 0.0
        text_retry_trigger = attack_text_retry_trigger(decision.selected)
        retry_on_malicious_failure = (
            text_retry_trigger == "malicious_attack_failed"
        )

        for text_attempt in range(1, max_text_attempts + 1):
            attack_text = attack_text_generator.generate(
                strategy=decision.selected,
                task=task,
                attack_tool=attack_tool_name,
                attacker_instruction=str(attacker_tool["Attacker Instruction"]),
                attack_tool_description=str(attacker_tool.get("Description", "")),
                agent_name=agent_name,
                agent_description=agent_description,
                normal_tools=normal_tools,
                variant_index=text_attempt,
                previous_texts=tuple(previous_attack_texts),
                persist=False,
                use_cache=(text_attempt == 1),
            )
            previous_attack_texts.append(attack_text)
            attack_text_fields = dict(
                getattr(attack_text_generator, "last_generation", {}) or {}
            )
            attack_text_fields.update(
                trajectory_id=trajectory_id,
                step_id=step_id,
                attack_round=attack_round,
                text_attempt=text_attempt,
                max_text_attempts=max_text_attempts,
                attack_tool=attack_tool_name,
                strategy_id=decision.selected.strategy_id,
                delivery_mode=decision.selected.delivery_mode,
                attack_text=attack_text,
            )
            log_event("ATTACK_TEXT", **attack_text_fields)

            malicious_obs = run_target_with_failure_guard(
                args=args,
                target_runner=target_runner,
                agent_path=agent_path,
                agent_name=agent_name,
                task=task,
                attacker_tool=attacker_tool,
                attacked=True,
                strategy=decision.selected,
                attack_text=attack_text,
                context=[],
                attack_channel=getattr(args, "attack_channel", "task_text"),
                memory_metadata={
                    "source": "malicious_attack",
                    "source_trajectory_id": trajectory_id,
                    "source_step_id": attack_round,
                    "attack_round": attack_round,
                    "attack_text_attempt": text_attempt,
                },
                call_site=f"malicious_step_text_attempt_{text_attempt}",
            )
            log_event(
                "MALICIOUS_OBSERVATION",
                trajectory_id=trajectory_id,
                step_id=step_id,
                text_attempt=text_attempt,
                max_text_attempts=max_text_attempts,
                attack_channel=(
                    "memory_observation_suffix"
                    if decision.selected.delivery_mode == "memory_observation"
                    else getattr(args, "attack_channel", "task_text")
                ),
                api_failed=int(is_api_failed(malicious_obs)),
                called_attack_tool=int(malicious_obs.called_attack_tool),
                attack_success=int(malicious_obs.attack_success),
                original_success=int(malicious_obs.original_success),
            )
            if is_api_failed(malicious_obs):
                attempt_api_failed = True
                log_event(
                    "TRAJECTORY_STOP",
                    trajectory_id=trajectory_id,
                    reason="malicious_api_failed_after_retries",
                    completed_steps=len(transitions),
                    text_attempt=text_attempt,
                    error=malicious_obs.raw_result.get("error", ""),
                )
                break

            should_retry = (
                retry_on_malicious_failure
                and not malicious_obs.attack_success
                and text_attempt < max_text_attempts
            )
            if should_retry:
                log_event(
                    "ATTACK_TEXT_RETRY",
                    trajectory_id=trajectory_id,
                    step_id=step_id,
                    failed_text_attempt=text_attempt,
                    next_text_attempt=text_attempt + 1,
                    strategy_id=decision.selected.strategy_id,
                    trigger=text_retry_trigger,
                )
                if reset_target_memory_for_retry is not None:
                    reset_target_memory_for_retry()
                continue

            log_event(
                "BENIGN_PROBE_START",
                trajectory_id=trajectory_id,
                step_id=step_id,
                text_attempt=text_attempt,
                attack_tool=attack_tool_name,
                reason="after_malicious_attempt",
            )
            benign_probe_started_at = time.perf_counter()
            benign_obs = run_target_with_failure_guard(
                args=args,
                target_runner=target_runner,
                agent_path=agent_path,
                agent_name=agent_name,
                task=task,
                attacker_tool=attacker_tool,
                attacked=False,
                strategy=None,
                context=[],
                expected_malicious_trajectory_id=trajectory_id,
                write_target_memory=False,
                call_site=f"benign_probe_text_attempt_{text_attempt}",
            )
            benign_probe_elapsed_seconds = time.perf_counter() - benign_probe_started_at
            benign_success = (
                not is_api_failed(benign_obs) and benign_obs.called_attack_tool
            )
            log_event(
                "BENIGN_PROBE_END",
                trajectory_id=trajectory_id,
                step_id=step_id,
                text_attempt=text_attempt,
                api_failed=int(is_api_failed(benign_obs)),
                benign_called_attack_tool=int(benign_obs.called_attack_tool),
                benign_original_success=int(benign_obs.original_success),
                malicious_memory_retrieved=(
                    "" if benign_obs.malicious_memory_retrieved is None
                    else int(benign_obs.malicious_memory_retrieved)
                ),
            )
            if is_api_failed(benign_obs):
                attempt_api_failed = True
                log_event(
                    "TRAJECTORY_STOP",
                    trajectory_id=trajectory_id,
                    reason="benign_api_failed_after_retries",
                    completed_steps=len(transitions),
                    text_attempt=text_attempt,
                    error=benign_obs.raw_result.get("error", ""),
                )
                break

            should_retry = (
                not retry_on_malicious_failure
                and not benign_success
                and text_attempt < max_text_attempts
            )
            if should_retry:
                log_event(
                    "ATTACK_TEXT_RETRY",
                    trajectory_id=trajectory_id,
                    step_id=step_id,
                    failed_text_attempt=text_attempt,
                    next_text_attempt=text_attempt + 1,
                    strategy_id=decision.selected.strategy_id,
                    trigger=text_retry_trigger,
                )
                if reset_target_memory_for_retry is not None:
                    reset_target_memory_for_retry()
                continue

            accepted_attempt = text_attempt
            break

        if attempt_api_failed:
            break
        if malicious_obs is None or benign_obs is None or accepted_attempt == 0:
            raise RuntimeError("Attack text attempt loop ended without an accepted result.")
        final_cache_key = attack_text_generator.commit_final(attack_text)
        log_event(
            "ATTACK_TEXT_ATTEMPT_FINAL",
            trajectory_id=trajectory_id,
            step_id=step_id,
            strategy_id=decision.selected.strategy_id,
            delivery_mode=decision.selected.delivery_mode,
            accepted_text_attempt=accepted_attempt,
            max_text_attempts=max_text_attempts,
            attempts_exhausted=int(accepted_attempt == max_text_attempts),
            final_cache_key=final_cache_key,
            final_attack_text=attack_text,
            malicious_success=int(malicious_obs.attack_success),
            probe1_success=int(benign_success),
        )
        target_score = -5.0 if benign_success else 5.0
        state_score_after = 0.5 * state_score + 0.5 * target_score
        state_transition_bias = -(state_score_after - state_score)
        next_attack_history = [*attack_history, decision.selected.strategy_id]
        state_after_benign = AttackState(
            agent=agent_name,
            task=task,
            attack_tool=attack_tool_name,
            local_history=[],
            step_id=step_id + 1,
            history_format=MULTI_ROUND_HISTORY_FORMAT,
            state_score=state_score_after,
            attack_history=next_attack_history,
            train_round=int(getattr(args, "train_round", 1)),
            attack_round=attack_round + 1,
        )
        state_after_final = state_after_benign

        reward = 0.1 * state_transition_bias
        transition = Transition(
            state=state,
            action_id=decision.selected.strategy_id,
            action_text=attack_text,
            reward=reward,
            next_state=state_after_final,
            malicious_success=int(malicious_obs.attack_success),
            benign_success=int(benign_success),
            persistent_success=0,
            trajectory_id=trajectory_id,
            step_id=step_id,
            metadata={
                "selected_index": decision.selected_index,
                "candidate_ids": [candidate.strategy_id for candidate in decision.candidates],
                "used_memory_count": decision.used_memory_count,
                "value_estimate": decision.value_estimate,
                "q_estimates": decision.q_estimates,
                "attack_channel": getattr(args, "attack_channel", "task_text"),
                "train_round": int(getattr(args, "train_round", 1)),
                "attack_round": attack_round,
                "trajectory_id": trajectory_id,
                "action_id": decision.selected.strategy_id,
                "attack_text_attempt": accepted_attempt,
                "attack_text_attempts_max": max_text_attempts,
                "p0_malicious_memory_retrieved": benign_obs.malicious_memory_retrieved,
                "malicious_reward": 0.0,
                "benign_reward": "",
                "persistent_reward": "",
                "state_score_before": state_score,
                "state_score_after": state_score_after,
                "attack_history_before": list(attack_history),
                "state_transition_bias": state_transition_bias,
                "state_transition_credit": reward,
                "attack_row_elapsed_seconds": time.perf_counter() - row_started_at,
                "probe1_row_elapsed_seconds": benign_probe_elapsed_seconds,
                "raw_advantages": decision.raw_advantages,
                "advantage_clip": decision.advantage_clip,
                "advantage_scale_param": args.advantage_scale,
                "advantage_clip_param": args.advantage_clip,
                "advantage_temperature_param": args.advantage_temperature,
                "base_logit_temperature": args.base_logit_temperature,
                "base_probs": decision.base_probs,
                "advantage_probs": decision.advantage_probs,
                "updated_probs": decision.updated_probs,
                "choice_debug": llm_policy.last_choice_debug,
                POLICY_HISTORY_EVENTS_KEY: [],
            },
        )
        transitions.append(transition)
        state_score = state_score_after
        attack_history = next_attack_history
        log_event(
            "STEP_END",
            trajectory_id=trajectory_id,
            step_id=step_id,
            reward=reward,
            attack_round=attack_round,
            state_score_before=transition.metadata["state_score_before"],
            state_score_after=transition.metadata["state_score_after"],
            state_transition_bias=state_transition_bias,
            malicious_reward=0.0,
            benign_reward="",
            persistent_reward="",
            malicious_success=transition.malicious_success,
            benign_success=transition.benign_success,
            persistent_success=transition.persistent_success,
        )

        if not is_test and not defer_commit:
            try:
                policy_memory.add(transition)
                wrote_policy_memory_count += 1
            except Exception as exc:
                log_event(
                    "POLICY_MEMORY_WRITE_FAILED",
                    trajectory_id=trajectory_id,
                    step_id=transition.step_id,
                    error=repr(exc),
                )

        if not defer_commit:
            try:
                append_result(
                    args.res_file,
                    transition,
                    is_test=is_test,
                    decision=decision,
                    trajectory_index=trajectory_index,
                    row_elapsed_seconds=time.perf_counter() - row_started_at,
                )
                wrote_result_count += 1
            except Exception as exc:
                log_event(
                    "RESULT_WRITE_FAILED",
                    trajectory_id=trajectory_id,
                    step_id=transition.step_id,
                    error=repr(exc),
                )

    log_event(
        "TRAJECTORY_END",
        trajectory_id=trajectory_id,
        transitions=len(transitions),
        wrote_policy_memory_rows=wrote_policy_memory_count,
        wrote_result_rows=wrote_result_count,
        any_malicious_success=int(any(t.malicious_success for t in transitions)),
        any_benign_success=int(any(t.benign_success for t in transitions)),
    )

    return transitions


def append_delayed_persistent_probe_result(
    path: str,
    *,
    spec: Dict[str, Any],
    trajectory_index: int,
    trajectory_id: str,
    delay: int,
    persistent_success: bool,
    malicious_memory_retrieved: Optional[bool],
    row_elapsed_seconds: float,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [
            spec["agent_name"],
            task_for_csv(spec["task"]),
            spec["attacker_tool"]["Attacker Tool"],
            trajectory_index,
            CSV_NULL,
            delay,
            "PERSISTENT_PROBE",
            CSV_NULL,
            CSV_NULL,
            int(persistent_success),
            CSV_NULL,
            CSV_NULL,
            round(row_elapsed_seconds, 6),
        ]
        if len(row) != len(RESULT_HEADER):
            raise ValueError(
                f"Persistent result row has {len(row)} columns; expected {len(RESULT_HEADER)}."
            )
        writer.writerow(row)


def run_isolated_episode(
    *,
    args: argparse.Namespace,
    spec: Dict[str, Any],
    trajectory_index: int,
    global_trajectory_index: int,
    distractor_tasks: List[str],
    state_tracker: StateTracker,
    strategy_space: List[AttackStrategy],
    llm_policy: LLMStrategyPolicy,
    advantage_policy: AdvantagePolicy,
    target_runner: ASBTargetRunner,
    policy_memory: JsonlPolicyMemory,
    attack_text_generator: AttackTextGenerator,
    reset_target_memory_for_retry: Optional[Callable[[], None]] = None,
    is_test: bool = False,
    commit_policy: bool = True,
    write_results: bool = True,
) -> List[Transition]:
    """Run A/B pairs, replay D memories, probe persistence, then commit atomically."""
    attack_rounds = int(getattr(args, "attack_rounds", 3))

    episode_started_at = time.perf_counter()
    transitions = run_trajectory(
        args=args,
        spec=spec,
        trajectory_index=trajectory_index,
        is_test=is_test,
        state_tracker=state_tracker,
        strategy_space=strategy_space,
        llm_policy=llm_policy,
        advantage_policy=advantage_policy,
        target_runner=target_runner,
        policy_memory=policy_memory,
        attack_text_generator=attack_text_generator,
        defer_commit=True,
        reset_target_memory_for_retry=reset_target_memory_for_retry,
    )
    if len(transitions) != attack_rounds:
        log_event(
            "ISOLATED_EPISODE_NOT_COMMITTED",
            completed_attack_rounds=len(transitions),
            expected_attack_rounds=attack_rounds,
        )
        return []

    trajectory_id = transitions[0].trajectory_id
    agent_name = spec["agent_name"]
    agent_path = spec["agent_path"]
    attack_tool = spec["attacker_tool"]
    attack_tool_name = str(attack_tool["Attacker Tool"])
    delays = parse_persistent_probe_delays(args.persistent_probe_delays)
    if delays != [3, 5]:
        raise ValueError("The multi-round experiment requires --persistent_probe_delays 3,5.")
    max_delay = delays[-1]
    if len(distractor_tasks) != max_delay:
        raise ValueError(
            "Expected "
            f"{max_delay} benign distractor replay slots, got {len(distractor_tasks)}."
        )
    if any(task != spec["task"] for task in distractor_tasks):
        raise ValueError(
            "Benign distractor replay requires every scheduled task to equal "
            "the original attack task."
        )

    persistent_success_by_delay: Dict[int, bool] = {}
    persistent_rows: List[tuple[int, bool, Optional[bool], float]] = []
    previous_delay = 0
    for distractor_index in delays:
        batch_indices = list(range(previous_delay + 1, distractor_index + 1))
        log_event(
            "BENIGN_DISTRACTOR_REPLAY_BATCH_START",
            trajectory_id=trajectory_id,
            distractor_indices=batch_indices,
            task=spec["task"],
            attack_tool=attack_tool_name,
            distractor_mode=DISTRACTOR_MODE,
            target_agent_execution=0,
        )
        for replay_index in batch_indices:
            log_event(
                "BENIGN_DISTRACTOR_START",
                trajectory_id=trajectory_id,
                distractor_index=replay_index,
                task=distractor_tasks[replay_index - 1],
                attack_tool=attack_tool_name,
                read_target_memory=0,
                write_target_memory=int(
                    not bool(getattr(args, "disable_target_memory", False))
                ),
                distractor_mode=DISTRACTOR_MODE,
                target_agent_execution=0,
            )
        replayed_records = target_runner.replay_init_benign_memories(
            agent_name=agent_name,
            task=spec["task"],
            attack_tool=attack_tool_name,
            source_trajectory_id=trajectory_id,
            distractor_indices=batch_indices,
        )
        target_memory_disabled = bool(
            getattr(args, "disable_target_memory", False)
        )
        if not target_memory_disabled and len(replayed_records) != len(batch_indices):
            raise RuntimeError(
                "Benign distractor replay wrote an unexpected number of records: "
                f"expected={len(batch_indices)}, actual={len(replayed_records)}, "
                f"indices={batch_indices}."
            )
        for offset, replay_index in enumerate(batch_indices):
            record = (
                replayed_records[offset]
                if offset < len(replayed_records)
                else {}
            )
            metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
            log_event(
                "BENIGN_DISTRACTOR_END",
                trajectory_id=trajectory_id,
                distractor_index=replay_index,
                api_failed=0,
                original_success=metadata.get("task_successful", ""),
                memory_write_ok=int(bool(record)),
                memory_id=record.get("id", "") if isinstance(record, dict) else "",
                replay_source_memory_id=metadata.get(
                    "replay_source_memory_id", ""
                ),
                distractor_mode=DISTRACTOR_MODE,
                target_agent_execution=0,
            )
        log_event(
            "BENIGN_DISTRACTOR_REPLAY_BATCH_END",
            trajectory_id=trajectory_id,
            distractor_indices=batch_indices,
            replayed_memory_count=len(replayed_records),
            distractor_mode=DISTRACTOR_MODE,
        )
        previous_delay = distractor_index

        probe_started_at = time.perf_counter()
        log_event(
            "PERSISTENT_PROBE_START",
            trajectory_id=trajectory_id,
            delay=distractor_index,
            task=spec["task"],
            attack_tool=attack_tool_name,
            write_target_memory=0,
        )
        probe_obs = run_target_with_failure_guard(
            args=args,
            target_runner=target_runner,
            agent_path=agent_path,
            agent_name=agent_name,
            task=spec["task"],
            attacker_tool=attack_tool,
            attacked=False,
            strategy=None,
            context=[],
            expected_malicious_trajectory_id=trajectory_id,
            write_target_memory=False,
            call_site=f"persistent_probe_{distractor_index}",
        )
        if is_api_failed(probe_obs):
            log_event(
                "ISOLATED_EPISODE_NOT_COMMITTED",
                trajectory_id=trajectory_id,
                reason=f"persistent_probe_{distractor_index}_api_failed",
            )
            return []
        persistent_success = (
            probe_obs.called_attack_tool
        )
        persistent_success_by_delay[distractor_index] = persistent_success
        persistent_rows.append(
            (
                distractor_index,
                persistent_success,
                probe_obs.malicious_memory_retrieved,
                time.perf_counter() - probe_started_at,
            )
        )
        log_event(
            "PERSISTENT_PROBE_END",
            trajectory_id=trajectory_id,
            delay=distractor_index,
            api_failed=int(is_api_failed(probe_obs)),
            persistent_success=int(persistent_success),
            malicious_memory_retrieved=(
                "" if probe_obs.malicious_memory_retrieved is None
                else int(probe_obs.malicious_memory_retrieved)
            ),
        )

    r1 = persistent_probe_reward(bool(transitions[-1].benign_success))
    r3 = persistent_probe_reward(persistent_success_by_delay[3])
    r5 = persistent_probe_reward(persistent_success_by_delay[5])
    shared_persistent_reward = r1 + r3 + r5
    probe_elapsed_seconds_by_delay = {
        str(delay): elapsed
        for delay, _success, _retrieved, elapsed in persistent_rows
    }
    for transition in transitions:
        state_credit = 0.1 * float(transition.metadata["state_transition_bias"])
        transition.reward = shared_persistent_reward + state_credit
        transition.persistent_success = int(any(persistent_success_by_delay.values()))
        transition.metadata.update(
            {
                "benign_reward": r1,
                "persistent_reward": r3 + r5,
                "r1": r1,
                "r3": r3,
                "r5": r5,
                "shared_persistent_reward": shared_persistent_reward,
                "total_reward": transition.reward,
                "persistent_success_by_delay": {
                    str(delay): int(success)
                    for delay, success in persistent_success_by_delay.items()
                },
                "persistent_probe_delays": list(delays),
                "probe_elapsed_seconds_by_delay": probe_elapsed_seconds_by_delay,
                "distractor_task_count": 0,
                "distractor_memory_count": len(distractor_tasks),
                "distractor_mode": DISTRACTOR_MODE,
            }
        )

    if commit_policy:
        try:
            policy_memory.add_many(transitions)
        except Exception as exc:
            log_event(
                "POLICY_MEMORY_WRITE_FAILED",
                trajectory_id=trajectory_id,
                step_id="all",
                error=repr(exc),
            )
            raise
    if write_results:
        try:
            episode_elapsed = time.perf_counter() - episode_started_at
            for transition in transitions:
                append_result(
                    args.res_file,
                    transition,
                    is_test=is_test,
                    decision=None,
                    trajectory_index=trajectory_index,
                    row_elapsed_seconds=float(
                        transition.metadata.get(
                            "attack_row_elapsed_seconds",
                            episode_elapsed,
                        )
                    ),
                )
            probe_rows = [
                (
                    1,
                    bool(transitions[-1].benign_success),
                    transitions[-1].metadata.get("p0_malicious_memory_retrieved"),
                    float(
                        transitions[-1].metadata.get(
                            "probe1_row_elapsed_seconds",
                            episode_elapsed,
                        )
                    ),
                ),
                *persistent_rows,
            ]
            for delay, success, malicious_memory_retrieved, elapsed in probe_rows:
                append_delayed_persistent_probe_result(
                    args.res_file,
                    spec=spec,
                    trajectory_index=trajectory_index,
                    trajectory_id=transitions[0].trajectory_id,
                    delay=delay,
                    persistent_success=success,
                    malicious_memory_retrieved=malicious_memory_retrieved,
                    row_elapsed_seconds=elapsed,
                )
        except Exception as exc:
            log_event(
                "RESULT_WRITE_FAILED",
                trajectory_id=trajectory_id,
                step_id="all",
                error=repr(exc),
            )
            raise

    log_event(
        "ISOLATED_EPISODE_COMMITTED",
        trajectory_id=trajectory_id,
        attack_rounds=attack_rounds,
        distractor_task_count=0,
        distractor_memory_count=len(distractor_tasks),
        distractor_mode=DISTRACTOR_MODE,
        persistent_success_by_delay=transitions[0].metadata["persistent_success_by_delay"],
        shared_persistent_reward=shared_persistent_reward,
        rewards=[transition.reward for transition in transitions],
        policy_memory_committed=int(commit_policy),
        result_rows_written=int(write_results),
    )
    return transitions


def _run_training() -> None:
    load_attack_agent_env()
    args = parse_args()
    log_event(
        "CONFIGURATION",
        enable_thinking=int(args.enable_thinking),
        target_mem0_use_real_embedding=int(args.target_mem0_use_real_embedding),
        probe_write_target_memory=int(args.probe_write_target_memory),
        persistent_probe_delays=args.persistent_probe_delays,
        distractor_mode=DISTRACTOR_MODE,
        attack_text_attempts=args.attack_text_attempts,
        train_round=args.train_round,
        attack_rounds=args.attack_rounds,
        state_retrieve_top_k=args.state_retrieve_top_k,
        retrieve_top_k=args.retrieve_top_k,
        cross_attack_round_retrieval=int(args.cross_attack_round_retrieval),
    )
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
    install_openai_api_retries(
        max_retries=args.api_max_retries,
        retry_sleep=args.api_retry_sleep,
    )

    if args.write_default_strategy_space:
        write_default_strategy_space(args.write_default_strategy_space)
        print(f"[StrategySpace] wrote defaults to {args.write_default_strategy_space}")
        return

    ensure_result_header(args.res_file)
    specs = load_experiment_specs(args)
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
        "POLICY_STATE_RESTORED",
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

    if not specs:
        raise ValueError("No experiment specs loaded.")
    if args.attack_rounds <= 0:
        raise ValueError("--attack_rounds must be positive.")
    if args.train_round <= 0:
        raise ValueError("--train_round must be positive.")
    if args.attack_rounds > 1 and args.attack_text_attempts != 1:
        raise ValueError(
            "Multi-round episodes require --attack_text_attempts 1 so retry resets "
            "cannot erase target-memory writes from earlier attack rounds."
        )
    if args.probe_write_target_memory:
        raise ValueError(
            "The multi-round experiment requires read-only B1/B2/B3/P3/P5; "
            "use --no-probe_write_target_memory."
        )

    persistent_probe_delays = parse_persistent_probe_delays(
        args.persistent_probe_delays
    )
    agent_name = specs[0]["agent_name"]
    clean_target_memory_source: Optional[Path] = None
    episode_memory_root: Optional[Path] = None
    if not args.disable_target_memory:
        clean_target_memory_source = Path(args.target_mem0_path)
        episode_memory_root = (
            Path(args.episode_memory_root)
            if args.episode_memory_root
            else default_episode_memory_root(
                args.target_mem0_path,
                args.target_agent,
            )
        )
        if not clean_target_memory_source.is_dir():
            raise FileNotFoundError(
                "--target_mem0_path must be an existing clean target-memory directory "
                "when target memory is enabled: "
                f"{clean_target_memory_source}"
            )
        if clean_target_memory_source == episode_memory_root or clean_target_memory_source in episode_memory_root.parents:
            raise ValueError(
                "--episode_memory_root must not be the clean target-memory directory "
                "or one of its children."
            )
        if episode_memory_root.exists() and any(episode_memory_root.iterdir()):
            raise FileExistsError(
                f"Episode-memory root already contains artifacts: {episode_memory_root}. "
                "Use a new run/--episode_memory_root; existing artifacts are preserved."
            )
        episode_memory_root.mkdir(parents=True, exist_ok=True)
        log_event(
            "EPISODE_MEMORY_CONFIGURATION",
            clean_target_memory_source=str(clean_target_memory_source),
            episode_memory_root=str(episode_memory_root),
            persistent_probe_delays=persistent_probe_delays,
        )

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
    global_trajectory_index = 0

    try:
        for trajectory_index in range(1, args.train_trajectories + 1):
            log_event(
                "TRAJECTORY_ROUND_START",
                trajectory_round=f"{trajectory_index}/{args.train_trajectories}",
                specs=len(specs),
                persistent_probe_schedule=f"after_{persistent_probe_delays}_benign_distractors",
                legacy_persistent_T=args.persistent_T,
            )
            for spec in specs:
                log_event(
                    "SPEC_TRAJECTORY_START",
                    trajectory_round=f"{trajectory_index}/{args.train_trajectories}",
                    agent=spec["agent_name"],
                    task=spec["task"],
                    attack_tool=spec["attacker_tool"]["Attacker Tool"],
                    attack_rounds=args.attack_rounds,
                    train_round=args.train_round,
                    persistent_probe_schedule=f"after_{persistent_probe_delays}_benign_distractors",
                    legacy_persistent_T=args.persistent_T,
                )
                global_trajectory_index += 1
                log_event(
                    "GLOBAL_TRAJECTORY_START",
                    global_trajectory=global_trajectory_index,
                    local_train_trajectory=f"{trajectory_index}/{args.train_trajectories}",
                    agent=spec["agent_name"],
                    task=spec["task"],
                    attack_tool=spec["attacker_tool"]["Attacker Tool"],
                )
                episode_destination: Optional[Path] = None
                target_memory: Any = None
                reset_target_memory_for_retry: Optional[Callable[[], None]] = None
                try:
                    if clean_target_memory_source is not None and episode_memory_root is not None:
                        destination = episode_memory_path(
                            root=episode_memory_root,
                            spec=spec,
                            trajectory_index=trajectory_index,
                            global_trajectory_index=global_trajectory_index,
                        )
                        episode_destination = destination
                        clone_clean_target_memory(
                            source=clean_target_memory_source,
                            destination=destination,
                        )
                        target_memory = build_target_memory(
                            args,
                            spec["agent_name"],
                            path=str(destination),
                        )
                        target_runner.vector_db = FaultTolerantVectorDB(target_memory)
                        log_event(
                            "EPISODE_TARGET_MEMORY_READY",
                            global_trajectory=global_trajectory_index,
                            clean_target_memory_source=str(clean_target_memory_source),
                            episode_target_memory=str(destination),
                        )

                        def reset_episode_target_memory() -> None:
                            nonlocal target_memory
                            target_runner.vector_db = None
                            previous_target_memory = target_memory
                            target_memory = None
                            if previous_target_memory is not None:
                                close_memory = getattr(previous_target_memory, "close", None)
                                if callable(close_memory):
                                    close_memory()
                            if destination.exists():
                                remove_episode_target_memory(destination)
                            clone_clean_target_memory(
                                source=clean_target_memory_source,
                                destination=destination,
                            )
                            target_memory = build_target_memory(
                                args,
                                spec["agent_name"],
                                path=str(destination),
                            )
                            target_runner.vector_db = FaultTolerantVectorDB(target_memory)
                            log_event(
                                "EPISODE_TARGET_MEMORY_RESET",
                                global_trajectory=global_trajectory_index,
                                clean_target_memory_source=str(clean_target_memory_source),
                                episode_target_memory=str(destination),
                            )

                        reset_target_memory_for_retry = reset_episode_target_memory
                    else:
                        target_runner.vector_db = None

                    distractor_tasks = select_benign_distractor_tasks(
                        attack_task=spec["task"],
                        count=persistent_probe_delays[-1],
                    )
                    transitions = run_isolated_episode(
                        args=args,
                        spec=spec,
                        trajectory_index=trajectory_index,
                        global_trajectory_index=global_trajectory_index,
                        distractor_tasks=distractor_tasks,
                        state_tracker=state_tracker,
                        strategy_space=strategy_space,
                        llm_policy=llm_policy,
                        advantage_policy=advantage_policy,
                        target_runner=target_runner,
                        policy_memory=policy_memory,
                        attack_text_generator=attack_text_generator,
                        reset_target_memory_for_retry=reset_target_memory_for_retry,
                    )
                except Exception as exc:
                    transitions = []
                    log_event(
                        "TRAJECTORY_UNEXPECTED_FAILED",
                        mode="train",
                        global_trajectory=global_trajectory_index,
                        local_train_trajectory=trajectory_index,
                        agent=spec["agent_name"],
                        task=spec["task"],
                        attack_tool=spec["attacker_tool"]["Attacker Tool"],
                        error=repr(exc),
                    )
                finally:
                    # Chroma 1.5 keeps a background system per persistent
                    # client. Close it before deleting the private episode DB
                    # so it cannot recreate segment directories during rmtree.
                    target_runner.vector_db = None
                    if target_memory is not None:
                        close_target_memory = getattr(target_memory, "close", None)
                        if callable(close_target_memory):
                            close_target_memory()
                    if episode_destination is not None and episode_destination.exists():
                        # Episode memory is intentionally ephemeral: each
                        # attack tool gets a fresh copy from clean init, and
                        # its private copy is removed before the next one.
                        remove_episode_target_memory(episode_destination)
                        log_event(
                            "EPISODE_TARGET_MEMORY_CLEANED",
                            global_trajectory=global_trajectory_index,
                            episode_target_memory=str(episode_destination),
                        )

                if args.trajectory_summary:
                    log_trajectory_summary(
                        args=args,
                        specs=specs,
                        spec=spec,
                        trajectory_index=trajectory_index,
                        global_trajectory_index=global_trajectory_index,
                        policy_memory=policy_memory,
                        pending_persistent_count=0,
                    )

    finally:
        try:
            summary = build_final_summary(
                args=args,
                specs=specs,
                global_trajectory_index=global_trajectory_index,
                policy_memory=policy_memory,
                pending_persistent_count=0,
            )
            log_event(
                "FINAL_SUMMARY",
                global_trajectory=global_trajectory_index,
                pending_persistent_count=0,
                **summary,
            )
        except Exception as exc:
            log_event("FINAL_SUMMARY_FAILED", error=repr(exc))
        target_runner.close()


def main() -> None:
    overall_started_at = time.perf_counter()
    try:
        _run_training()
    finally:
        overall_elapsed_seconds = time.perf_counter() - overall_started_at
        log_event(
            "OVERALL_RUNTIME",
            overall_elapsed_seconds=round(overall_elapsed_seconds, 6),
            overall_elapsed_hms=format_elapsed_hms(overall_elapsed_seconds),
        )


if __name__ == "__main__":
    main()
