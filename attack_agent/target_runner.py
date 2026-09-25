from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from aios.llm_core import llms
from aios.scheduler.fifo_scheduler import FIFOScheduler
from pyopenagi.agents.agent_factory import AgentFactory
from pyopenagi.agents.agent_process import AgentProcessFactory

from eval_all_agents_mem0_agent_isolated import (
    build_normal_episode_memory_text,
    check_attack_success,
    check_original_success,
)

from .models import AttackStrategy, TargetObservation


def _coerce_tool_record(record: Any) -> Dict[str, Any]:
    if hasattr(record, "to_dict"):
        return dict(record.to_dict())
    return dict(record)


def _empty_attacker_tool_record() -> Dict[str, Any]:
    """Compatibility placeholder for ASB agent constructors.

    The underlying ASB agent always expects these fields during construction.
    With ``memory_attack=False`` this record is never registered as a tool, so
    the target agent receives only its normal tools.
    """
    return {
        "Attacker Tool": "",
        "Attacker Instruction": "",
        "Attack goal": "",
        "Aggressive": "False",
    }


def strip_init_memory_observations(text: str) -> str:
    """Remove tool observations while retaining task, plan, and actions.

    The ASB agent's ``messages_for_memory`` uses fixed message positions, so
    some action messages can retain their observation payload.  Init seeds are
    intentionally clean normal-tool references and must not persist those
    payloads.
    """
    return re.sub(
        r"\s*;?\s*\[(?:Observation|观察)\]\s*[:：].*?(?=(?:\s+(?:assistant|user):\s+(?:\[Action\]|At step))|$)",
        "",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def build_asb_args(
    *,
    llm_name: str,
    tools_info_path: str,
    workflow_mode: str = "automatic",
    read_db: bool = True,
    write_db: bool = True,
    max_new_tokens: int = 128,
    use_backend: Optional[str] = None,
    log_mode: str = "console",
    enable_thinking: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        llm_name=llm_name,
        max_gpu_memory=None,
        eval_device=None,
        max_new_tokens=max_new_tokens,
        scheduler_log_mode=log_mode,
        agent_log_mode=log_mode,
        llm_kernel_log_mode=log_mode,
        use_backend=use_backend,
        workflow_mode=workflow_mode,
        tools_info_path=tools_info_path,
        read_db=read_db,
        write_db=write_db,
        memory_attack=True,
        direct_prompt_injection=False,
        observation_prompt_injection=False,
        pot_backdoor=False,
        pot_clean=False,
        attack_type="naive",
        defense_type=None,
        target="",
        trigger="",
        database="",
        enable_thinking=enable_thinking,
    )


class ASBTargetRunner:
    """
    Wrapper around the existing ASB target-agent execution path.

    This class intentionally does not modify ASB source code. It uses
    AgentFactory.run_agent and performs optional normal-memory writes after each
    interaction, mirroring the eval script behavior.
    """

    def __init__(
        self,
        *,
        llm_name: str,
        tools_info_path: str = "data/all_normal_tools.jsonl",
        workflow_mode: str = "automatic",
        vector_db: Any = None,
        read_db: bool = True,
        write_db: bool = True,
        max_new_tokens: int = 128,
        use_backend: Optional[str] = None,
        log_mode: str = "console",
        enable_thinking: bool = False,
    ) -> None:
        self.args_template = build_asb_args(
            llm_name=llm_name,
            tools_info_path=tools_info_path,
            workflow_mode=workflow_mode,
            read_db=read_db,
            write_db=write_db,
            max_new_tokens=max_new_tokens,
            use_backend=use_backend,
            log_mode=log_mode,
            enable_thinking=enable_thinking,
        )
        self.vector_db = vector_db
        self.llm = llms.LLMKernel(
            llm_name=llm_name,
            max_gpu_memory=None,
            eval_device=None,
            max_new_tokens=max_new_tokens,
            log_mode=log_mode,
            use_backend=use_backend,
            enable_thinking=enable_thinking,
        )
        self.scheduler = FIFOScheduler(llm=self.llm, log_mode=log_mode)
        self.agent_process_factory = AgentProcessFactory()
        self.agent_factory = AgentFactory(
            agent_process_queue=self.scheduler.agent_process_queue,
            agent_process_factory=self.agent_process_factory,
            agent_log_mode=log_mode,
        )
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.scheduler.start()

    def close(self) -> None:
        self.scheduler.stop()
        self.thread_pool.shutdown(wait=True)

    def replay_init_benign_memories(
        self,
        *,
        agent_name: str,
        task: str,
        attack_tool: str,
        source_trajectory_id: str,
        distractor_indices: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Replay the matching init seed as newer benign distractor records."""
        indices = [int(index) for index in distractor_indices]
        if not indices:
            return []
        if any(index < 1 for index in indices) or indices != sorted(set(indices)):
            raise ValueError(
                "distractor_indices must be unique positive integers in ascending order"
            )
        if self.vector_db is None:
            return []

        task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]
        source_filters: Dict[str, Any] = {
            "source": "init_benign_seed",
            "task_hash": task_hash,
            "attack_tool": str(attack_tool),
            "init_copy_index": 1,
        }
        memory_user_id = getattr(self.vector_db, "user_id", None)
        memory_agent_id = getattr(self.vector_db, "agent_id", None)
        if memory_user_id:
            source_filters["user_id"] = str(memory_user_id)
        if memory_agent_id:
            source_filters["agent_id"] = str(memory_agent_id)
        elif agent_name:
            source_filters["agent_id"] = str(agent_name)

        metadata_overrides = [
            {
                "source": "benign_distractor",
                "distractor_mode": "init_seed_replay",
                "distractor_index": index,
                "source_trajectory_id": str(source_trajectory_id),
                "attack_tool": str(attack_tool),
                "task_hash": task_hash,
                "attacked": 0,
                "attack_channel": "none",
                "attack_observation_appended": 0,
            }
            for index in indices
        ]
        return self.vector_db.clone_memory_records(
            source_filters=source_filters,
            metadata_overrides=metadata_overrides,
            drop_metadata_keys={
                "phase",
                "task_index",
                "attack_tool_index",
                "benign_seed_index",
                "init_copy_index",
            },
        )

    def run(
        self,
        *,
        agent_path: str,
        agent_name: str,
        task: str,
        attacker_tool: Optional[Mapping[str, Any]],
        attacked: bool,
        strategy: Optional[AttackStrategy] = None,
        attack_text: Optional[str] = None,
        context: Optional[Sequence[str]] = None,
        expected_malicious_trajectory_id: Optional[str] = None,
        read_target_memory: Optional[bool] = None,
        write_target_memory: Optional[bool] = None,
        memory_metadata: Optional[Mapping[str, Any]] = None,
        attack_channel: Optional[str] = None,
        memory_observation_suffix: Optional[str] = None,
        observation_strategy_id: Optional[str] = None,
    ) -> TargetObservation:
        args = copy.copy(self.args_template)
        # Keep retrieval/write provenance tied to the original task.  For a
        # malicious call, ``task_input`` below additionally contains attack
        # text and must not change the task partition.
        args.memory_task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]
        if read_target_memory is not None:
            args.read_db = bool(read_target_memory)
        if write_target_memory is not None:
            args.write_db = bool(write_target_memory)
        has_attacker_tool = attacker_tool is not None
        if attacked and not has_attacker_tool:
            raise ValueError("attacker_tool is required for malicious interactions.")
        tool_record = (
            _coerce_tool_record(attacker_tool)
            if has_attacker_tool
            else _empty_attacker_tool_record()
        )
        original_attacker_instruction = str(tool_record.get("Attacker Instruction", ""))
        attack_tool_name = str(tool_record.get("Attacker Tool", ""))
        if attacked:
            if strategy is None:
                raise ValueError("strategy is required for malicious attack interactions.")
            strategy_text = attack_text
            if strategy_text is None:
                strategy_text = strategy.render(
                    task=task,
                    attack_tool=attack_tool_name,
                    attacker_instruction=original_attacker_instruction,
                )
            delivery_mode = str(strategy.delivery_mode or "task_text")
            if delivery_mode not in {"task_text", "memory_observation"}:
                raise ValueError(f"Unsupported strategy delivery_mode={delivery_mode!r}")
            attack_channel = str(attack_channel or "task_text")
            if delivery_mode == "memory_observation":
                # S10+ keeps the malicious task clean.  The selected payload
                # is appended only after the ordinary episode serializer has
                # removed real observations, immediately before the write.
                task_input = task
                args.attack_channel = "memory_observation_suffix"
                args.observation_prompt_injection = False
                args.observation_attack_text = ""
                args.observation_attack_include_final = False
                args.preserve_observation_memory = False
                memory_observation_suffix = strategy_text
                observation_strategy_id = strategy.strategy_id
            else:
                if attack_channel not in {"task_text", "observation"}:
                    raise ValueError(f"Unsupported attack_channel={attack_channel!r}")
                task_input = (
                    f"{task}\n\n{strategy_text}"
                    if attack_channel == "task_text"
                    else task
                )
                args.attack_channel = attack_channel
                args.observation_prompt_injection = attack_channel == "observation"
                args.observation_attack_text = strategy_text
                args.observation_attack_include_final = attack_channel == "observation"
                args.preserve_observation_memory = attack_channel == "observation"
            args.direct_prompt_injection = False
            args.memory_attack = True
            args.attack_type = "naive"
        else:
            task_input = task
            args.direct_prompt_injection = False
            args.memory_attack = has_attacker_tool
            args.observation_prompt_injection = False
            args.attack_channel = "none"
            args.observation_attack_text = ""
            args.observation_attack_include_final = False
            args.preserve_observation_memory = False

        future = self.thread_pool.submit(
            self.agent_factory.run_agent,
            agent_path,
            task_input,
            args,
            tool_record,
            self.vector_db,
            tool_record.get("Aggressive", "False"),
        )
        res = future.result()

        malicious_memory_retrieved: Optional[bool] = None
        if expected_malicious_trajectory_id is not None:
            retrieved_metadata = res.get("retrieved_memory_metadata", [])
            if not isinstance(retrieved_metadata, list):
                retrieved_metadata = []
            malicious_memory_retrieved = any(
                isinstance(metadata, Mapping)
                and metadata.get("source") == "malicious_attack"
                and metadata.get("source_trajectory_id")
                == expected_malicious_trajectory_id
                for metadata in retrieved_metadata
            )

        attacker_goal = str(tool_record.get("Attack goal", ""))
        attack_success = (
            check_attack_success(res.get("messages", []), args, attacker_goal)
            if has_attacker_tool
            else False
        )
        original_success = check_original_success(res.get("messages", []), args, res.get("normal_tools", {}))
        called_attack_tool = attack_success

        self._write_target_memory_if_enabled(
            args=args,
            res=res,
            task=task_input,
            embedding_task=task,
            attack_tool=attack_tool_name,
            attacked=attacked,
            original_successful=original_success,
            memory_metadata=memory_metadata,
            memory_observation_suffix=memory_observation_suffix,
            observation_strategy_id=observation_strategy_id,
        )

        return TargetObservation(
            agent=agent_name,
            task=task,
            attack_tool=attack_tool_name,
            attacked=attacked,
            called_attack_tool=called_attack_tool,
            attack_success=attack_success,
            original_success=original_success,
            malicious_memory_retrieved=malicious_memory_retrieved,
            messages=res.get("messages", []),
            raw_result=res,
        )

    def _write_target_memory_if_enabled(
        self,
        *,
        args: SimpleNamespace,
        res: Dict[str, Any],
        task: str,
        embedding_task: Optional[str] = None,
        attack_tool: str,
        attacked: bool,
        original_successful: bool,
        memory_metadata: Optional[Mapping[str, Any]] = None,
        memory_observation_suffix: Optional[str] = None,
        observation_strategy_id: Optional[str] = None,
    ) -> None:
        if not args.write_db or self.vector_db is None:
            return
        text = build_normal_episode_memory_text(
            res=res,
            task=task,
            original_successful=original_successful,
        )
        if memory_metadata and memory_metadata.get("source") == "init_benign_seed":
            text = strip_init_memory_observations(text)
        observation_suffix = str(memory_observation_suffix or "").strip()
        if observation_suffix:
            text = f"{text}\n[Observation]: {observation_suffix}"
        metadata_overrides = dict(memory_metadata or {})
        # Use the attack context for retrieval metadata even though D1-D5 run
        # with normal tools only. This intentionally lets malicious and
        # distractor episodes share the probe's exact (task, tools) partition;
        # the actual executed tools remain recoverable from the episode body.
        context_attack_tool = str(
            metadata_overrides.get("attack_tool") or attack_tool or ""
        )
        available_tool_names = set(
            str(name) for name in res.get("normal_tools", {})
        )
        if context_attack_tool:
            available_tool_names.add(context_attack_tool)
        metadata = {
            "source": "adaptive_attack_target_episode",
            "agent_name": res.get("agent_name"),
            "attack_tool": context_attack_tool,
            "task_hash": str(getattr(args, "memory_task_hash", "")),
            "attacked": int(bool(attacked)),
            "task_successful": int(bool(original_successful)),
            "attack_channel": str(getattr(args, "attack_channel", "task_text")),
            "attack_observation_appended": int(bool(observation_suffix)),
            # The full body above intentionally keeps task_input (including a
            # task-text attack).  Embedding and task_hash remain tied to the
            # original benign task so malicious and distractor writes share
            # the same retrieval partition as P0/P3/P5.
            "embedding_task": str(embedding_task or task),
            "available_tools": json.dumps(
                sorted(available_tool_names),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
        if observation_suffix:
            metadata["attack_observation_strategy_id"] = str(observation_strategy_id or "")
        if metadata_overrides:
            metadata.update(metadata_overrides)
        if observation_suffix:
            print("\n[AdaptiveAttack::TARGET_MEMORY_WRITE]", flush=True)
            print("memory_write_mode: memory_observation_suffix", flush=True)
            print(f"attack_observation_strategy_id: {observation_strategy_id or ''}", flush=True)
            print(f"attack_observation_text: {observation_suffix}", flush=True)
            print(f"final_memory_text: {text}", flush=True)
        self.vector_db.add_texts([text], metadatas=[metadata])
