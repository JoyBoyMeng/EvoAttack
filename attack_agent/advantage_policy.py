from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

from .models import ActionDecision, AttackState, AttackStrategy, Transition
from .policy_memory import JsonlPolicyMemory


def softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits) for _ in logits]
    return [x / total for x in exps]


def temperature_softmax(logits: Sequence[float], temperature: float) -> List[float]:
    temperature = temperature if temperature > 0 else 1.0
    return softmax([float(x) / temperature for x in logits])


def normalize_probs(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    clipped = [max(0.0, float(x)) for x in values]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped) for _ in clipped]
    return [x / total for x in clipped]


def choose_from_logits(
    strategies: Sequence[AttackStrategy],
    logits: Sequence[float],
    *,
    mode: str,
    rng: random.Random,
) -> tuple[int, AttackStrategy]:
    if not strategies:
        raise ValueError("No candidate strategies are available.")
    if mode == "argmax":
        idx = max(range(len(logits)), key=lambda i: logits[i])
        return idx, strategies[idx]
    probs = softmax(logits)
    r = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if r <= cumulative:
            return idx, strategies[idx]
    return len(strategies) - 1, strategies[-1]


def choose_from_probs(
    strategies: Sequence[AttackStrategy],
    probs: Sequence[float],
    *,
    mode: str,
    rng: random.Random,
) -> tuple[int, AttackStrategy]:
    if not strategies:
        raise ValueError("No candidate strategies are available.")
    if mode == "argmax":
        idx = max(range(len(probs)), key=lambda i: probs[i])
        return idx, strategies[idx]
    r = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if r <= cumulative:
            return idx, strategies[idx]
    return len(strategies) - 1, strategies[-1]


def estimate_advantages(
    memories: Sequence[Transition],
    candidates: Sequence[AttackStrategy],
) -> tuple[float, Dict[str, float | None], List[float]]:
    if not memories:
        return 0.0, {c.strategy_id: None for c in candidates}, [0.0 for _ in candidates]

    value = sum(m.reward for m in memories) / len(memories)
    q_estimates: Dict[str, float | None] = {}
    advantages: List[float] = []

    for candidate in candidates:
        same_action = [m.reward for m in memories if m.action_id == candidate.strategy_id]
        if same_action:
            q = sum(same_action) / len(same_action)
        else:
            q = 0.0
        adv = q - value
        q_estimates[candidate.strategy_id] = q
        advantages.append(adv)

    return value, q_estimates, advantages


def scale_advantages(
    advantages: Sequence[float],
    *,
    scale: float,
    clip: float,
) -> List[float]:
    scale = scale if scale > 0 else 1.0
    clip = abs(clip)
    scaled = [adv / scale for adv in advantages]
    if clip <= 0:
        return scaled
    return [max(-clip, min(clip, adv)) for adv in scaled]


class AdvantagePolicy:
    def __init__(
        self,
        *,
        memory: JsonlPolicyMemory,
        beta: float = 1.0,
        eps: float = 2.5,
        top_k: int = 50,
        state_top_k: int = 400,
        cross_attack_round_retrieval: bool = False,
        advantage_scale: float = 0.5,
        advantage_clip: float = 5.0,
        advantage_temperature: float = 0.3,
        selection_mode: str = "argmax",
        seed: int = 0,
    ) -> None:
        self.memory = memory
        self.beta = beta
        self.eps = eps
        self.top_k = top_k
        self.state_top_k = state_top_k
        self.cross_attack_round_retrieval = cross_attack_round_retrieval
        self.advantage_scale = advantage_scale
        self.advantage_clip = advantage_clip
        self.advantage_temperature = advantage_temperature
        self.selection_mode = selection_mode
        self.rng = random.Random(seed)

    def decide(
        self,
        *,
        state: AttackState,
        candidates: List[AttackStrategy],
        base_logits: List[float],
        use_memory: bool,
    ) -> ActionDecision:
        if len(candidates) != len(base_logits):
            raise ValueError("candidates and base_logits must have the same length.")

        memories: List[Transition] = []
        value = 0.0
        q_estimates: Dict[str, float | None] = {c.strategy_id: None for c in candidates}
        advantages = [0.0 for _ in candidates]

        if use_memory:
            memories = self.memory.retrieve(
                state=state,
                eps=self.eps,
                top_k=self.top_k,
                same_task=True,
                same_attack_tool=True,
                state_top_k=self.state_top_k,
                cross_attack_round_retrieval=self.cross_attack_round_retrieval,
            )
            value, q_estimates, advantages = estimate_advantages(memories, candidates)

        scaled_advantages = scale_advantages(
            advantages,
            scale=self.advantage_scale,
            clip=self.advantage_clip,
        )
        base_probs = softmax(base_logits)
        advantage_probs = temperature_softmax(
            scaled_advantages,
            self.advantage_temperature,
        )
        has_advantage_signal = bool(memories) and any(abs(adv) > 1e-12 for adv in scaled_advantages)
        if has_advantage_signal:
            updated_probs = normalize_probs(
                base_prob + self.beta * advantage_prob
                for base_prob, advantage_prob in zip(base_probs, advantage_probs)
            )
        else:
            updated_probs = list(base_probs)
        idx, selected = choose_from_probs(
            candidates,
            updated_probs,
            mode=self.selection_mode,
            rng=self.rng,
        )
        return ActionDecision(
            candidates=candidates,
            base_logits=base_logits,
            advantage_clip=scaled_advantages,
            selected=selected,
            selected_index=idx,
            used_memory_count=len(memories),
            value_estimate=value,
            q_estimates=q_estimates,
            raw_advantages=advantages,
            base_probs=base_probs,
            advantage_probs=advantage_probs,
            updated_probs=updated_probs,
        )
