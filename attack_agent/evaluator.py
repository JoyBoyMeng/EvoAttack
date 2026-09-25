from __future__ import annotations

from typing import Dict, Iterable, List

from .models import TargetObservation, Transition

BENIGN_SUCCESS_REWARD = 1.0
BENIGN_FAILURE_REWARD = -1.0
PERSISTENT_SUCCESS_BONUS = 1.0
PERSISTENT_FAILURE_PENALTY = -1.0


def malicious_step_reward(
    *,
    benign_success: bool = False,
) -> float:
    components = malicious_step_reward_components(
        benign_success=benign_success,
    )
    return (
        components["benign_reward"]
        + components["persistent_reward"]
    )


def malicious_step_reward_components(
    *,
    benign_success: bool = False,
) -> Dict[str, float]:
    """Return reward components for a malicious write and its clean probes.

    The malicious interaction is a target-memory write opportunity, not an
    attacker-policy objective. Its result remains in the transition for state
    tracking and diagnosis, but contributes no reward.
    """
    benign_reward = BENIGN_SUCCESS_REWARD if benign_success else BENIGN_FAILURE_REWARD

    return {
        # Retain the zero component so existing CSV/event readers can audit
        # that direct malicious success is excluded from policy learning.
        "malicious_reward": 0.0,
        "benign_reward": benign_reward,
        "persistent_reward": 0.0,
    }


def persistent_probe_reward(persistent_success: bool) -> float:
    return (
        PERSISTENT_SUCCESS_BONUS
        if persistent_success
        else PERSISTENT_FAILURE_PENALTY
    )


def add_persistent_reward(transitions: Iterable[Transition], persistent_success: bool) -> List[Transition]:
    updated: List[Transition] = []
    bonus = persistent_probe_reward(persistent_success)
    for item in transitions:
        item.reward += bonus
        item.persistent_success = int(persistent_success)
        updated.append(item)
    return updated
