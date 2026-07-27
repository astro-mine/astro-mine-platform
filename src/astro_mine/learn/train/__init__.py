"""Trainer orchestration: the rollout executor seam + the Ray RLlib scale-out path.

- :class:`RolloutExecutor` / :class:`LocalExecutor` / :class:`KubeRayExecutor` — the single
  call site RM-P1-LEARN-04 swaps local ⇄ distributed behind (Torch-free; always importable).
- ``astro_mine.learn.train.rllib`` — the Ray RLlib (PyTorch) config path (learn.md §11);
  imported separately because it requires the optional ``[rllib]`` extra.
"""

from __future__ import annotations

from astro_mine.learn.train.executor import (
    AgentStepFn,
    BroadcastableStep,
    EnvFactory,
    KubeRayExecutor,
    LocalExecutor,
    RewardFn,
    Rollout,
    RolloutExecutor,
    RolloutStep,
    StepDecision,
    WorkerBatch,
    WorkerResult,
    derive_seeds,
    fan_out_rollout,
    in_process_worker_batch,
)

__all__ = [
    "AgentStepFn",
    "BroadcastableStep",
    "EnvFactory",
    "KubeRayExecutor",
    "LocalExecutor",
    "RewardFn",
    "Rollout",
    "RolloutExecutor",
    "RolloutStep",
    "StepDecision",
    "WorkerBatch",
    "WorkerResult",
    "derive_seeds",
    "fan_out_rollout",
    "in_process_worker_batch",
]
