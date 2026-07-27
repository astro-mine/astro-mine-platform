"""GPU-vectorized rollout tier (RM-P1-LEARN-04; learn.md §3 module map, §8).

The **third** rollout-executor choice — after the in-process :class:`LocalExecutor` and the
distributed :class:`~astro_mine.learn.train.executor.KubeRayExecutor` — and the one learn.md §8
names first among the throughput strategies: "GPU-vectorized envs (JAX/Brax / Surrogate tiers) —
thousands of envs resident on one GPU". It is a fidelity tier Learn *selects* per curriculum
stage (``TrainConfig.fidelity == "gpu_vectorized"``), not a trainer fork: rollouts still reach
the world only through the Core Environment contract and return the same
:class:`~astro_mine.learn.train.executor.Rollout` shape (surrogate.md §2 — "the surrogate/vector
tier is Sim's decision, not a sibling import").

- :class:`VectorExecutor` — the seam. Runs the **batched kernel** when a :class:`BatchedWorld`
  is available and the policy step is a
  :class:`~astro_mine.learn.train.executor.BatchedStep`, and degrades **gracefully** to a
  sequential CPU loop otherwise (no GPU, no JAX, tier-1 workstation — which learn.md §7 says
  MUST always work). :attr:`VectorExecutor.backend` reports which path it actually took.
- :class:`BatchedWorld` / :func:`batched_rollout` — the batched-world protocol and the lockstep
  kernel: one world step and one policy forward per agent per tick, for the *whole* batch.
  **Sim's Brax/MJX GPU tier plugs in here**, behind the same protocol, without Learn importing
  Sim.
- :class:`JaxBatchedWorld` — Learn's reference JAX/XLA realization (the ``[jax]`` extra): one
  jit-compiled, vmapped program over a device-resident batch. It makes the seam real, backs the
  ``gpu``-marked test, and provides the throughput benchmark
  (:mod:`~astro_mine.learn.envs.vector.benchmark`) — it is deliberately *not* a physics model.
"""

from __future__ import annotations

from astro_mine.learn.envs.vector.batched import (
    BatchedObservation,
    BatchedTransition,
    BatchedWorld,
    batched_rollout,
    peers_from_mask,
)
from astro_mine.learn.envs.vector.jax_world import (
    JaxBatchedWorld,
    accelerator_devices,
    jax_available,
    jax_batched_world_factory,
)
from astro_mine.learn.envs.vector.vector import Backend, BatchedWorldFactory, VectorExecutor

__all__ = [
    "Backend",
    "BatchedObservation",
    "BatchedTransition",
    "BatchedWorld",
    "BatchedWorldFactory",
    "JaxBatchedWorld",
    "VectorExecutor",
    "accelerator_devices",
    "batched_rollout",
    "jax_available",
    "jax_batched_world_factory",
    "peers_from_mask",
]
