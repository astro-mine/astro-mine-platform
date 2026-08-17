# SPDX-License-Identifier: Apache-2.0
"""Ray/KubeRay rollout-worker batch for :class:`~astro_mine.learn.train.executor.KubeRayExecutor`
(RM-P1-LEARN-04; learn.md §2.1, §7 tier 2).

The Ray-dependent half of the distributed executor — the ``WorkerBatch`` implementation that
runs each worker's rollout on a Ray actor pool — kept in its own module behind a top-level
``import ray`` so :mod:`astro_mine.learn.train.executor` stays Ray-free-importable (the same
isolation ``train/rllib.py`` uses). It is imported only when a :class:`KubeRayExecutor` with
the default (Ray) runner actually rolls out. The executor's *orchestration* — broadcast,
per-worker seeding, fixed-order concatenation, RNG write-back — lives in ``executor.py`` and
is shared with the Ray-free :func:`~astro_mine.learn.train.executor.in_process_worker_batch`,
so the correctness/determinism gates run deterministically without a cluster (a real
cluster's autoscaling/preemption/gang-scheduling is Cloud's, cloud.md §7, and unverifiable in
CI regardless).

The learner-worker split (on-policy): the learner broadcasts its per-agent weights + net
architecture + sampling-RNG state **once** into the object store; each :func:`_worker_rollout`
task rebuilds a byte-equivalent step and runs the **unchanged** :class:`LocalExecutor` loop
over an env it builds from the factory. No durable worker state — everything a preemptible
worker needs is re-shipped each iteration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import ray

from astro_mine.learn.train.executor import LocalExecutor, Rollout, WorkerResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from astro_mine.learn.train.executor import AgentStepFn, EnvFactory, RewardFn

__all__ = ["run_worker_batch"]


def _worker_rollout(
    rebuild: Any,
    state: Any,
    env_factory: EnvFactory,
    *,
    steps: int,
    seed: int,
    reward_fn: RewardFn | None,
) -> tuple[Rollout, Any]:
    """One rollout worker: rebuild the broadcast step, run the exact LocalExecutor loop over
    a freshly built env, and return the trajectory plus the step's advanced RNG state.

    Pins single-threaded, deterministic Torch in the worker process (the driver already does
    so via ``seed_everything``) so a worker's float results are byte-identical to the
    workstation's — the property the CX-REPRO determinism gate rests on."""
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    step = rebuild(state)
    env = env_factory()
    rollout = LocalExecutor().rollout(env, step, steps=steps, seed=seed, reward_fn=reward_fn)
    return rollout, step.rng_state()


#: The Ray task handle for :func:`_worker_rollout`. ``ray.remote``'s typed overloads model
#: only positional-arg task signatures, not our keyword-only one, so register + submit through
#: an untyped handle (``ray.*`` is untyped glue here, per the pyproject mypy override).
_worker_rollout_task: Any = ray.remote(_worker_rollout)  # type: ignore[arg-type]


def run_worker_batch(
    rebuild: Callable[[Any], AgentStepFn],
    state: Any,
    env_factory: EnvFactory,
    seeds: list[int],
    *,
    steps: int,
    reward_fn: RewardFn | None,
    ray_address: str | None,
    ray_init_kwargs: Mapping[str, Any],
) -> list[WorkerResult]:
    """Run one rollout worker per seed on a Ray actor pool; return results in seed order.

    The Ray realization of :class:`~astro_mine.learn.train.executor.WorkerBatch`. Connects to
    an already-running Ray session when present (the KubeRay ``RayJob`` case) and only
    bootstraps — and tears down — a session it started itself, so it composes with Cloud's
    orchestration without owning Ray's lifecycle. The weight payload is broadcast **once** via
    the object store (``ray.put``); Ray auto-dereferences the ObjectRef into each task."""
    started_here = False
    if not ray.is_initialized():
        ray.init(address=ray_address, **ray_init_kwargs)
        started_here = True
    try:
        state_ref = ray.put(state)
        futures = [
            _worker_rollout_task.remote(
                rebuild,
                state_ref,
                env_factory,
                steps=steps,
                seed=worker_seed,
                reward_fn=reward_fn,
            )
            for worker_seed in seeds
        ]
        results: list[WorkerResult] = ray.get(futures)  # fixed worker order preserved
    finally:
        if started_here:
            ray.shutdown()
    return results
