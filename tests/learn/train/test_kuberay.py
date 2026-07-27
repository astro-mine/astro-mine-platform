"""The KubeRay distributed executor (RM-P1-LEARN-04) — needs the [rllib] extra (Torch).

"The distributed path is the same code with a different executor, never a fork"
(learn.md §2.1). The executor's *orchestration* — broadcast the learner's weights once,
run each worker via a pluggable ``WorkerBatch``, concatenate in fixed worker order, write the
worker-0 RNG state back — is identical whether the workers run on a Ray actor pool or
in-process. So the correctness/determinism gates run through the Ray-free
``in_process_worker_batch`` runner: deterministic, fast, and immune to real-cluster timing
(whose autoscaling/preemption is unverifiable in CI anyway, cloud.md §7):

- **The cluster reproduces the workstation:** a single-worker rollout is byte-identical to
  :class:`LocalExecutor`, and a full single-worker training run reproduces the LocalExecutor
  learning curve exactly (the RNG write-back keeps the learner in lockstep) — for ippo/mappo/qmix.
- **Topology-fixed reproducibility:** a fixed ``(num_workers, seed)`` reproduces run-to-run;
  more workers collect strictly more experience (batch composition is per-topology).

One ``cluster``-marked smoke test drives the real Ray actor-pool runner end-to-end and asserts
it agrees with the in-process runner — proving the Ray plumbing. It is deselected in CI (Ray
worker bootstrap is unreliable on hosted runners) and run on demand; the CI gate rides the
deterministic in-process tests above.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from astro_mine.learn import SwarmEnv, TrainConfig, default_registry, make_swarm_env
from astro_mine.learn.train import (
    KubeRayExecutor,
    LocalExecutor,
    Rollout,
    in_process_worker_batch,
)
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _factory() -> SwarmEnv:
    return make_swarm_env(FakeSwarmWorld(), build_assets())


def _in_process_executor(num_workers: int) -> KubeRayExecutor:
    # The distributed executor with the Ray-free worker runner injected — same orchestration,
    # no cluster (KubeRayExecutor(...) with the default runner is the real Ray path).
    return KubeRayExecutor(_factory, num_workers=num_workers, runner=in_process_worker_batch)


def _signature(rollout: Rollout) -> list[object]:
    """A byte-comparable projection of a rollout: per-step observations, decisions, rewards,
    and the global CTDE state — everything the trainer builds tensors from."""
    return [
        (
            sorted(step.obs),
            [float(v) for a in sorted(step.obs) for v in step.obs[a].tolist()],
            [step.log_prob[a] for a in sorted(step.obs)],
            [step.value[a] for a in sorted(step.obs)],
            [step.reward[a] for a in sorted(step.obs)],
            [str(step.action[a]) for a in sorted(step.obs)],
            step.state.tolist(),
        )
        for step in rollout.steps
    ]


@pytest.mark.parametrize("tag", ["ippo", "qmix"])
def test_single_worker_rollout_is_byte_identical_to_local(tag: str) -> None:
    config = TrainConfig(seed=3, rollout_steps=8, hidden_sizes=(16, 16), epsilon=0.2)
    # Two trainers, same seed ⇒ byte-identical initial weights + generator (seed_everything).
    local_trainer = default_registry().get(tag).make_trainer(_factory(), config)
    dist_trainer = default_registry().get(tag).make_trainer(_factory(), config)

    local = LocalExecutor().rollout(_factory(), local_trainer._step, steps=8, seed=3)
    dist = _in_process_executor(1).rollout(_factory(), dist_trainer._step, steps=8, seed=3)
    assert _signature(local) == _signature(dist)


@pytest.mark.parametrize("tag", ["ippo", "mappo", "qmix"])
def test_single_worker_training_reproduces_the_local_learning_curve(tag: str) -> None:
    # The strongest form of the "cluster reproduces the workstation" gate: the *same trainer
    # code* run through the distributed executor (single worker) yields the identical learning
    # curve to the in-process LocalExecutor, iteration for iteration — the RNG write-back
    # advances the learner generator in lockstep.
    config = TrainConfig(seed=11, iterations=3, rollout_steps=6, hidden_sizes=(16, 16))
    local_trainer = default_registry().get(tag).make_trainer(_factory(), config)
    local_curve = [local_trainer.train_iteration() for _ in range(config.iterations)]

    dist_trainer = (
        default_registry()
        .get(tag)
        .make_trainer(_factory(), config, executor=_in_process_executor(1))
    )
    dist_curve = [dist_trainer.train_iteration() for _ in range(config.iterations)]
    assert local_curve == dist_curve


def test_fixed_multi_worker_topology_is_reproducible() -> None:
    config = TrainConfig(seed=7, rollout_steps=6, hidden_sizes=(16, 16))

    def run() -> list[object]:
        trainer = default_registry().get("ippo").make_trainer(_factory(), config)
        return _signature(
            _in_process_executor(2).rollout(_factory(), trainer._step, steps=6, seed=7)
        )

    assert run() == run()  # a fixed (num_workers, seed) reproduces run-to-run


def test_more_workers_collect_more_experience() -> None:
    config = TrainConfig(seed=9, rollout_steps=6, hidden_sizes=(16, 16))
    one = default_registry().get("ippo").make_trainer(_factory(), config)
    two = default_registry().get("ippo").make_trainer(_factory(), config)

    single = _in_process_executor(1).rollout(_factory(), one._step, steps=6, seed=9)
    double = _in_process_executor(2).rollout(_factory(), two._step, steps=6, seed=9)
    # Each worker runs a full independent rollout; two workers ⇒ strictly more collected steps.
    assert double.env_steps() > single.env_steps()
    # Worker 0 always uses the run seed verbatim, so its slice is exactly the single-worker run.
    assert _signature(double)[: len(single.steps)] == _signature(single)


def test_kuberay_rejects_a_non_broadcastable_step() -> None:
    # A bare callable cannot be reconstructed on a worker; rejected before any run.
    with pytest.raises(TypeError, match=r"BroadcastableStep|broadcast"):
        _in_process_executor(1).rollout(_factory(), lambda flat: {}, steps=1, seed=0)


@pytest.mark.ray
@pytest.mark.cluster
def test_real_ray_actor_pool_agrees_with_in_process_runner() -> None:
    # The one end-to-end check that the real Ray actor-pool runner works: it must agree with
    # the deterministic in-process runner for the same step + seed. Ray 2.56 removed local_mode,
    # so this spins up a real Ray cluster (worker processes); correctness/repro only (not
    # autoscaling). ``cluster``-marked and deselected in CI — Ray worker bootstrap is unreliable
    # on hosted runners (RaySystemError: failed to startup worker); run on demand locally.
    ray = pytest.importorskip("ray")
    config = TrainConfig(seed=3, rollout_steps=6, hidden_sizes=(16, 16))
    in_proc_trainer = default_registry().get("ippo").make_trainer(_factory(), config)
    ray_trainer = default_registry().get("ippo").make_trainer(_factory(), config)

    reference = _in_process_executor(1).rollout(_factory(), in_proc_trainer._step, steps=6, seed=3)
    started = ray.is_initialized()
    if not started:
        ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    try:
        on_ray = KubeRayExecutor(_factory, num_workers=1).rollout(
            _factory(), ray_trainer._step, steps=6, seed=3
        )
    finally:
        if not started:
            ray.shutdown()
    assert _signature(on_ray) == _signature(reference)
