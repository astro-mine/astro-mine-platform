# SPDX-License-Identifier: Apache-2.0
"""Rollout-throughput benchmark: batched GPU kernel vs. the sequential CPU loop (LEARN-04).

"Every baseline ships a reproducible throughput benchmark" (learn.md §8, conventions.md §8
"measure before optimizing"), and **sample-throughput, not FLOPs, is the headline metric**
(learn.md §8). This module is that benchmark for the rollout tier: it runs the *same* policy
step over the *same* number of env copies through both
:class:`~astro_mine.learn.envs.vector.VectorExecutor` backends and reports env-steps/second.

The comparison it makes is the honest one — **batched kernel vs. sequential loop**, the axis
RM-P1-LEARN-04 actually changes. It reports the device the batched kernel ran on
(``cpu``/``gpu``/``tpu``), because "GPU-vectorized" measured on an XLA-CPU fallback is a
throughput claim nobody should believe: the speedup on the CPU wheel comes from batching alone,
and the accelerator multiplies it.

Run it from a checkout::

    uv run python -m astro_mine.learn.envs.vector.benchmark --num-envs 64 --steps 64

which is exactly what the ``gpu``-marked test and the README's recorded numbers come from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from astro_mine.learn.envs.vector.jax_world import (
    accelerator_devices,
    jax_available,
    jax_batched_world_factory,
)
from astro_mine.learn.envs.vector.vector import VectorExecutor
from astro_mine.learn.train.executor import AgentStepFn, EnvFactory

__all__ = ["BenchmarkResult", "ThroughputReport", "baseline_step", "main", "rollout_throughput"]


@dataclass(frozen=True)
class ThroughputReport:
    """One backend's measured rollout throughput (env-steps/second, summed over live agents)."""

    backend: str
    num_envs: int
    steps: int
    env_steps: int
    wall_clock_s: float
    device: str = "cpu"
    #: Set on the batched report: how many times faster it collected experience than the loop.
    speedup_vs_sequential: float | None = None

    @property
    def steps_per_s(self) -> float:
        return self.env_steps / self.wall_clock_s if self.wall_clock_s > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "num_envs": self.num_envs,
            "steps": self.steps,
            "env_steps": self.env_steps,
            "wall_clock_s": round(self.wall_clock_s, 4),
            "env_steps_per_s": round(self.steps_per_s, 1),
            "speedup_vs_sequential": (
                None if self.speedup_vs_sequential is None else round(self.speedup_vs_sequential, 2)
            ),
        }


@dataclass
class BenchmarkResult:
    """Both backends' reports plus the environment they were measured in."""

    sequential: ThroughputReport
    batched: ThroughputReport | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequential_cpu": self.sequential.to_dict(),
            "batched": None if self.batched is None else self.batched.to_dict(),
            "jax_available": jax_available(),
            "accelerators": accelerator_devices(),
            "notes": self.notes,
        }


def _measure(
    executor: VectorExecutor,
    env_factory: EnvFactory,
    agent_step: AgentStepFn,
    *,
    steps: int,
    seed: int,
    num_envs: int,
    device: str,
) -> ThroughputReport:
    # One untimed rollout first: it pays the XLA jit-compilation cost (and warms the Torch
    # allocator), which would otherwise be charged to the batched backend as if it were
    # per-step work. Compilation is amortized over a whole training run, not one rollout.
    executor.rollout(env_factory(), agent_step, steps=steps, seed=seed)
    start = perf_counter()
    rollout = executor.rollout(env_factory(), agent_step, steps=steps, seed=seed)
    elapsed = perf_counter() - start
    return ThroughputReport(
        backend=executor.backend,
        num_envs=num_envs,
        steps=steps,
        env_steps=rollout.env_steps(),
        wall_clock_s=elapsed,
        device=device,
    )


def rollout_throughput(
    env_factory: EnvFactory,
    agent_step: AgentStepFn,
    *,
    num_envs: int = 32,
    steps: int = 32,
    seed: int = 0,
) -> BenchmarkResult:
    """Measure both rollout backends on the same policy step and env copies.

    Returns the sequential-CPU report always, and the batched report when the ``[jax]`` extra is
    installed (with the batched world resident on an accelerator if one is visible). The
    ``speedup_vs_sequential`` on the batched report is the number RM-P1-LEARN-04 is judged by."""
    sequential = _measure(
        VectorExecutor(env_factory, num_envs=num_envs, backend="cpu"),
        env_factory,
        agent_step,
        steps=steps,
        seed=seed,
        num_envs=num_envs,
        device="cpu",
    )
    result = BenchmarkResult(sequential=sequential)

    if not jax_available():
        result.notes.append(
            "the [jax] extra is not installed — only the sequential CPU loop was measured; "
            "install it with `uv sync --extra jax` to measure the batched kernel"
        )
        return result

    accelerators = accelerator_devices()
    device = accelerators[0] if accelerators else "cpu"
    executor = VectorExecutor(
        env_factory,
        num_envs=num_envs,
        batched_world=jax_batched_world_factory(env_factory, num_envs=num_envs, horizon=steps + 1),
        backend="jax",
    )
    batched = _measure(
        executor,
        env_factory,
        agent_step,
        steps=steps,
        seed=seed,
        num_envs=num_envs,
        device=device,
    )
    speedup = batched.steps_per_s / sequential.steps_per_s if sequential.steps_per_s > 0 else None
    result.batched = ThroughputReport(
        backend=batched.backend,
        num_envs=batched.num_envs,
        steps=batched.steps,
        env_steps=batched.env_steps,
        wall_clock_s=batched.wall_clock_s,
        device=device,
        speedup_vs_sequential=speedup,
    )
    if device == "cpu":
        result.notes.append(
            "no accelerator visible: the batched kernel ran the identical jit/vmap XLA program "
            "on CPU, so this speedup is from BATCHING alone — a GPU multiplies it"
        )
    return result


def baseline_step(
    env_factory: EnvFactory, *, algorithm: str = "mappo", hidden: tuple[int, ...] = (32, 32)
) -> AgentStepFn:
    """A **real** baseline's rollout step, for benchmarking the executor rather than a toy.

    Builds the named baseline's trainer over ``env_factory`` and hands back its live rollout
    step (:attr:`PpoTrainer.rollout_step`) — a genuine batched Torch forward with the trained
    net shapes, so the measured env-steps/second is what a training rollout actually achieves,
    not what a numpy stub would."""
    from astro_mine.learn.algos import TrainConfig, default_registry

    trainer = (
        default_registry()
        .get(algorithm)
        .make_trainer(env_factory(), TrainConfig(hidden_sizes=hidden))
    )
    step: AgentStepFn = trainer.rollout_step  # type: ignore[attr-defined]
    return step


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI dispatch
    # The argv from this module's own docstring. `astro-mine learn` takes flags, not verbs, so
    # there is no `learn vector benchmark` to name — this tier is run as a module, not typed.
    parser = argparse.ArgumentParser(
        prog="python -m astro_mine.learn.envs.vector.benchmark", description=__doc__
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm", default="mappo", help="baseline whose step is benchmarked")
    parser.add_argument(
        "--env-factory",
        required=True,
        help="importable 'module:attr' zero-arg SwarmEnv factory",
    )
    args = parser.parse_args(argv)

    from astro_mine.learn.train.run import resolve_env_factory

    env_factory = resolve_env_factory(args.env_factory)
    result = rollout_throughput(
        env_factory,
        baseline_step(env_factory, algorithm=args.algorithm),
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI dispatch
    sys.exit(main())
