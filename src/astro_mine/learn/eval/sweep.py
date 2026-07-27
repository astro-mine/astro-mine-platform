"""Seed sweeps with variance + single-seed rejection (RM-P1-LEARN-06).

A single lucky seed is an anti-pattern (learn.md §2 principle 8, §8): a reported score must
sweep multiple seeds and carry its **variance**. :func:`seed_sweep` wraps the
:func:`~astro_mine.learn.bench.evaluate` rollout seam, **rejecting** any request with fewer
than ``min_seeds`` distinct seeds (a ``ValueError``), and reports the return distribution's
mean/std plus the eval throughput, wall-clock, and — when a training
:class:`~astro_mine.learn.bench.ReferenceReport` is supplied — a sample-efficiency scalar
alongside reward (issue AC: "sample-efficiency and wall-clock reported alongside reward").

The single-seed guard lives **here**, not in :func:`~astro_mine.learn.bench.evaluate`: the
lower-level evaluate seam is deliberately callable single-seed (it records one comms-stress
ledger per seed); honesty is a policy the eval harness enforces on top of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.learn.algos.policy import LearnedPolicy
from astro_mine.learn.bench.reference import EnvFactory, EvalReport, ReferenceReport, evaluate
from astro_mine.learn.train.executor import RewardFn

__all__ = ["PolicyUnderTest", "SweepReport", "sample_efficiency", "seed_sweep"]

#: The default honest-eval floor: a reported score sweeps at least two distinct seeds.
DEFAULT_MIN_SEEDS = 2


@runtime_checkable
class PolicyUnderTest(Protocol):
    """Anything scoreable by the harness: it maps flat per-agent observations to action
    samples (the :meth:`~astro_mine.learn.algos.policy.LearnedPolicy.act` contract).

    A live trained/reference :class:`~astro_mine.learn.algos.policy.LearnedPolicy` satisfies
    it directly; an exported ONNX ``PolicyPackage`` is adapted into one by
    :func:`~astro_mine.learn.eval.onnx.onnx_policy_under_test` — the same rollout scores
    both."""

    def act(
        self, flat_obs: Mapping[AgentId, NDArray[np.float32]]
    ) -> Mapping[AgentId, Mapping[str, Any]]: ...


@dataclass(frozen=True)
class SweepReport:
    """A held-out seed sweep: the return distribution with variance, plus the honest
    cost/efficiency metrics reported alongside reward."""

    seeds: tuple[int, ...]
    returns: tuple[float, ...]
    mean_return: float
    std_return: float
    eval_throughput_steps_per_s: float
    wall_clock_s: float
    #: Area under a training learning curve — ``None`` for a pure held-out eval of a policy
    #: whose training curve was not supplied.
    sample_efficiency: float | None
    #: The underlying per-seed :class:`~astro_mine.learn.bench.EvalReport` (carries the
    #: per-seed comms-stress ledgers the comms-stress curve reads).
    report: EvalReport


def sample_efficiency(reference: ReferenceReport) -> float:
    """A sample-efficiency scalar from a training run's learning curve.

    The trapezoidal area under ``reference.learning_curve`` — higher means the baseline
    reached higher return sooner over the same budget (learn.md §8 "measure before
    optimizing"). Computed without NumPy's deprecated ``trapz`` so it is stable across the
    pinned NumPy line."""
    curve = reference.learning_curve
    if not curve:
        return 0.0
    if len(curve) == 1:
        return float(curve[0])
    return float(sum((curve[i] + curve[i + 1]) / 2.0 for i in range(len(curve) - 1)))


def seed_sweep(
    policy: PolicyUnderTest,
    env_factory: EnvFactory,
    seeds: Sequence[int],
    *,
    steps: int = 64,
    reward_fn: RewardFn | None = None,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    reference: ReferenceReport | None = None,
) -> SweepReport:
    """Sweep ``policy`` over held-out ``seeds`` (one fresh env each) and report mean/variance.

    Raises :class:`ValueError` if fewer than ``min_seeds`` *distinct* seeds are given — a
    single-seed result is rejected as an anti-pattern (learn.md §2, §8). Otherwise delegates
    the rollout to :func:`~astro_mine.learn.bench.evaluate` (the same
    :class:`~astro_mine.learn.train.executor.LocalExecutor` path training uses), measures the
    sweep wall-clock, and folds in ``reference``'s sample-efficiency when supplied."""
    distinct = {int(s) for s in seeds}
    if len(distinct) < min_seeds:
        raise ValueError(
            f"honest evaluation needs >= {min_seeds} distinct seeds; got {sorted(distinct)} — "
            "a single-seed result is an anti-pattern (learn.md §2, §8)"
        )
    start = perf_counter()
    report = evaluate(
        cast(LearnedPolicy, policy), env_factory, seeds, reward_fn=reward_fn, steps=steps
    )
    wall_clock_s = perf_counter() - start
    return SweepReport(
        seeds=report.seeds,
        returns=report.returns,
        mean_return=report.mean_return,
        std_return=report.std_return,
        eval_throughput_steps_per_s=report.throughput_steps_per_s,
        wall_clock_s=wall_clock_s,
        sample_efficiency=sample_efficiency(reference) if reference is not None else None,
        report=report,
    )
