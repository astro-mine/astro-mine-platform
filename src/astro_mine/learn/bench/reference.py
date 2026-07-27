"""Reproducible reference-score + throughput harness, and the honest-eval seam (LEARN-03).

Every baseline ships a **reproducible reference score + throughput benchmark** (issue AC;
learn.md §8 "measure before optimizing"). This module provides:

- :func:`evaluate` — the **RM-P1-LEARN-06 honest-eval seam**: it separates *training* from
  *held-out evaluation* envs, sweeps seeds, reports mean **and variance** (a single lucky
  seed is an anti-pattern; learn.md §8), and exposes a comms-stress hook (the env's
  ``comms_report()`` ledger). Only the single reproducible reference score is shipped here;
  the full comms-stress sweep is RM-P1-LEARN-06.
- :func:`reference_score` — trains a baseline briefly, then reports its learning curve,
  training throughput (env-steps/sec), and a held-out evaluation.

**CI-smoke vs. real-anchor split (documented, deliberate).** The FakeSwarmWorld reference
(``@pytest`` default) exercises the *loop and reproducibility* on a reward-free fake; the
**real anchor lunar-polar-prospecting reference score needs Sim** (un-importable from Learn)
and is a ``slow``-marked, out-of-CI artifact. CI runs only the FakeSwarmWorld smoke reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from time import perf_counter

from astro_mine.core.env.model import AgentId
from astro_mine.learn.algos._contract import Algorithm
from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.algos.policy import LearnedPolicy
from astro_mine.learn.envs import SwarmEnv
from astro_mine.learn.train.executor import (
    LocalExecutor,
    RewardFn,
    StepDecision,
    default_reward_fn,
)

__all__ = ["EvalReport", "ReferenceReport", "evaluate", "reference_score"]

#: A factory for fresh evaluation envs — held-out envs differ from the training env
#: (different world seed / scenario), the honest-eval separation.
EnvFactory = Callable[[], SwarmEnv]


@dataclass(frozen=True)
class EvalReport:
    """Held-out evaluation across a seed sweep, with variance and comms stress."""

    seeds: tuple[int, ...]
    returns: tuple[float, ...]
    mean_return: float
    std_return: float
    throughput_steps_per_s: float
    #: Per-seed comms-budget ledgers (``SwarmEnv.comms_report()``) — the LEARN-06 denominator.
    comms_stress: tuple[Mapping[AgentId, Mapping[str, float]], ...] = ()


@dataclass(frozen=True)
class ReferenceReport:
    """A baseline's reproducible reference score + throughput benchmark."""

    algorithm: str
    learning_curve: tuple[float, ...]
    train_throughput_steps_per_s: float
    evaluation: EvalReport
    config: TrainConfig = field(default_factory=TrainConfig)


def evaluate(
    policy: LearnedPolicy,
    env_factory: EnvFactory,
    seeds: Sequence[int],
    *,
    reward_fn: RewardFn | None = None,
    steps: int = 64,
) -> EvalReport:
    """Roll ``policy`` (greedily) through fresh held-out envs, one per seed, and report the
    return distribution + throughput + comms stress.

    Uses the same :class:`~astro_mine.learn.train.executor.LocalExecutor` as training so the
    eval path is identical infrastructure. Reproducible: fixed policy + fixed seeds ⇒ fixed
    report."""
    executor = LocalExecutor()
    shaped = reward_fn if reward_fn is not None else default_reward_fn

    def agent_step(flat_obs: Mapping[AgentId, object]) -> Mapping[AgentId, StepDecision]:
        return {
            agent: StepDecision(action_sample=sample)
            for agent, sample in policy.act(flat_obs).items()  # type: ignore[arg-type]
        }

    returns: list[float] = []
    comms: list[Mapping[AgentId, Mapping[str, float]]] = []
    total_steps = 0
    start = perf_counter()
    for seed in seeds:
        env = env_factory()
        rollout = executor.rollout(env, agent_step, steps=steps, seed=seed, reward_fn=shaped)
        returns.append(sum(r for step in rollout.steps for r in step.reward.values()))
        comms.append(env.comms_report())
        total_steps += rollout.env_steps()
    elapsed = perf_counter() - start
    return EvalReport(
        seeds=tuple(seeds),
        returns=tuple(returns),
        mean_return=fmean(returns) if returns else 0.0,
        std_return=pstdev(returns) if len(returns) > 1 else 0.0,
        throughput_steps_per_s=(total_steps / elapsed) if elapsed > 0 else 0.0,
        comms_stress=tuple(comms),
    )


def reference_score(
    algorithm: Algorithm,
    env_factory: EnvFactory,
    config: TrainConfig | None = None,
    *,
    eval_seeds: Sequence[int] = (100, 101, 102),
    reward_fn: RewardFn | None = None,
) -> ReferenceReport:
    """Train ``algorithm`` briefly on a training env, then evaluate on held-out seeds.

    Returns the learning curve, training throughput, and the held-out
    :class:`EvalReport`. On the reward-free FakeSwarmWorld this measures the loop +
    reproducibility (the CI smoke reference); the real anchor-scenario score needs Sim and is
    the out-of-CI ``slow`` artifact."""
    cfg = config if config is not None else TrainConfig()
    trainer = algorithm.make_trainer(env_factory(), cfg)
    start = perf_counter()
    metrics = [trainer.train_iteration() for _ in range(cfg.iterations)]
    elapsed = perf_counter() - start
    total_steps = sum(int(m.get("env_steps", 0.0)) for m in metrics)
    policy = trainer.policy()
    assert isinstance(policy, LearnedPolicy)  # the baselines produce LearnedPolicy
    evaluation = evaluate(policy, env_factory, eval_seeds, reward_fn=reward_fn)
    return ReferenceReport(
        algorithm=trainer.spec.name,
        learning_curve=tuple(trainer.learning_curve()),
        train_throughput_steps_per_s=(total_steps / elapsed) if elapsed > 0 else 0.0,
        evaluation=evaluation,
        config=cfg,
    )
