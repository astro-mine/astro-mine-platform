# SPDX-License-Identifier: Apache-2.0
"""Sibling client seam for the design loop (studio.md §3 ``orchestrate/clients``, §6).

**Studio computes nothing** (studio.md §2 principle 1): each of the seven design-loop
steps is delegated to a sibling, and Studio speaks only Core contracts — the design loop
composes over Core's in-process ``Environment``/``Policy`` Protocols, with **no
sibling-package imports**. Because Core ``v0.1.0`` exposes those as in-process Python
Protocols (not gRPC services — none exist yet platform-wide), the client seam is a set of
typed Protocols with an injected implementation. A real deployment binds them to
Sim/Learn/Mind/Allocate/Guard/Bench (over gRPC/HTTP when those services exist); the
``Local*`` implementations here are deterministic, in-process, and let the whole loop run
and be contract-tested on the local tier with no siblings and no network (conventions.md
§7 — the local tier MUST work). Guard's certification is authoritative on the safety path.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from astro_mine.core.env import AgentId, ResetResult, StepResult
from astro_mine.core.env.protocol import Environment
from astro_mine.core.messages import ActionBatch, Observation
from astro_mine.core.objective import (
    MetricAggregation,
    MetricDirection,
    ObjectiveDocument,
    ObjectiveSpec,
    to_wire,
)
from astro_mine.core.policy import DecisionContext
from astro_mine.core.policy.protocol import Policy

from .._base import FrozenStudioModel
from ..hashing import content_hash, content_hash_json
from ..models import CandidateScore, DesignCandidate

_METRICS_KEY = "metrics"
_WORLD_AGENT = "world"


class GuardRejection(RuntimeError):
    """Guard refused to certify a candidate — the safety path is authoritative and the
    candidate is not simulated or scored (studio.md §6 step 5, §9)."""


class EpisodeResult(FrozenStudioModel):
    """A simulated rollout's per-metric samples, ready for scoring. Studio-owned: a thin
    hand-off between the (delegated) Simulator and Scorer steps."""

    world_ref: str
    sim_time_s: float
    steps: int
    metric_samples: dict[str, list[float]]


def objective_content_hash(objective: ObjectiveSpec) -> str:
    """Content address of an objective — the byte-stable Core wire form of its document."""
    document = ObjectiveDocument(objective_version="0.1", objective=objective)
    return content_hash(to_wire(document))


# --------------------------------------------------------------------------- #
# The seven client Protocols (one per design-loop step)
# --------------------------------------------------------------------------- #


@runtime_checkable
class SwarmComposer(Protocol):
    """Step 1 — compose an SADF swarm from Fleet assets → the agent ids."""

    def compose(self, candidate: DesignCandidate) -> tuple[AgentId, ...]: ...


@runtime_checkable
class EnvironmentFactory(Protocol):
    """Step 2 — instantiate the world via the Environment API (Worlds/Prospect/Link,
    implemented by Sim). Returns the Core ``Environment`` and a content-addressed
    ``world_ref``."""

    def instantiate(
        self,
        candidate: DesignCandidate,
        objective: ObjectiveSpec,
        *,
        agents: tuple[AgentId, ...],
        seed: int,
    ) -> tuple[Environment, str]: ...


@runtime_checkable
class PolicyConditioner(Protocol):
    """Steps 3 — obtain/condition policies via Learn (training) and Mind (planning)."""

    def condition(
        self, candidate: DesignCandidate, objective: ObjectiveSpec, *, seed: int
    ) -> Policy: ...


@runtime_checkable
class AllocatorClient(Protocol):
    """Step 4 — solve task assignment with Allocate (wraps/composes the policy)."""

    def allocate(
        self, policy: Policy, candidate: DesignCandidate, objective: ObjectiveSpec
    ) -> Policy: ...


@runtime_checkable
class GuardClient(Protocol):
    """Step 5 — wrap and certify with Guard. Raises :class:`GuardRejection` if the
    candidate cannot be certified (authoritative safety path)."""

    def certify(
        self, policy: Policy, candidate: DesignCandidate, objective: ObjectiveSpec
    ) -> Policy: ...


@runtime_checkable
class Simulator(Protocol):
    """Step 6 — simulate the candidate on Sim (multi-fidelity), producing metric
    samples."""

    def rollout(
        self,
        env: Environment,
        policy: Policy,
        objective: ObjectiveSpec,
        world_ref: str,
        *,
        seed: int,
        max_steps: int,
    ) -> EpisodeResult: ...


@runtime_checkable
class Scorer(Protocol):
    """Step 7 — score the candidate via Bench against the objective metrics."""

    def score(self, episode: EpisodeResult, objective: ObjectiveSpec) -> CandidateScore: ...


#: The evaluator identity a :class:`~astro_mine.studio.models.TradeStudy` records for the local tier
#: — the deterministic stand-in below, not sibling physics. A study's evaluator is part of its
#: identity, exactly as a scorecard's runner is Bench's (``REFERENCE_EPISODE_RUNNER_ID``,
#: bench.md §2.1): a stand-in-scored front and a physics-scored front must be distinguishable by
#: *provenance*, not only by their numbers, because their numbers look identical (studio.md §2;
#: gap report G1.1 — never let a stand-in look like the real thing).
LOCAL_STAND_IN_EVALUATOR_ID = "stand-in/0.1.0"


@dataclass(frozen=True)
class SiblingClients:
    """The injected bundle of the seven design-loop clients, plus the identity of whatever
    evaluated them."""

    composer: SwarmComposer
    environment: EnvironmentFactory
    conditioner: PolicyConditioner
    allocator: AllocatorClient
    guard: GuardClient
    simulator: Simulator
    scorer: Scorer
    #: Who produced the metric values — ``"stand-in/0.1.0"`` for the local tier, a sibling's own id
    #: (e.g. ``"astro-mine-sim/0.1.0"``) once real physics is wired. Required, with no default: a
    #: bundle that forgets to say what it is would publish a study that cannot be read honestly.
    evaluator: str


# --------------------------------------------------------------------------- #
# Deterministic in-process implementations (the local tier)
# --------------------------------------------------------------------------- #


def _metric_base(candidate: DesignCandidate, objective: ObjectiveSpec) -> dict[str, float]:
    """A deterministic, candidate-sensitive stand-in for sibling physics: each metric's
    base value scales with swarm size + the decision vector. Not domain-realistic — a
    reproducible stub so the loop, determinism gate, and scorer are exercisable offline."""
    swarm_size = float(sum(sel.count for sel in candidate.swarm))
    dv = float(sum(candidate.decision_vector.values()))
    bases: dict[str, float] = {}
    for criterion in objective.success_criteria:
        metric = criterion.binding.metric
        factor = float(sum(ord(ch) for ch in metric) % 7 + 1)
        bases[metric] = swarm_size * factor + dv
    return bases


class StubEnvironment:
    """A deterministic Core ``Environment``: metrics ride in ``StepResult.infos`` (no
    heavy ``Observation``/``StateSample`` payloads needed for a stub world)."""

    def __init__(
        self, agents: tuple[AgentId, ...], world_ref: str, seed: int, bases: Mapping[str, float]
    ) -> None:
        self._agents = agents
        self._world_ref = world_ref
        self._seed = seed
        self._bases = dict(bases)
        self._tick = 0

    @property
    def possible_agents(self) -> tuple[AgentId, ...]:
        return self._agents

    @property
    def agents(self) -> tuple[AgentId, ...]:
        return self._agents

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        self._tick = 0
        obs: dict[AgentId, Observation] = {}
        return ResetResult(observations=obs, infos={})

    def step(self, actions: ActionBatch) -> StepResult:
        self._tick += 1
        # Deterministic per-tick dither so samples carry non-zero spread (uncertainty is
        # shown, not hidden — studio.md §2 principle 7), while the mean stays reproducible.
        dither = ((self._tick * 7 + self._seed) % 5 - 2) / 100.0
        samples = {metric: base * (1.0 + dither) for metric, base in self._bases.items()}
        obs: dict[AgentId, Observation] = {}
        infos: dict[AgentId, Mapping[str, Any]] = {_WORLD_AGENT: {_METRICS_KEY: samples}}
        return StepResult(
            observations=obs,
            sim_time_s=float(self._tick),
            rewards={agent: 0.0 for agent in self._agents},
            terminations={agent: False for agent in self._agents},
            truncations={agent: False for agent in self._agents},
            infos=infos,
            dt_s=1.0,
        )


class StubPolicy:
    """A no-op Core ``Policy`` — the stub world needs no actions to advance."""

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        return ActionBatch(actions=[])


class LocalSwarmComposer:
    def compose(self, candidate: DesignCandidate) -> tuple[AgentId, ...]:
        agents: list[AgentId] = []
        for selection in candidate.swarm:
            for index in range(selection.count):
                agents.append(f"{selection.sadf_ref}#{index}")
        return tuple(agents)


class LocalEnvironmentFactory:
    def instantiate(
        self,
        candidate: DesignCandidate,
        objective: ObjectiveSpec,
        *,
        agents: tuple[AgentId, ...],
        seed: int,
    ) -> tuple[Environment, str]:
        world_ref = content_hash_json(
            {
                "candidate": candidate.digest(),
                "objective": objective_content_hash(objective),
                "seed": seed,
            }
        )
        env = StubEnvironment(agents, world_ref, seed, _metric_base(candidate, objective))
        return env, world_ref


class LocalPolicyConditioner:
    def condition(
        self, candidate: DesignCandidate, objective: ObjectiveSpec, *, seed: int
    ) -> Policy:
        return StubPolicy()


class LocalAllocator:
    def allocate(
        self, policy: Policy, candidate: DesignCandidate, objective: ObjectiveSpec
    ) -> Policy:
        return policy


class LocalGuard:
    """Certifies a candidate unless it is flagged unsafe (decision variable ``unsafe`` >
    0). The real Guard runs a ``SafetySpec`` certification (RFC-0004); the trigger here
    is a deterministic stand-in so the safety-reject path is testable."""

    def certify(
        self, policy: Policy, candidate: DesignCandidate, objective: ObjectiveSpec
    ) -> Policy:
        if candidate.decision_vector.get("unsafe", 0.0) > 0.0:
            raise GuardRejection(f"candidate {candidate.id} failed safety certification")
        return policy


class LocalSimulator:
    def rollout(
        self,
        env: Environment,
        policy: Policy,
        objective: ObjectiveSpec,
        world_ref: str,
        *,
        seed: int,
        max_steps: int,
    ) -> EpisodeResult:
        env.reset(seed=seed)
        samples: dict[str, list[float]] = {}
        sim_time = 0.0
        for _ in range(max_steps):
            actions = policy.decide(
                {},
                DecisionContext(
                    sim_time_s=sim_time, objective=objective, upstream=None, seed=seed, extras={}
                ),
            )
            result = env.step(actions)
            sim_time = result.sim_time_s
            metrics = result.infos.get(_WORLD_AGENT, {}).get(_METRICS_KEY, {})
            for metric, value in metrics.items():
                samples.setdefault(metric, []).append(float(value))
        return EpisodeResult(
            world_ref=world_ref, sim_time_s=sim_time, steps=max_steps, metric_samples=samples
        )


def _aggregate(values: list[float], how: MetricAggregation) -> float:
    if how is MetricAggregation.MEAN:
        return statistics.fmean(values)
    if how is MetricAggregation.MEDIAN:
        return statistics.median(values)
    if how is MetricAggregation.MIN:
        return min(values)
    if how is MetricAggregation.MAX:
        return max(values)
    if how is MetricAggregation.SUM:
        return float(sum(values))
    ordered = sorted(values)
    index = 4 if how is MetricAggregation.P05 else len(ordered) - 5
    index = max(0, min(len(ordered) - 1, index))
    return ordered[index]


class LocalScorer:
    """Aggregates episode samples per binding and evaluates pass/fail against each
    criterion's target±tolerance (soft) or threshold (hard)."""

    def score(self, episode: EpisodeResult, objective: ObjectiveSpec) -> CandidateScore:
        metric_scores: dict[str, float] = {}
        metric_uncertainty: dict[str, float] = {}
        satisfactions: list[tuple[float, float]] = []
        passed = True

        for criterion in objective.success_criteria:
            binding = criterion.binding
            values = episode.metric_samples.get(binding.metric, [])
            if not values:
                passed = False
                continue
            score = _aggregate(values, binding.aggregation)
            metric_scores[binding.metric] = score
            metric_uncertainty[binding.metric] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )

            if binding.threshold is not None:
                ok = (
                    score >= binding.threshold
                    if binding.direction is MetricDirection.HIGHER_BETTER
                    else score <= binding.threshold
                )
            else:
                ok = abs(score - binding.target) <= binding.tolerance
            if criterion.required and not ok:
                passed = False

            if binding.direction is MetricDirection.HIGHER_BETTER:
                satisfaction = min(1.0, score / binding.target) if binding.target > 0 else 1.0
            else:
                satisfaction = min(1.0, binding.target / score) if score > 0 else 1.0
            satisfactions.append(
                (criterion.weight if criterion.weight is not None else 1.0, satisfaction)
            )

        total_weight = sum(weight for weight, _ in satisfactions)
        aggregate = (
            sum(weight * value for weight, value in satisfactions) / total_weight
            if total_weight > 0
            else 0.0
        )
        return CandidateScore(
            objective_hash=objective_content_hash(objective),
            metric_scores=metric_scores,
            metric_uncertainty=metric_uncertainty,
            aggregate=aggregate,
            passed=passed,
        )


def local_clients() -> SiblingClients:
    """The default all-local :class:`SiblingClients` bundle (the working local tier)."""
    return SiblingClients(
        composer=LocalSwarmComposer(),
        environment=LocalEnvironmentFactory(),
        conditioner=LocalPolicyConditioner(),
        allocator=LocalAllocator(),
        guard=LocalGuard(),
        simulator=LocalSimulator(),
        scorer=LocalScorer(),
        evaluator=LOCAL_STAND_IN_EVALUATOR_ID,
    )
