"""STUDIO-03 — client seam: local siblings, Core-Protocol conformance, scoring."""

from __future__ import annotations

import pytest

from astro_mine.core.env import check_environment
from astro_mine.core.env.protocol import Environment
from astro_mine.core.objective import (
    MetricAggregation,
    MetricBinding,
    MetricDirection,
    ObjectiveSpec,
    SuccessCriterion,
)
from astro_mine.core.policy import DecisionContext
from astro_mine.core.policy.conformance import check_policy
from astro_mine.core.policy.protocol import Policy
from astro_mine.studio.models import AssetSelection, DesignCandidate
from astro_mine.studio.orchestrate.clients import (
    EpisodeResult,
    GuardRejection,
    LocalAllocator,
    LocalEnvironmentFactory,
    LocalGuard,
    LocalPolicyConditioner,
    LocalScorer,
    LocalSimulator,
    LocalSwarmComposer,
    StubEnvironment,
    StubPolicy,
    _aggregate,
    local_clients,
    objective_content_hash,
)


def _binding(**kw: object) -> MetricBinding:
    base: dict[str, object] = dict(
        metric="m", unit="kg", direction=MetricDirection.HIGHER_BETTER, target=10.0, tolerance=1.0
    )
    base.update(kw)
    return MetricBinding(**base)  # type: ignore[arg-type]


def _objective(*criteria: SuccessCriterion) -> ObjectiveSpec:
    return ObjectiveSpec(id="o", name="n", success_criteria=list(criteria))


# ---- Core-Protocol conformance (consumer-driven contract) ----------------- #


def test_stub_environment_and_policy_satisfy_core_protocols() -> None:
    env = StubEnvironment(("a-0",), "sha256:w", 0, {"m": 1.0})
    assert isinstance(env, Environment)
    assert isinstance(StubPolicy(), Policy)
    # The Core policy conformance harness accepts the stub behaviourally.
    actions = check_policy(
        StubPolicy(),
        {},
        DecisionContext(sim_time_s=0.0, objective=None, upstream=None, seed=0, extras={}),
    )
    assert actions.actions == []


def test_stub_environment_reset_step_and_agents() -> None:
    env = StubEnvironment(("a-0", "a-1"), "sha256:w", 3, {"m": 2.0})
    assert env.possible_agents == ("a-0", "a-1") == env.agents
    env.reset(seed=3)
    step = env.step(
        StubPolicy().decide(
            {}, DecisionContext(sim_time_s=0.0, objective=None, upstream=None, seed=3, extras={})
        )
    )
    assert step.sim_time_s == 1.0 and step.dt_s == 1.0
    assert step.infos["world"]["metrics"]["m"] != 0.0
    # Core's stricter behavioural harness expects real observations (a full Sim's job).
    with pytest.raises(Exception, match="observations"):
        check_environment(env, seed=3)


# ---- local sibling implementations ---------------------------------------- #


def test_local_swarm_composer_expands_counts() -> None:
    candidate = DesignCandidate(
        id="c",
        swarm=[
            AssetSelection(sadf_ref="rover", count=2),
            AssetSelection(sadf_ref="relay", count=1),
        ],
    )
    assert LocalSwarmComposer().compose(candidate) == ("rover#0", "rover#1", "relay#0")


def test_local_environment_factory_world_ref_is_content_addressed() -> None:
    candidate = DesignCandidate(id="c", swarm=[AssetSelection(sadf_ref="rover", count=1)])
    objective = _objective(SuccessCriterion(id="c", binding=_binding()))
    factory = LocalEnvironmentFactory()
    env1, ref1 = factory.instantiate(candidate, objective, agents=("rover#0",), seed=1)
    _env2, ref2 = factory.instantiate(candidate, objective, agents=("rover#0",), seed=1)
    _env3, ref3 = factory.instantiate(candidate, objective, agents=("rover#0",), seed=2)
    assert isinstance(env1, StubEnvironment)
    assert ref1 == ref2 and ref1 != ref3  # deterministic; seed-sensitive


def test_local_conditioner_allocator_guard() -> None:
    candidate = DesignCandidate(id="c", swarm=[AssetSelection(sadf_ref="rover", count=1)])
    objective = _objective(SuccessCriterion(id="c", binding=_binding()))
    policy = LocalPolicyConditioner().condition(candidate, objective, seed=0)
    assert LocalAllocator().allocate(policy, candidate, objective) is policy  # identity compose
    assert LocalGuard().certify(policy, candidate, objective) is policy
    unsafe = DesignCandidate(
        id="u", swarm=[AssetSelection(sadf_ref="rover", count=1)], decision_vector={"unsafe": 1.0}
    )
    with pytest.raises(GuardRejection, match="safety certification"):
        LocalGuard().certify(policy, unsafe, objective)


def test_local_simulator_collects_samples_per_step() -> None:
    candidate = DesignCandidate(id="c", swarm=[AssetSelection(sadf_ref="rover", count=3)])
    objective = _objective(SuccessCriterion(id="c", binding=_binding(metric="water")))
    agents = LocalSwarmComposer().compose(candidate)
    env, world_ref = LocalEnvironmentFactory().instantiate(
        candidate, objective, agents=agents, seed=5
    )
    policy = LocalPolicyConditioner().condition(candidate, objective, seed=5)
    episode = LocalSimulator().rollout(env, policy, objective, world_ref, seed=5, max_steps=6)
    assert isinstance(episode, EpisodeResult)
    assert episode.steps == 6 and len(episode.metric_samples["water"]) == 6
    assert episode.sim_time_s == 6.0


def test_local_clients_bundle() -> None:
    clients = local_clients()
    assert isinstance(clients.composer, LocalSwarmComposer)
    assert isinstance(clients.scorer, LocalScorer)


# ---- aggregation ---------------------------------------------------------- #


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        (MetricAggregation.MEAN, 3.0),
        (MetricAggregation.MEDIAN, 3.0),
        (MetricAggregation.MIN, 1.0),
        (MetricAggregation.MAX, 5.0),
        (MetricAggregation.SUM, 15.0),
    ],
)
def test_aggregate_basic(how: MetricAggregation, expected: float) -> None:
    assert _aggregate([1.0, 2.0, 3.0, 4.0, 5.0], how) == expected


def test_aggregate_percentiles() -> None:
    values = [float(i) for i in range(10)]
    assert _aggregate(values, MetricAggregation.P05) == 4.0  # clamped low index
    assert _aggregate(values, MetricAggregation.P95) == 5.0  # clamped high index


# ---- scoring -------------------------------------------------------------- #


def test_scorer_soft_pass_and_uncertainty() -> None:
    objective = _objective(
        SuccessCriterion(id="c", binding=_binding(target=10.0, tolerance=1.0), weight=2.0)
    )
    episode = EpisodeResult(
        world_ref="w", sim_time_s=1.0, steps=3, metric_samples={"m": [9.5, 10.0, 10.5]}
    )
    score = LocalScorer().score(episode, objective)
    assert score.passed is True
    assert score.metric_scores["m"] == 10.0
    assert score.metric_uncertainty["m"] > 0.0
    assert 0.0 < score.aggregate <= 1.0
    assert score.objective_hash == objective_content_hash(objective)


def test_scorer_hard_threshold_and_required_failure() -> None:
    objective = _objective(
        SuccessCriterion(
            id="c",
            binding=_binding(
                direction=MetricDirection.LOWER_BETTER, threshold=5.0, target=5.0, tolerance=0.0
            ),
        )
    )
    fail = LocalScorer().score(
        EpisodeResult(world_ref="w", sim_time_s=1.0, steps=1, metric_samples={"m": [7.0]}),
        objective,
    )
    assert fail.passed is False
    ok = LocalScorer().score(
        EpisodeResult(world_ref="w", sim_time_s=1.0, steps=1, metric_samples={"m": [3.0]}),
        objective,
    )
    assert ok.passed is True


def test_scorer_higher_better_threshold() -> None:
    objective = _objective(SuccessCriterion(id="c", binding=_binding(threshold=8.0)))
    ok = LocalScorer().score(
        EpisodeResult(world_ref="w", sim_time_s=1.0, steps=1, metric_samples={"m": [9.0]}),
        objective,
    )
    assert ok.passed is True


def test_scorer_non_required_failure_keeps_passed() -> None:
    objective = _objective(
        SuccessCriterion(id="c", binding=_binding(target=10.0, tolerance=0.0), required=False)
    )
    score = LocalScorer().score(
        EpisodeResult(world_ref="w", sim_time_s=1.0, steps=1, metric_samples={"m": [3.0]}),
        objective,
    )
    assert score.passed is True  # a soft/stretch miss does not fail the objective


def test_scorer_missing_samples_fail_and_zero_aggregate() -> None:
    objective = _objective(SuccessCriterion(id="c", binding=_binding()))
    score = LocalScorer().score(
        EpisodeResult(world_ref="w", sim_time_s=0.0, steps=0, metric_samples={}), objective
    )
    assert score.passed is False and score.aggregate == 0.0 and score.metric_scores == {}


def test_scorer_zero_target_and_zero_score_satisfaction() -> None:
    # higher_better with target 0 -> satisfaction defaults to 1.0
    hb = _objective(SuccessCriterion(id="c", binding=_binding(target=0.0, tolerance=100.0)))
    s1 = LocalScorer().score(
        EpisodeResult(world_ref="w", sim_time_s=0.0, steps=1, metric_samples={"m": [5.0]}), hb
    )
    assert s1.aggregate == 1.0
    # lower_better with score 0 -> satisfaction defaults to 1.0
    lb = _objective(
        SuccessCriterion(
            id="c",
            binding=_binding(direction=MetricDirection.LOWER_BETTER, target=1.0, tolerance=100.0),
        )
    )
    s2 = LocalScorer().score(
        EpisodeResult(world_ref="w", sim_time_s=0.0, steps=1, metric_samples={"m": [0.0]}), lb
    )
    assert s2.aggregate == 1.0
