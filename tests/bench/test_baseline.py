"""The baseline policy + local scoring path (RM-P0-BENCH-05).

Covers the reference :class:`BaselinePolicy` (Core Policy conformance + determinism), the
dependency-clean :func:`reference_episode_runner` fixture (determinism, seed- and
policy-sensitivity, full metric coverage), the offline :func:`run` scoring path, and the
scoring-path determinism gate :func:`assert_score_reproducible`.
"""

from __future__ import annotations

import pytest

from astro_mine.bench.baseline import (
    BaselinePolicy,
    BenchRunnerProvider,
    DefaultPolicyProvider,
    EpisodeRunner,
    assert_score_reproducible,
    default_policy_for,
    fixture_runner_provider,
    reference_episode_runner,
    run,
)
from astro_mine.bench.harness import DeterminismError, Runner, reference_runner
from astro_mine.bench.metrics import Scorecard
from astro_mine.bench.scenario import ResolvedScenario, ScenarioSpec, resolve_scenario
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID, load_scenario
from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.policy import DecisionContext, Policy, check_policy
from astro_mine.core.scoring import EpisodeTrace

from ._factories import make_observation, make_scenario_spec

ANCHOR_METRICS = frozenset(
    {
        "water_mass",
        "energy_per_kg",
        "information_gain",
        "psr_area_characterized",
        "nights_survived",
        "comms_robustness",
        "discovery_latency",
    }
)


@pytest.fixture(scope="module")
def anchor() -> ScenarioSpec:
    return load_scenario(ANCHOR_SCENARIO_ID)


class _StandbyPolicy:
    """A do-nothing policy: emits an empty batch (a distinct fingerprint from the baseline)."""

    def decide(self, observations: object, context: DecisionContext) -> object:
        from astro_mine.core.messages import ActionBatch

        return ActionBatch()


# --- baseline policy ----------------------------------------------------------------------------


def test_baseline_satisfies_core_policy() -> None:
    policy = BaselinePolicy()
    assert isinstance(policy, Policy)  # runtime-checkable structural conformance
    observations = {
        "rover-b": make_observation(0, 0.0, agent_id="rover-b"),
        "rover-a": make_observation(0, 0.0, agent_id="rover-a"),
    }
    batch = check_policy(policy, observations, DecisionContext())  # validates the ActionBatch
    assert [a.agent_id for a in batch.actions] == ["rover-a", "rover-b"]  # sorted, order-stable
    assert all(a.kind is ActionKind.MODE for a in batch.actions)
    assert all(a.mode is not None and a.mode.mode == "prospect" for a in batch.actions)


def test_baseline_is_deterministic() -> None:
    policy = BaselinePolicy()
    obs = {"r": make_observation(0, 0.0, agent_id="r")}
    ctx = DecisionContext(seed=7)
    assert policy.decide(obs, ctx) == policy.decide(obs, ctx)


def test_baseline_empty_observations_empty_batch() -> None:
    assert BaselinePolicy().decide({}, DecisionContext()).actions == []


# --- reference episode runner -------------------------------------------------------------------


def test_reference_runner_is_deterministic_and_seed_sensitive() -> None:
    resolved = resolve_scenario(make_scenario_spec())
    policy = BaselinePolicy()
    first = reference_episode_runner(resolved, policy, 1001)
    assert isinstance(first, EpisodeTrace)
    assert reference_episode_runner(resolved, policy, 1001) == first  # same inputs -> same trace
    assert reference_episode_runner(resolved, policy, 1002) != first  # seed changes the trace


def test_reference_runner_is_policy_sensitive() -> None:
    resolved = resolve_scenario(make_scenario_spec())
    prospecting = reference_episode_runner(resolved, BaselinePolicy(mode="prospect"), 1001)
    idle = reference_episode_runner(resolved, BaselinePolicy(mode="idle"), 1001)
    standby = reference_episode_runner(resolved, _StandbyPolicy(), 1001)
    assert prospecting != idle  # a different action stream -> a different trace
    assert prospecting != standby


def test_reference_runner_populates_every_metric_channel() -> None:
    resolved = resolve_scenario(make_scenario_spec())
    trace = reference_episode_runner(resolved, BaselinePolicy(), 1003)
    species = {r.resource_species for obs in trace.observations for r in obs.sensors}
    assert {"water", "hydrogen"} <= species  # ISRU water + neutron hydrogen are distinct channels
    assert any(o.comms is not None for o in trace.observations)  # comms mask present
    assert any(o.self_state.battery_soc_j is not None for o in trace.observations)  # power/thermal
    assert trace.context.belief_history and trace.context.psr_cells  # belief-quality inputs
    assert trace.context.night_intervals  # survival inputs


# --- run(): the local scoring path --------------------------------------------------------------


def test_run_scores_the_anchor_baseline(anchor: ScenarioSpec) -> None:
    card = run(anchor, BaselinePolicy())
    assert isinstance(card, Scorecard)
    assert card.scenario_id == ANCHOR_SCENARIO_ID
    assert {m.metric for m in card.metrics} == ANCHOR_METRICS
    assert all(m.seeds == anchor.seeds.public for m in card.metrics)


def test_run_scores_only_the_requested_seeds(anchor: ScenarioSpec) -> None:
    card = run(anchor, BaselinePolicy(), seeds=(1001,))
    assert all(m.seeds == (1001,) for m in card.metrics)


def test_run_is_policy_sensitive(anchor: ScenarioSpec) -> None:
    prospecting = run(anchor, BaselinePolicy(mode="prospect"))
    idle = run(anchor, BaselinePolicy(mode="idle"))
    assert prospecting.content_hash != idle.content_hash


def test_run_reproduces_byte_for_byte(anchor: ScenarioSpec) -> None:
    assert run(anchor, BaselinePolicy()).content_hash == run(anchor, BaselinePolicy()).content_hash


# --- assert_score_reproducible: the scoring-path determinism gate --------------------------------


def test_score_gate_returns_the_canonical_scorecard(anchor: ScenarioSpec) -> None:
    card = assert_score_reproducible(anchor, BaselinePolicy())
    assert isinstance(card, Scorecard)
    assert card.content_hash == run(anchor, BaselinePolicy()).content_hash


def test_score_gate_needs_two_runs(anchor: ScenarioSpec) -> None:
    with pytest.raises(ValueError, match="runs >= 2"):
        assert_score_reproducible(anchor, BaselinePolicy(), runs=1)


class _FlakyRunner:
    """A non-deterministic episode runner: a fresh water reading each call (trips the gate)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, resolved: ResolvedScenario, policy: Policy, seed: int) -> EpisodeTrace:
        self.calls += 1
        return EpisodeTrace(observations=(make_observation(0, 0.0, water_kg=float(self.calls)),))


def test_score_gate_trips_on_nondeterminism(anchor: ScenarioSpec) -> None:
    with pytest.raises(DeterminismError, match="did not reproduce"):
        assert_score_reproducible(
            anchor, BaselinePolicy(), runner=_FlakyRunner(), seeds=(1001, 1002)
        )


# --- the optional default-policy seam (astro-mine-sim#61) ----------------------------------------


class _PolicylessProvider:
    """A provider that offers no policy — the shape every already-published runner has."""

    runner_id = "policyless/0.1.0"

    def episode_runner(self, store: object | None = None) -> EpisodeRunner:
        return reference_episode_runner

    def harness_runner(self, store: object | None = None, *, scorer: object = None) -> Runner:
        return reference_runner


class _OpinionatedProvider(_PolicylessProvider):
    """A provider that supplies its own baseline, built from whatever its store resolved."""

    runner_id = "opinionated/0.1.0"

    def __init__(self) -> None:
        self.seen: object | None = None

    def default_policy(self, spec: ScenarioSpec, store: object | None = None) -> BaselinePolicy:
        self.seen = (spec.scenario_id, store)
        return BaselinePolicy(mode="extract")


def test_a_provider_without_a_default_policy_still_satisfies_the_runner_protocol() -> None:
    """The seam must not break existing providers: it is a *separate* opt-in Protocol.

    Folding ``default_policy`` into ``BenchRunnerProvider`` would make every already-published
    provider — Sim's included — fail ``load_runner_provider``'s isinstance gate, because a
    runtime-checkable Protocol matches on method presence.
    """
    provider = _PolicylessProvider()
    assert isinstance(provider, BenchRunnerProvider)
    assert not isinstance(provider, DefaultPolicyProvider)


def test_default_policy_falls_back_to_the_baseline(anchor: ScenarioSpec) -> None:
    assert default_policy_for(_PolicylessProvider(), anchor) == BaselinePolicy()


def test_the_fixture_provider_scores_the_baseline_policy(anchor: ScenarioSpec) -> None:
    """The fixture path is unchanged — its baseline still reaches `score` through the fallback."""
    assert default_policy_for(fixture_runner_provider, anchor) == BaselinePolicy()


def test_a_providers_own_policy_is_used_and_is_handed_the_spec_and_store(
    anchor: ScenarioSpec,
) -> None:
    provider = _OpinionatedProvider()

    policy = default_policy_for(provider, anchor, "/tmp/registry")

    assert isinstance(policy, BaselinePolicy)
    assert policy.mode == "extract"
    # Both halves reach the builder: *where* the bundles are, and *which* ones this run pins.
    assert provider.seen == (anchor.scenario_id, "/tmp/registry")
    # Whatever a provider hands back is still a plain Core Policy — that is what keeps Bench from
    # ever naming an engine type (conventions.md §1.1).
    assert isinstance(policy, Policy)
    batch = check_policy(policy, {"rover": make_observation(0, 0.0)}, DecisionContext())
    assert all(a.mode is not None and a.mode.mode == "extract" for a in batch.actions)
