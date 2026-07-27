"""``StackPolicy`` — a composed stack behind the Core Policy contract.

The adapter that lets a runtime which owns the loop (Bench's ``EpisodeRunner``, Sim's
``run_episode``) drive a Mind stack. It is exercised for real against Sim in
``test_sim_e2e.py``; here it is pinned against the toy env, where the *Executive* is available as
an independent oracle for what the very same stack should decide.

The load-bearing property is the equivalence test: a stack driven tick-by-tick through
``StackPolicy`` must produce a byte-identical decision trace to the one the Executive produces
driving the same stack over the same environment. If those ever diverge, the two runtimes disagree
about what Mind decided — and the Bench score would be scoring something other than the stack the
golden traces gate.
"""

from __future__ import annotations

import pytest

from astro_mine.core.env.model import ResetResult
from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.policy import DecisionContext, check_policy
from astro_mine.mind.compose import compose
from astro_mine.mind.compose.graph import HierarchyGraph
from astro_mine.mind.exec import Executive, StackPolicy
from astro_mine.mind.trace import to_canonical_json
from tests.mind.support.harness import (
    REFERENCE_SEED,
    TAGGING_SENTINEL,
    TaggingShield,
    compose_stack,
    policy_plugin,
    reference_doc_with_shield,
    reference_registry,
)
from tests.mind.support.toy_env import ToyProspectingEnv

_HORIZON = 8


def _drive(policy: StackPolicy, env: ToyProspectingEnv, *, max_ticks: int, seed: int) -> None:
    """Step ``env`` under ``policy`` the way Sim/Bench do — the loop belongs to the caller."""
    reset: ResetResult = env.reset(seed=seed)
    observations = reset.observations
    for _ in range(max_ticks):
        batch = policy.decide(observations, DecisionContext(seed=seed))
        result = env.step(batch)
        observations = result.observations
        if not observations:
            break


def test_stack_policy_honors_the_core_policy_contract() -> None:
    graph = compose_stack("lunar_prospecting.yaml", seed=REFERENCE_SEED)
    env = ToyProspectingEnv(horizon=_HORIZON)
    observations = env.reset(seed=REFERENCE_SEED).observations

    # Core's own conformance check — Mind claims the Policy/Planner interface, so it must hold it.
    batch = check_policy(StackPolicy(graph), observations, DecisionContext(seed=REFERENCE_SEED))
    assert isinstance(batch, ActionBatch)


def test_the_policy_and_the_executive_decide_identically() -> None:
    """The adapter is faithful: same stack + same env => byte-identical decision trace."""
    seed = REFERENCE_SEED
    by_executive = Executive(compose_stack("lunar_prospecting.yaml", seed=seed)).run(
        ToyProspectingEnv(horizon=_HORIZON), max_ticks=_HORIZON, seed=seed
    )

    policy = StackPolicy(compose_stack("lunar_prospecting.yaml", seed=seed))
    _drive(policy, ToyProspectingEnv(horizon=_HORIZON), max_ticks=_HORIZON, seed=seed)

    assert to_canonical_json(policy.trace) == to_canonical_json(by_executive.trace)


def test_the_shield_is_still_the_only_egress() -> None:
    """Inverting the loop must not open a second path to the environment (RM-P1-MIND-05)."""
    registry = reference_registry()
    registry.register(
        policy_plugin("test.tagging_shield", lambda _: TaggingShield(), tier="shield")
    )
    graph: HierarchyGraph = compose(
        reference_doc_with_shield("test.tagging_shield"), registry, seed=REFERENCE_SEED
    )

    policy = StackPolicy(graph)
    _drive(policy, ToyProspectingEnv(horizon=_HORIZON), max_ticks=_HORIZON, seed=REFERENCE_SEED)

    emitted = [a for tick in policy.trace.ticks for a in tick.action_batch.actions]
    assert emitted, "no actions were emitted"
    assert all(a.sim_time_s == TAGGING_SENTINEL for a in emitted), (
        "an action reached the environment without passing through the shield"
    )


def test_a_rewound_tick_begins_a_fresh_episode() -> None:
    """Bench reuses ONE policy object across every seed it scores, and the Core Policy contract has
    no "new episode" signal. If the strategy's cached tier decisions survived into the next
    episode, a score would depend on the order seeds were evaluated in — so a rewound tick must
    reset the stack."""
    policy = StackPolicy(compose_stack("lunar_prospecting.yaml", seed=REFERENCE_SEED))

    _drive(policy, ToyProspectingEnv(horizon=_HORIZON), max_ticks=_HORIZON, seed=REFERENCE_SEED)
    first = to_canonical_json(policy.trace)
    assert len(policy.trace.ticks) == _HORIZON

    # Re-drive the SAME policy object over a fresh env, exactly as Bench does for the next seed.
    _drive(policy, ToyProspectingEnv(horizon=_HORIZON), max_ticks=_HORIZON, seed=REFERENCE_SEED)
    second = to_canonical_json(policy.trace)

    assert len(policy.trace.ticks) == _HORIZON, "the previous episode's ticks leaked into this one"
    assert second == first, "a reused policy object did not reproduce the episode"


def test_explicit_reset_clears_the_trace() -> None:
    policy = StackPolicy(compose_stack("lunar_prospecting.yaml", seed=REFERENCE_SEED))
    _drive(policy, ToyProspectingEnv(horizon=_HORIZON), max_ticks=_HORIZON, seed=REFERENCE_SEED)
    assert policy.trace.ticks

    policy.reset()

    assert policy.trace.ticks == ()
    assert policy.graph.stack_id == "reference-lunar-prospecting"


def test_no_observations_decides_nothing() -> None:
    """Every agent terminated: there is no tick to record and nothing to command."""
    policy = StackPolicy(compose_stack("lunar_prospecting.yaml", seed=REFERENCE_SEED))

    batch = policy.decide({}, DecisionContext(seed=REFERENCE_SEED))

    assert batch == ActionBatch()
    assert policy.trace.ticks == ()


@pytest.mark.parametrize("blackout", [(), (3, 4, 5, 6, 7)])
def test_the_degrade_path_fires_through_the_adapter(blackout: tuple[int, ...]) -> None:
    """The comms signal reaches the degrade strategy through ``Observation.comms``, not through the
    Executive — so a loop owned by Sim degrades exactly as one owned by Mind does.

    The blackout has to outlast the mission tier's 3 s validity horizon to observe act-while-stale:
    on the tick comms drops, the stack's ``comms_lost`` trigger *replans* (that is the point of the
    trigger), which re-arms the horizon. Only once that fresh plan expires — still dark — does the
    mission replan get suppressed and the swarm act on cached intent."""
    policy = StackPolicy(compose_stack("lunar_prospecting_degrade.yaml", seed=REFERENCE_SEED))
    env = ToyProspectingEnv(horizon=10, comms_denied_ticks=blackout)

    _drive(policy, env, max_ticks=10, seed=REFERENCE_SEED)

    denied = [t.tick for t in policy.trace.ticks if t.comms_denied]
    assert denied == list(blackout)
    notes = {rec.note for tick in policy.trace.ticks for rec in tick.tiers}
    if blackout:
        assert "comms_stale_hold" in notes, "the mission tier did not act on cached intent"
        assert "comms_recovered" in notes, "the mission tier did not reconcile on recovery"
    else:
        assert "comms_stale_hold" not in notes
