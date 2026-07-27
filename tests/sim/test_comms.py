"""RM-P0-SIM-08 — comms masking: a Core ContactPlan → per-tick CommsObservationMasks.

Proves Sim consumes Link connectivity through the Core contract (a ``ContactPlan`` in, a
``CommsObservationMask`` per tick out) and applies it through the Environment API, so PSR
comms-denial and relay-window gaps are real in simulation: an agent inside a gap sees its peers
*denied*, not absent, and Bench's comms-robustness scoring reads the mask off the observation.
Determinism is preserved (masks are a pure function of plan + epoch) and a scenario with no
ContactPlan is byte-identical to before.
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.messages.model import (
    ActionBatch,
    CommsObservationMask,
    ContactInterval,
    ContactNode,
    ContactPlan,
    Observation,
)
from astro_mine.core.units import J2000_EPOCH, Epoch
from astro_mine.sim.comms import (
    ConnectivitySource,
    ReferenceConnectivitySampler,
    apply_comms_mask,
)
from astro_mine.sim.runtime import AgentSpec, Scenario, Simulator, run_episode

# A small contact graph: one ground station (DSN), one relay, two surface rovers. The rover-a<->dsn
# window is open only over the half-open span [2, 4) TDB seconds; rover-a<->relay is open the whole
# episode; rover-b has no window at all (always denied). Nodes are declared out of sorted order so
# the sampler's order-independent pair keying is exercised both ways.
_DSN = ContactNode(id="dsn", role=NodeRole.GROUND)
_RELAY = ContactNode(id="relay", role=NodeRole.SPACE)
_ROVER_A = ContactNode(id="rover-a", role=NodeRole.SPACE)
_ROVER_B = ContactNode(id="rover-b", role=NodeRole.SPACE)


def _plan() -> ContactPlan:
    return ContactPlan(
        nodes=[_DSN, _RELAY, _ROVER_A, _ROVER_B],
        intervals=[
            ContactInterval(
                node_a="rover-a",
                node_b="dsn",
                start_tdb_s=2.0,
                end_tdb_s=4.0,
                max_rate_bps=1_000_000.0,
                mean_latency_s=1.3,
                margin_db=3.0,
            ),
            # node_a/node_b stored relay-first to exercise pair-key normalization; mean
            # latency is left unset so the fallback to the light-time floor is covered.
            ContactInterval(
                node_a="relay",
                node_b="rover-a",
                start_tdb_s=0.0,
                end_tdb_s=10.0,
                max_rate_bps=500_000.0,
                min_latency_s=0.4,
            ),
        ],
    )


def _epoch(tdb_s: float) -> Epoch:
    return Epoch(tdb_seconds=tdb_s, scale=J2000_EPOCH.scale)


def _peer(mask: CommsObservationMask, peer: str) -> object:
    (link,) = (link for link in mask.links if link.peer == peer)
    return link


# --- ReferenceConnectivitySampler: ContactPlan → mask ----------------------------------------


def test_sampler_reports_the_contact_graph_nodes() -> None:
    sampler = ReferenceConnectivitySampler(_plan())
    assert sampler.nodes == ("dsn", "relay", "rover-a", "rover-b")


def test_a_peer_is_reachable_only_inside_its_open_window() -> None:
    sampler = ReferenceConnectivitySampler(_plan())
    # Denied before the window, reachable at its (inclusive) start, denied at its (exclusive) end.
    assert _peer(sampler.comms_mask("rover-a", _epoch(0.0)), "dsn").reachable is False
    assert _peer(sampler.comms_mask("rover-a", _epoch(2.0)), "dsn").reachable is True
    assert _peer(sampler.comms_mask("rover-a", _epoch(3.999)), "dsn").reachable is True
    assert _peer(sampler.comms_mask("rover-a", _epoch(4.0)), "dsn").reachable is False


def test_a_reachable_peer_carries_its_link_quality() -> None:
    sampler = ReferenceConnectivitySampler(_plan())
    mask = sampler.comms_mask("rover-a", _epoch(2.0))
    dsn = _peer(mask, "dsn")
    assert (dsn.rate_bps, dsn.latency_s, dsn.margin_db) == (1_000_000.0, 1.3, 3.0)
    # The relay interval leaves mean latency unset, so the representative latency is its floor.
    relay = _peer(mask, "relay")
    assert (relay.reachable, relay.rate_bps, relay.latency_s) == (True, 500_000.0, 0.4)


def test_a_denied_peer_carries_no_link_quality() -> None:
    sampler = ReferenceConnectivitySampler(_plan())
    dsn = _peer(sampler.comms_mask("rover-a", _epoch(0.0)), "dsn")
    assert (dsn.reachable, dsn.rate_bps, dsn.latency_s, dsn.margin_db) == (False, None, None, None)


def test_an_agent_with_no_windows_sees_every_peer_denied() -> None:
    sampler = ReferenceConnectivitySampler(_plan())
    mask = sampler.comms_mask("rover-b", _epoch(2.0))
    assert all(link.reachable is False for link in mask.links)
    assert mask.earth_contact is False


def test_earth_contact_tracks_a_reachable_ground_node() -> None:
    sampler = ReferenceConnectivitySampler(_plan())
    # At t=0 only the relay (a SPACE node) is reachable → no Earth contact; at t=2 the DSN
    # (a GROUND node) opens → Earth contact, even though a space link was reachable throughout.
    assert sampler.comms_mask("rover-a", _epoch(0.0)).earth_contact is False
    assert sampler.comms_mask("rover-a", _epoch(2.0)).earth_contact is True


def test_a_mask_never_lists_the_agent_itself_as_a_peer() -> None:
    sampler = ReferenceConnectivitySampler(_plan())
    mask = sampler.comms_mask("rover-a", _epoch(2.0))
    assert "rover-a" not in {link.peer for link in mask.links}
    assert {link.peer for link in mask.links} == {"dsn", "relay", "rover-b"}


def test_masking_an_agent_outside_the_plan_is_an_error() -> None:
    sampler = ReferenceConnectivitySampler(_plan())
    with pytest.raises(KeyError, match="not a node of the contact plan"):
        sampler.comms_mask("ghost", _epoch(2.0))


def test_content_hash_is_stable_and_plan_sensitive() -> None:
    plan = _plan()
    assert (
        ReferenceConnectivitySampler(plan).content_hash
        == ReferenceConnectivitySampler(plan).content_hash
    )
    empty = ContactPlan(nodes=[_ROVER_A], intervals=[])
    assert (
        ReferenceConnectivitySampler(plan).content_hash
        != ReferenceConnectivitySampler(empty).content_hash
    )


def test_reference_sampler_satisfies_the_connectivity_source_protocol() -> None:
    assert isinstance(ReferenceConnectivitySampler(_plan()), ConnectivitySource)
    assert not isinstance(object(), ConnectivitySource)


# --- apply_comms_mask: the Environment-surface primitive -------------------------------------


def test_apply_comms_mask_sets_comms_without_mutating_the_input() -> None:
    sampler = ReferenceConnectivitySampler(_plan())
    sim = Simulator(_scenario())
    base = sim.reset().observations["rover-c"]  # an agent outside the plan → comms starts unset
    assert base.comms is None
    mask = sampler.comms_mask("rover-a", _epoch(2.0))
    masked = apply_comms_mask(base, mask)
    assert masked.comms == mask
    assert base.comms is None  # pure: the original is untouched
    assert isinstance(masked, Observation)


# --- Simulator integration: masks applied per tick through the Environment API ----------------


def _scenario() -> Scenario:
    # rover-c is deliberately NOT a contact-graph node: it must pass through with comms unset,
    # proving only agents that are plan nodes are masked. Batteries are ample so no agent retires
    # before the window closes at tick 4.
    return Scenario(
        name="comms-masking",
        agents=(
            AgentSpec(agent_id="rover-a", battery_soc_j=1000.0),
            AgentSpec(agent_id="rover-b", battery_soc_j=1000.0),
            AgentSpec(agent_id="rover-c", battery_soc_j=1000.0),
        ),
        seed=7,
        horizon_steps=6,
    )


def test_simulator_masks_only_agents_that_are_contact_nodes() -> None:
    sim = Simulator(_scenario(), connectivity=ReferenceConnectivitySampler(_plan()))
    observations = sim.reset().observations
    assert observations["rover-a"].comms is not None  # a plan node → masked
    assert observations["rover-b"].comms is not None  # a plan node → masked
    assert observations["rover-c"].comms is None  # not a plan node → untouched


def test_simulator_applies_the_window_per_tick() -> None:
    sim = Simulator(_scenario(), connectivity=ReferenceConnectivitySampler(_plan()))
    reachable: list[bool] = []
    earth: list[bool] = []
    result = sim.reset()  # tick 0
    for _ in range(5):  # ticks 1..5
        comms = result.observations["rover-a"].comms
        assert comms is not None
        reachable.append(bool(_peer(comms, "dsn").reachable))
        earth.append(comms.earth_contact)
        result = sim.step(ActionBatch())
    # The rover-a<->dsn window is open over [2, 4) TDB seconds, i.e. exactly ticks 2 and 3.
    assert reachable == [False, False, True, True, False]
    assert earth == [False, False, True, True, False]


def test_without_connectivity_observations_are_unmasked() -> None:
    sim = Simulator(_scenario())  # no connectivity source
    observations = sim.reset().observations
    assert all(obs.comms is None for obs in observations.values())


# --- run_episode: determinism, provenance, and backward compatibility -------------------------


def test_same_plan_and_seed_reproduce_byte_for_byte() -> None:
    scenario, sampler = _scenario(), ReferenceConnectivitySampler(_plan())
    first = run_episode(scenario, connectivity=sampler)
    second = run_episode(scenario, connectivity=sampler)
    assert first.content_hash == second.content_hash


def test_masking_changes_the_trace() -> None:
    scenario = _scenario()
    masked = run_episode(scenario, connectivity=ReferenceConnectivitySampler(_plan()))
    plain = run_episode(scenario)
    assert masked.content_hash != plain.content_hash


def test_provenance_records_the_contact_plan_hash() -> None:
    scenario, sampler = _scenario(), ReferenceConnectivitySampler(_plan())
    masked = run_episode(scenario, connectivity=sampler)
    plain = run_episode(scenario)
    assert masked.provenance["source_content_hashes"]["contact_plan"] == sampler.content_hash
    assert "contact_plan" not in plain.provenance["source_content_hashes"]
