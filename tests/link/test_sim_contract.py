"""RM-P0-LINK-04 — the consumer-driven contract test: Link's sampler *is* Sim's ConnectivitySource.

Sim documents the seam (``astro_mine.sim.comms``): *"Structurally identical to Link's
``ConnectivitySampler``: a caller that has Link installed can inject Link's optimized sampler
directly, since Sim depends only on this shape, never on the Link package."* Until now that was an
assertion in a docstring — Sim never imports Link, Link never imports Sim, and Sim's own
``ReferenceConnectivitySampler`` was the only thing ever checked against the Protocol. So nothing
proved a **real Link-produced ContactPlan** could drive the comms-denied benchmark.

This test closes that: it imports Sim's *actual* ``ConnectivitySource`` Protocol and its reference
implementation and asserts, against Link's real
:class:`~astro_mine.link.products.ConnectivitySampler`,

1. **structural conformance** — Link's sampler satisfies Sim's ``@runtime_checkable`` Protocol, and
   carries the ``content_hash`` Sim's run provenance duck-types off an injected source;
2. **behavioural equivalence** — over the same Core ``ContactPlan`` and a swept epoch grid, Link's
   sampler and Sim's reference produce *identical* ``CommsObservationMask``\\ es, for every agent;
3. **Environment-API wiring** — a mask Link emits is applied to a Core ``Observation`` by Sim's own
   ``apply_comms_mask``, which is how comms denial reaches a policy (``LUNAR-TR-003``).

``astro-mine-sim`` is a **test-only** dependency (the ``sim-contract`` dependency group). Nothing
under ``src/`` imports it, Sim does not depend on Link, and the runtime coupling stays Core-typed
(``ContactPlan`` in, ``CommsObservationMask`` out) — no edge→edge side-channel (conventions.md
§1.1). The test skips where the group is not installed, as the GMAT oracle regression does.

Backlog: RM-P0-LINK-04 -- astro-mine-link#25
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages import ContactInterval, ContactNode, ContactPlan, Observation
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.core.units import MOON_BODY_FIXED, Epoch, TimeScale
from astro_mine.link.products import ConnectivitySampler

pytest.importorskip(
    "astro_mine.sim.comms",
    reason="astro-mine-sim not installed (uv sync --group sim-contract)",
)

# Imported after the importorskip guard above, on purpose: the module is optional.
from astro_mine.sim.comms import (
    ConnectivitySource,
    ReferenceConnectivitySampler,
    apply_comms_mask,
)

_STEP_S = 30.0


def _plan() -> ContactPlan:
    """A comms-denied plan: a PSR rover reaching Earth only through the relay's passes."""
    nodes = [
        ContactNode(id="prospecting-rover", role=NodeRole.SPACE, kind="surface_agent"),
        ContactNode(id="isru-plant", role=NodeRole.SPACE, kind="surface_agent"),
        ContactNode(id="relay-orbiter", role=NodeRole.SPACE, kind="relay_orbiter"),
        ContactNode(id="DSS-14-Goldstone", role=NodeRole.GROUND, kind="ground_station"),
    ]
    intervals = [
        # The rover sees the relay in two disjoint passes; between them it is denied.
        ContactInterval(
            node_a="prospecting-rover",
            node_b="relay-orbiter",
            start_tdb_s=0.0,
            end_tdb_s=900.0,
            max_rate_bps=2.0e6,
            min_latency_s=0.008,
            mean_latency_s=0.01,
            margin_db=6.5,
            modcod="qpsk_r1_2",
        ),
        ContactInterval(
            node_a="prospecting-rover",
            node_b="relay-orbiter",
            start_tdb_s=3600.0,
            end_tdb_s=4500.0,
            max_rate_bps=1.5e6,
        ),
        # The lit ridge plant has a direct Earth link; the PSR rover never does.
        ContactInterval(
            node_a="isru-plant",
            node_b="DSS-14-Goldstone",
            start_tdb_s=600.0,
            end_tdb_s=5400.0,
            max_rate_bps=8.0e6,
            mean_latency_s=1.28,
        ),
        ContactInterval(
            node_a="relay-orbiter",
            node_b="DSS-14-Goldstone",
            start_tdb_s=1200.0,
            end_tdb_s=4800.0,
            max_rate_bps=1.0e7,
        ),
        ContactInterval(
            node_a="prospecting-rover", node_b="isru-plant", start_tdb_s=0.0, end_tdb_s=1200.0
        ),
    ]
    return ContactPlan(
        nodes=nodes, intervals=intervals, epoch_start_tdb_s=0.0, epoch_end_tdb_s=5400.0
    )


def _observation(agent_id: str, epoch: Epoch) -> Observation:
    """A minimal Core Observation — the Environment-API surface a comms mask is applied onto."""
    pose = Transform(
        translation_m=Vec3(x=0.0, y=0.0, z=0.0),
        rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    state = StateSample(agent_id=agent_id, frame=MOON_BODY_FIXED, pose=pose)
    return Observation(
        tick=0,
        sim_time_s=epoch.tdb_seconds,
        agent_id=agent_id,
        self_state=state,
        epoch=epoch,
    )


def _grid() -> list[Epoch]:
    plan = _plan()
    start = plan.epoch_start_tdb_s or 0.0
    end = plan.epoch_end_tdb_s or 0.0
    n = int((end - start) / _STEP_S) + 1
    return [Epoch(tdb_seconds=start + i * _STEP_S, scale=TimeScale.TDB) for i in range(n)]


# --- 1. structural conformance ---------------------------------------------------------------


def test_link_sampler_satisfies_sims_connectivity_source() -> None:
    sampler = ConnectivitySampler(_plan())
    # Sim's @runtime_checkable Protocol accepts Link's sampler as-is — no adapter, no shim.
    assert isinstance(sampler, ConnectivitySource)
    # runtime_checkable only checks member *presence*, so exercise the signatures too: this is
    # exactly the call pattern Simulator._step drives per tick against the injected source.
    source: ConnectivitySource = sampler
    epoch = Epoch(tdb_seconds=100.0, scale=TimeScale.TDB)
    assert source.nodes == sampler.nodes
    assert source.comms_mask(source.nodes[0], epoch).agent_id == source.nodes[0]


def test_link_sampler_carries_the_contact_plan_content_hash() -> None:
    # Sim's run provenance duck-types `content_hash` off the injected source to record
    # source_content_hashes["contact_plan"] (RM-P0-SIM-09). Link's sampler must carry it, or a
    # Link-driven run silently loses the comms input from its provenance.
    plan = _plan()
    sampler = ConnectivitySampler(plan)
    assert isinstance(sampler.content_hash, str)
    assert sampler.content_hash == ConnectivitySampler(plan).content_hash


# --- 2. behavioural equivalence over a real plan ----------------------------------------------


def test_link_and_sim_samplers_agree_on_every_mask() -> None:
    plan = _plan()
    link = ConnectivitySampler(plan)
    reference = ReferenceConnectivitySampler(plan)

    assert set(link.nodes) == set(reference.nodes)
    for epoch in _grid():
        for agent in reference.nodes:
            assert link.comms_mask(agent, epoch) == reference.comms_mask(agent, epoch)


def test_the_plan_is_actually_comms_denied() -> None:
    # Guard the guard: a plan that is never denied would make the equivalence check vacuous.
    link = ConnectivitySampler(_plan())
    masks = [link.comms_mask("prospecting-rover", epoch) for epoch in _grid()]
    assert any(mask.earth_contact for mask in masks) is False  # PSR: never a direct Earth link
    reachability = {
        link_state.reachable
        for mask in masks
        for link_state in mask.links
        if link_state.peer == "relay-orbiter"
    }
    assert reachability == {True, False}  # the relay rises and sets


# --- 3. the Environment-API wiring (LUNAR-TR-003) ---------------------------------------------


def test_link_mask_flows_through_sims_environment_api() -> None:
    link = ConnectivitySampler(_plan())
    epoch = Epoch(tdb_seconds=2000.0, scale=TimeScale.TDB)
    mask = link.comms_mask("prospecting-rover", epoch)

    masked = apply_comms_mask(_observation("prospecting-rover", epoch), mask)
    assert masked.comms == mask
    # At t=2000 s the rover is between relay passes: every peer link is denied.
    assert all(not peer.reachable for peer in masked.comms.links)
    assert masked.comms.earth_contact is False
