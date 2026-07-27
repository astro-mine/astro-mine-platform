"""Sim instantiation smoke test (RM-P0-FLEET-07).

A representative reference asset must **spawn and step** in Sim, and SADF that *validates*
but **cannot be realized** by any engine must be caught here -- this is the Fleet->Sim
seam (``fleet.md`` §10). The two halves of the acceptance criterion:

* **Positive** -- the relay orbiter and the prospecting rover each load from the reference
  library (RM-P0-FLEET-04), map onto a Sim ``Scenario`` + engine, and advance under
  ``Simulator.reset()`` / ``step()`` (RM-P0-SIM-01/02/03): the orbiter propagates, the
  rover traverses.
* **Negative** -- a schema-valid asset that no engine realizes (the ISRU plant, which
  declares no ``mobility`` regime; and a doc whose only edit is a valid-but-unserved
  ``mobility.regimes``) fails the smoke test, while Core's loader still accepts it.

The SADF->Sim bridge below is a **test-local** mapping, not a public Fleet API: the
production data plumbing from a SADF asset onto a Sim scenario is deliberately deferred
(``astro-mine-sim`` ``runtime/scenario.py``; the issue's "full asset coverage" is P1), so
this is the minimal realization a CI smoke test needs -- engine-served-regime gate plus a
one-agent scenario per asset. Sim's engine factories *silently skip* agents whose dynamics
they don't own, so unrealizability is an explicit pre-check here, never a caught engine error.

Backlog: RM-P0-FLEET-07 -- https://github.com/astro-mine/astro-mine-fleet/issues/7
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pytest

from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.sadf import SadfDocument, load_sadf
from astro_mine.core.sadf.enums import Regime
from astro_mine.core.sadf.model import Asset, Mobility
from astro_mine.core.units import INERTIAL_J2000
from astro_mine.fleet.library import load_reference
from astro_mine.sim.engines import (
    BUILTIN_ENGINES,
    EngineFactory,
    mobility_engine_factory,
    orbital_engine_factory,
)
from astro_mine.sim.runtime import (
    AgentSpec,
    MobilityDynamics,
    OrbitalDynamics,
    Scenario,
    Simulator,
)

# --- the realizability gate ------------------------------------------------------

#: The regimes the Phase-0 engine set actually realizes, read from Sim's own built-in
#: engine descriptors (RM-P0-SIM-03) rather than hard-coded, so the gate tracks Sim as
#: engines are added. Today: ``{surface, proximity_orbit}``.
SERVED_REGIMES: frozenset[Regime] = frozenset(
    regime for descriptor, _ in BUILTIN_ENGINES for regime in descriptor.regimes
)


class UnrealizableAsset(Exception):
    """A schema-valid SADF asset that no Sim engine can instantiate.

    Raised by :func:`realize` when an asset declares no mobility regime, or only regimes
    outside :data:`SERVED_REGIMES` -- the "validates but cannot be realized" case the smoke
    test must catch (``fleet.md`` §10)."""


@dataclass(frozen=True)
class Realization:
    """A reference asset mapped onto a runnable one-agent Sim episode."""

    scenario: Scenario
    engine_factory: EngineFactory


# --- SADF -> Sim parameter sourcing (reduced-order, smoke-test only) --------------
# Only the fields a reduced-order engine needs are sourced from SADF; the rest are
# documented Phase-0 stand-ins (the authoritative Worlds/Prospect plumbing is deferred).

#: Circular lunar orbit radius for the relay smoke run: ~100 km above the lunar mean
#: radius (1737.4 km). Only the spawn+step matters here, not orbit design.
ORBIT_RADIUS_M = 1.8374e6
#: A modest rover top speed (m/s). SADF carries no intrinsic top-speed field, so the smoke
#: test picks a reasonable lunar-rover default.
DEFAULT_MAX_SPEED_MPS = 0.5
#: The seeded drive speed (m/s) -- half the cap, so the rover clearly traverses while the
#: mobility engine coasts it at its commanded velocity (no action plumbing in a smoke test).
SMOKE_DRIVE_SPEED_MPS = 0.25
#: Fallbacks for assets that omit the sourced field (the reference assets don't, but the
#: bridge stays total). All must satisfy the engines' ``> 0`` constraints.
DEFAULT_MASS_KG = 100.0
DEFAULT_TRACTION_N = 100.0
DEFAULT_BATTERY_SOC_J = 1.0e6


def _battery_soc_j(asset: Asset) -> float:
    """Seed state-of-charge from the first storage cell's capacity, else a default."""
    if asset.power and asset.power.storage:
        return asset.power.storage[0].capacity_j
    return DEFAULT_BATTERY_SOC_J


def _idle_power_w(asset: Asset) -> float:
    """The asset's declared ``idle``-mode power draw, if any (else no idle draw)."""
    if asset.power:
        for load in asset.power.loads_by_mode:
            if load.mode == "idle":
                return load.power_w
    return 0.0


def _traction_n(mobility: Mobility) -> float:
    """A drawbar-pull proxy from the first ground-contact element that declares both a
    pressure and a footprint (``ground_pressure x footprint``), else a default."""
    for element in mobility.contact:
        if element.max_ground_pressure_pa is not None and element.footprint_m2 is not None:
            return element.max_ground_pressure_pa * element.footprint_m2
    return DEFAULT_TRACTION_N


def _orbital_spec(asset: Asset) -> AgentSpec:
    """Spawn an orbiter on a circular lunar orbit (inertial frame, default mu)."""
    dynamics = OrbitalDynamics()
    speed_mps = math.sqrt(dynamics.mu_m3_s2 / ORBIT_RADIUS_M)
    return AgentSpec(
        agent_id=asset.identity.id,
        initial_position_m=(ORBIT_RADIUS_M, 0.0, 0.0),
        velocity_mps=(0.0, speed_mps, 0.0),
        battery_soc_j=_battery_soc_j(asset),
        frame=INERTIAL_J2000,
        dynamics=dynamics,
    )


def _surface_spec(asset: Asset, mobility: Mobility) -> AgentSpec:
    """Spawn a surface rover seeded with a small drive velocity (body-fixed frame)."""
    mass_kg = asset.bodies[0].mass_kg if asset.bodies else DEFAULT_MASS_KG
    dynamics = MobilityDynamics(
        mass_kg=mass_kg,
        max_speed_mps=DEFAULT_MAX_SPEED_MPS,
        max_traction_n=_traction_n(mobility),
        idle_power_w=_idle_power_w(asset),
    )
    return AgentSpec(
        agent_id=asset.identity.id,
        velocity_mps=(SMOKE_DRIVE_SPEED_MPS, 0.0, 0.0),
        battery_soc_j=_battery_soc_j(asset),
        dynamics=dynamics,
    )


def realize(doc: SadfDocument) -> Realization:
    """Map a validated SADF asset onto a one-agent Sim episode + its engine.

    Raises :class:`UnrealizableAsset` if the asset declares no mobility regime an engine
    serves -- the realizability gate. Otherwise it spawns the asset on the engine for its
    regime (orbital for ``proximity_orbit``, mobility for ``surface``)."""
    asset = doc.asset
    mobility = asset.mobility
    declared = set(mobility.regimes) if mobility else set()
    served = declared & SERVED_REGIMES
    if not served:
        raise UnrealizableAsset(
            f"asset {asset.identity.id!r} declares regimes "
            f"{sorted(r.value for r in declared)} that no Sim engine realizes "
            f"(served: {sorted(r.value for r in SERVED_REGIMES)})"
        )

    factory: EngineFactory
    if Regime.PROXIMITY_ORBIT in served:
        spec, factory = _orbital_spec(asset), orbital_engine_factory
    else:  # SERVED_REGIMES is exactly {proximity_orbit, surface}, so this is surface
        assert mobility is not None  # narrowed: a served surface regime implies a block
        spec, factory = _surface_spec(asset, mobility), mobility_engine_factory

    scenario = Scenario(
        name=f"fleet-07-smoke-{asset.identity.id}",
        agents=(spec,),
        dt_s=1.0,
        horizon_steps=16,
    )
    return Realization(scenario=scenario, engine_factory=factory)


def _position(observation: Observation) -> tuple[float, float, float]:
    translation = observation.self_state.pose.translation_m
    return (translation.x, translation.y, translation.z)


# --- tests -----------------------------------------------------------------------

#: Steps to advance in the positive smoke run -- enough to move and accrue sim-time.
_SMOKE_STEPS = 3


def test_served_regimes_cover_the_mapped_engines() -> None:
    # Sanity: the gate is built from Sim's engines and covers the two regimes we map.
    assert SERVED_REGIMES
    assert {Regime.SURFACE, Regime.PROXIMITY_ORBIT} <= SERVED_REGIMES


@pytest.mark.parametrize("name", ["relay_orbiter", "prospecting_rover"])
def test_reference_asset_spawns_and_steps_in_sim(name: str) -> None:
    doc = load_reference(name)
    realization = realize(doc)
    agent_id = doc.asset.identity.id

    sim = Simulator(realization.scenario, engine_factory=realization.engine_factory)
    reset = sim.reset()
    assert agent_id in reset.observations
    start = _position(reset.observations[agent_id])

    result = None
    for _ in range(_SMOKE_STEPS):
        result = sim.step(ActionBatch())
    assert result is not None

    # It stayed live and actually advanced: in the active set, sim-time accrued, and the
    # pose moved (the orbiter propagated / the rover traversed) -- not merely instantiated.
    assert agent_id in result.observations
    assert agent_id in sim.agents
    assert result.sim_time_s == pytest.approx(_SMOKE_STEPS * realization.scenario.dt_s)
    assert _position(result.observations[agent_id]) != start


def test_isru_plant_validates_but_cannot_be_realized() -> None:
    # The ISRU plant is shipped, schema-valid SADF (Core's loader accepts it), but it
    # declares no mobility regime, so no engine can spawn it -- the smoke test must reject it.
    doc = load_reference("isru_plant")
    assert doc.asset.mobility is None
    with pytest.raises(UnrealizableAsset):
        realize(doc)


def test_valid_sadf_with_unserved_regime_is_unrealizable() -> None:
    # The purest "validates but cannot be realized" case: take a real asset and change only
    # its mobility regime to a valid-but-unserved one. Core's loader still accepts the doc
    # (it's well-formed SADF, a real Regime member), yet the realizability gate rejects it.
    data = load_reference("relay_orbiter").model_dump(mode="json", by_alias=True, exclude_none=True)
    data["asset"]["mobility"]["regimes"] = [Regime.INTERPLANETARY_TRANSIT.value]

    doc = load_sadf(json.dumps(data))
    assert doc.asset.mobility is not None
    assert Regime.INTERPLANETARY_TRANSIT in doc.asset.mobility.regimes
    with pytest.raises(UnrealizableAsset):
        realize(doc)
