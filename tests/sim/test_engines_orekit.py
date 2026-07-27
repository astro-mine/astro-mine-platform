"""RM-P0-SIM-03 — the Orekit higher-fidelity orbital tier.

The real orbital backend RM-P0-SIM-03 names, behind the same ``RegimeEngine`` waist as the
reduced-order RK4 two-body engine. These tests prove it registers **manifest-only** (no JVM needed
to
register), satisfies the engine Protocol, round-trips the coupling triad, is validated against the
**external oracle** (the closed-form Keplerian solution, with J2 disabled), carries a force model
the
two-body tier cannot represent (J2 oblateness genuinely perturbs the orbit), reproduces in-process,
and leaves the reduced-order tier intact as the CX-LOCAL fallback.

The Orekit-dependent tests skip without ``astro-mine-sim[orekit]``; the descriptor/manifest tests
run
regardless (which is what proves registration needs no JVM).
"""

from __future__ import annotations

import math

import pytest

from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.messages.model import Action, ActionBatch, ModeCommand
from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import INERTIAL_J2000
from astro_mine.sim.engines import RegimeEngine, default_engine_registry
from astro_mine.sim.engines.orekit import (
    OREKIT_ORBITAL_ENGINE_DESCRIPTOR,
    orekit_orbital_engine_factory,
)
from astro_mine.sim.runtime import (
    AgentSpec,
    OrekitOrbitalDynamics,
    RngStreams,
    Scenario,
    run_episode,
)
from astro_mine.sim.validation import kepler_propagate, validate_against_oracle

_ENGINE_NAME = "astro-mine.sim.orekit_orbital"
#: A circular low lunar orbit — the relay's reference state.
_MU = 4.902800118e12
_R_M = 1_837_400.0
_V_MPS = math.sqrt(_MU / _R_M)
_PERIOD_S = 2.0 * math.pi * math.sqrt(_R_M**3 / _MU)


@pytest.fixture
def orekit():
    """Skip unless the Orekit binding + its JVM (the ``[orekit]`` extra) are importable."""
    pytest.importorskip("orekit_jpype")
    pytest.importorskip("jdk4py")
    from astro_mine.sim.engines.orekit import _engine

    return _engine


def _scenario(*, j2: float = 0.0, dt_s: float = 60.0, steps: int = 8) -> Scenario:
    return Scenario(
        name="relay",
        seed=3,
        dt_s=dt_s,
        horizon_steps=steps,
        agents=(
            AgentSpec(
                agent_id="relay",
                frame=INERTIAL_J2000,
                initial_position_m=(_R_M, 0.0, 0.0),
                velocity_mps=(0.0, _V_MPS, 0.0),
                battery_soc_j=1.0e6,
                dynamics=OrekitOrbitalDynamics(mu_m3_s2=_MU, j2=j2),
            ),
        ),
    )


# --- registration needs no JVM ---------------------------------------------------


def test_the_orekit_engine_registers_manifest_only_without_a_jvm() -> None:
    # The whole point of the descriptor/factory split: the engine set is registrable — and gated
    # through Core's plugin registry — with no Orekit and no JVM anywhere.
    registry = default_engine_registry()
    manifest = registry.manifest(_ENGINE_NAME)
    assert manifest.kind is PluginKind.REGIME_ENGINE
    assert manifest.determinism_class is DeterminismClass.TOLERANCE
    assert manifest.regimes == [Regime.PROXIMITY_ORBIT]
    assert manifest.attributes["fidelity"]["tier"] == FidelityTier.MASSMODEL.value


def test_the_descriptor_declares_the_inertial_frame_and_a_tolerance_class() -> None:
    d = OREKIT_ORBITAL_ENGINE_DESCRIPTOR
    assert d.frames == (INERTIAL_J2000,)
    # TOLERANCE, not BIT_EXACT: an adaptive step sequence + the JVM's reductions are not
    # bit-portable
    # across builds, so the tier is gated by an error budget rather than a golden hash (sim.md §11).
    assert d.determinism_class is DeterminismClass.TOLERANCE
    assert not d.capability_tags  # no operational_targeting: maneuvers stay out of the commons


# --- the engine itself -----------------------------------------------------------


def test_the_orekit_engine_satisfies_the_regime_engine_protocol(orekit: object) -> None:
    engine = orekit_orbital_engine_factory(_scenario(), RngStreams(0))
    assert isinstance(engine, RegimeEngine)
    assert engine.descriptor is OREKIT_ORBITAL_ENGINE_DESCRIPTOR


def test_orekit_matches_the_analytic_keplerian_oracle_within_budget(orekit: object) -> None:
    # THE acceptance criterion: the higher-fidelity tier validated against an external oracle
    # (conventions.md §11). With J2 disabled the dynamics are pure two-body, so the closed-form
    # Kepler solution is exact truth — and Orekit is an *independent* implementation of it, so
    # agreement here is a real cross-check, not a tautology.
    scenario = _scenario(j2=0.0, dt_s=_PERIOD_S / 16.0, steps=16)
    engine = orekit_orbital_engine_factory(scenario, RngStreams(0))

    actual: list[list[float]] = []
    reference: list[list[float]] = []
    r0, v0 = (_R_M, 0.0, 0.0), (0.0, _V_MPS, 0.0)
    for step in range(1, scenario.horizon_steps + 1):
        engine.advance(scenario.dt_s)
        p = engine.export_coupling_state().by_agent["relay"].pose.translation_m
        actual.append([p.x, p.y, p.z])
        truth, _ = kepler_propagate(r0, v0, _MU, step * scenario.dt_s)
        reference.append(list(truth))

    report = validate_against_oracle(
        actual, reference, budget=1.0e-9, name="orekit-vs-kepler", relative=True
    )
    assert report.passed, f"{report.name}: {report.max_error} > {report.budget} ({report.detail})"


def test_the_j2_term_really_perturbs_the_orbit(orekit: object) -> None:
    # The tier is higher-fidelity in *model*, not just in integrator: J2 oblateness is a force the
    # two-body tier cannot represent at all. Over a full orbit it must move the spacecraft
    # measurably away from the pure two-body solution — otherwise the force model is not wired in.
    dt_s = _PERIOD_S / 8.0
    two_body = orekit_orbital_engine_factory(_scenario(j2=0.0, dt_s=dt_s), RngStreams(0))
    oblate = orekit_orbital_engine_factory(_scenario(j2=2.03e-4, dt_s=dt_s), RngStreams(0))
    for _ in range(8):  # one full orbit
        two_body.advance(dt_s)
        oblate.advance(dt_s)
    a = two_body.export_coupling_state().by_agent["relay"].pose.translation_m
    b = oblate.export_coupling_state().by_agent["relay"].pose.translation_m
    separation = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
    assert separation > 1.0  # metres of divergence after one orbit: a genuine perturbation


def test_orekit_conserves_the_orbit_radius_on_a_circular_orbit(orekit: object) -> None:
    # A sanity invariant a broken propagator would violate immediately.
    scenario = _scenario(dt_s=_PERIOD_S / 32.0, steps=32)
    engine = orekit_orbital_engine_factory(scenario, RngStreams(0))
    for _ in range(scenario.horizon_steps):
        engine.advance(scenario.dt_s)
        p = engine.export_coupling_state().by_agent["relay"].pose.translation_m
        assert math.dist((p.x, p.y, p.z), (0.0, 0.0, 0.0)) == pytest.approx(_R_M, rel=1e-9)


def test_orekit_reproduces_in_process_under_a_fixed_seed(orekit: object) -> None:
    # The in-process half of the TOLERANCE contract (conventions.md §11): the determinism gate
    # holds.
    scenario = _scenario()
    assert (
        run_episode(scenario, engine_factory=orekit_orbital_engine_factory).content_hash
        == run_episode(scenario, engine_factory=orekit_orbital_engine_factory).content_hash
    )


def test_the_coupling_triad_round_trips(orekit: object) -> None:
    engine = orekit_orbital_engine_factory(_scenario(), RngStreams(0))
    engine.advance(60.0)
    snapshot = engine.export_coupling_state()

    other = orekit_orbital_engine_factory(_scenario(), RngStreams(0))
    other.import_coupling_state(snapshot)  # re-seeds the Orekit propagator from the snapshot
    restored = other.export_coupling_state().by_agent["relay"].pose.translation_m
    original = snapshot.by_agent["relay"].pose.translation_m
    assert (restored.x, restored.y, restored.z) == pytest.approx(
        (original.x, original.y, original.z), rel=1e-12
    )
    # and both engines then propagate to the same place — the import restored the full state.
    engine.advance(60.0)
    other.advance(60.0)
    a = engine.export_coupling_state().by_agent["relay"].pose.translation_m
    b = other.export_coupling_state().by_agent["relay"].pose.translation_m
    assert (a.x, a.y, a.z) == pytest.approx((b.x, b.y, b.z), rel=1e-9)


def test_only_a_mode_command_is_honored_no_maneuvers(orekit: object) -> None:
    # A Δv/targeting capability is gated out of the open commons (operational_targeting), so the
    # orbital tiers honor a mode command and nothing else — Orekit's maneuver machinery stays
    # unwired.
    engine = orekit_orbital_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(
        ActionBatch(
            actions=[
                Action(
                    agent_id="relay",
                    kind=ActionKind.MODE,
                    mode=ModeCommand(mode="safe"),
                )
            ]
        )
    )
    assert engine.export_coupling_state().by_agent["relay"].mode == "safe"


def test_retire_drops_an_orbiter(orekit: object) -> None:
    engine = orekit_orbital_engine_factory(_scenario(), RngStreams(0))
    engine.retire(["relay"])
    assert engine.export_coupling_state().samples == ()


def test_a_scenario_with_no_orekit_agents_yields_an_empty_engine(orekit: object) -> None:
    # The heterogeneous co-step: an engine owns only its regime's agents (RM-P0-SIM-04).
    scenario = Scenario(name="surface", agents=(AgentSpec(agent_id="rover"),))
    engine = orekit_orbital_engine_factory(scenario, RngStreams(0))
    assert engine.export_coupling_state().samples == ()
