"""RM-P1-SIM-06 — the high-fidelity DEM granular engine (ground-truth excavation).

A soft-sphere particle bed a blade excavates behind the ``RegimeEngine`` waist: it settles to
static equilibrium (floor reaction ≈ weight), conserves momentum under internal contacts,
reproduces exactly under a seed, and exposes the draft force / particle state a Surrogate
learns. Configs are kept small so the O(N²) CPU kernel stays fast in CI.
"""

from __future__ import annotations

import numpy as np

from astro_mine.core.messages.enums import (
    ActionKind,
    ExcavationPattern,
    ExcavationTool,
    TaskKind,
)
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ExcavateTask,
    ModeCommand,
    TaskDirective,
    Vec3,
    Volume,
)
from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines import RegimeEngine, default_engine_registry
from astro_mine.sim.engines.dem import DEM_GRANULAR_ENGINE_DESCRIPTOR, dem_granular_engine_factory
from astro_mine.sim.engines.dem._engine import DemGranularEngine
from astro_mine.sim.engines.dem._solver import DemBed, build_params, substep
from astro_mine.sim.runtime import AgentSpec, DemGranularDynamics, RngStreams, Scenario


def _scenario(*, agents: tuple[AgentSpec, ...] | None = None, **dyn: object) -> Scenario:
    params: dict[str, object] = {"n_particles": 40, "settle_substeps": 300, "bed_width_m": 0.4}
    params.update(dyn)
    default = (
        AgentSpec(
            agent_id="digger",
            battery_soc_j=1.0e6,
            dynamics=DemGranularDynamics(**params),  # type: ignore[arg-type]
        ),
    )
    return Scenario(name="dem-dig", horizon_steps=1, dt_s=0.05, agents=agents or default)


def _build(scenario: Scenario, seed: int = 0) -> DemGranularEngine:
    engine = dem_granular_engine_factory(scenario, RngStreams(seed))
    assert isinstance(engine, DemGranularEngine)
    return engine


def _engine(**dyn: object) -> DemGranularEngine:
    return _build(_scenario(**dyn))


def _params(engine: DemGranularEngine) -> object:
    return engine._states["digger"].params


def _excavate(target: float | None) -> ActionBatch:
    return ActionBatch(
        actions=[
            Action(
                agent_id="digger",
                kind=ActionKind.TASK,
                task=TaskDirective(
                    task_kind=TaskKind.EXCAVATE,
                    excavate=ExcavateTask(
                        region=Volume(
                            frame="MOON_ME",
                            center_m=Vec3(x=0.0, y=0.0, z=0.0),
                            dimensions_m=Vec3(x=1.0, y=1.0, z=1.0),
                        ),
                        tool=ExcavationTool.BUCKET,
                        pattern=ExcavationPattern.TRENCH,
                        target_volume_m3=target,
                    ),
                ),
            )
        ]
    )


# -- registration / descriptor --------------------------------------------------------------


def test_descriptor_declares_surface_articulated_tolerance() -> None:
    d = DEM_GRANULAR_ENGINE_DESCRIPTOR
    assert d.name == "astro-mine.sim.dem_granular"
    assert d.regimes == (Regime.SURFACE,)
    assert d.frames == (MOON_BODY_FIXED,)
    assert d.fidelity.tier is FidelityTier.ARTICULATED
    assert d.determinism_class is DeterminismClass.TOLERANCE
    assert d.to_manifest().kind is PluginKind.REGIME_ENGINE


def test_registration_is_manifest_only_and_resolves() -> None:
    # The descriptor registers via the default registry (manifest-only) without the factory
    # ever running the numpy kernel — the [dem] extra is only needed to build an engine.
    registry = default_engine_registry()
    assert "astro-mine.sim.dem_granular" in registry.names
    assert registry.manifest("astro-mine.sim.dem_granular").kind is PluginKind.REGIME_ENGINE


def test_engine_satisfies_regime_engine_protocol() -> None:
    engine = _engine()
    assert isinstance(engine, RegimeEngine)
    assert engine.descriptor is DEM_GRANULAR_ENGINE_DESCRIPTOR


def test_factory_builds_only_dem_agents() -> None:
    scenario = _scenario(
        agents=(
            AgentSpec(
                agent_id="digger",
                dynamics=DemGranularDynamics(n_particles=30, settle_substeps=150),
            ),
            AgentSpec(agent_id="kin"),  # kinematic default — skipped
        )
    )
    assert set(_build(scenario).export_coupling_state().by_agent) == {"digger"}


# -- physics: settling, force balance, excavation -------------------------------------------


def test_bed_settles_and_excavation_rebaselines_to_zero() -> None:
    engine = _engine(settle_substeps=600)
    params = _params(engine)
    assert engine.bed("digger").kinetic_energy_j(params.mass_kg) < 1.0  # came to rest
    assert engine.excavated_mass_kg("digger") == 0.0  # measured from the settled rest state


def test_settled_bed_balances_its_weight_on_the_floor() -> None:
    engine = _engine(n_particles=60, settle_substeps=1200, bed_width_m=0.5)
    p = _params(engine)
    weight = p.n_particles * p.mass_kg * p.gravity_m_s2
    assert abs(engine.floor_reaction_n("digger") - weight) / weight < 0.05  # static equilibrium


def test_digging_builds_draft_force_and_excavates_material() -> None:
    engine = _engine(tool_speed_mps=0.08)
    engine.apply_actions(_excavate(None))
    drafts = []
    for _ in range(8):
        engine.advance(0.05)
        drafts.append(engine.tool_reaction_force_n("digger"))
    assert max(drafts) > 0.0  # the blade meets resistance
    assert engine.excavated_mass_kg("digger") > 0.0  # material is displaced
    assert engine.excavated_volume_m3("digger") > 0.0


def test_excavate_target_volume_stops_the_dig() -> None:
    engine = _engine(tool_speed_mps=0.08)
    engine.apply_actions(_excavate(0.002))  # a small target the sweep reaches quickly
    for _ in range(30):
        engine.advance(0.05)
    assert engine._states["digger"].digging is False
    assert engine.excavated_mass_kg("digger") > 0.0


def test_blade_sweeping_off_the_bed_stops_the_dig() -> None:
    engine = _engine(n_particles=30, settle_substeps=150, bed_width_m=0.3, tool_speed_mps=0.3)
    engine.apply_actions(_excavate(None))  # no target: digs until the blade exits
    for _ in range(40):
        engine.advance(0.05)
    assert engine._states["digger"].digging is False


def test_battery_is_drawn_while_digging() -> None:
    engine = _engine(tool_speed_mps=0.1)
    soc0 = engine.export_coupling_state().by_agent["digger"].battery_soc_j
    engine.apply_actions(_excavate(None))
    for _ in range(8):
        engine.advance(0.05)
    soc1 = engine.export_coupling_state().by_agent["digger"].battery_soc_j
    assert soc1 is not None and soc0 is not None and soc1 < soc0


# -- accessors / actions --------------------------------------------------------------------


def test_particles_accessor_returns_positions_and_velocities() -> None:
    engine = _engine(n_particles=40)
    pos, vel = engine.particles("digger")
    assert pos.shape == (40, 2)
    assert vel.shape == (40, 2)


def test_mode_command_sets_mode() -> None:
    engine = _engine()
    engine.apply_actions(
        ActionBatch(
            actions=[
                Action(agent_id="digger", kind=ActionKind.MODE, mode=ModeCommand(mode="excavating"))
            ]
        )
    )
    assert engine.export_coupling_state().by_agent["digger"].mode == "excavating"


def test_action_for_an_unowned_agent_is_ignored() -> None:
    engine = _engine()
    engine.apply_actions(
        ActionBatch(
            actions=[Action(agent_id="ghost", kind=ActionKind.MODE, mode=ModeCommand(mode="x"))]
        )
    )
    assert set(engine.export_coupling_state().by_agent) == {"digger"}


def test_retire_drops_the_agent() -> None:
    engine = _engine()
    engine.retire(["digger"])
    assert engine.export_coupling_state().by_agent == {}


def test_import_coupling_ignores_snapshots_without_this_agent() -> None:
    from astro_mine.sim.engines.adapter import CouplingState

    engine = _engine()
    before = engine.export_coupling_state().by_agent["digger"].pose
    engine.import_coupling_state(CouplingState(sim_time_s=0.0, samples=()))  # nothing to apply
    assert engine.export_coupling_state().by_agent["digger"].pose == before


def test_coupling_state_round_trips() -> None:
    engine = _engine(tool_speed_mps=0.08)
    engine.apply_actions(_excavate(None))
    engine.advance(0.05)
    snapshot = engine.export_coupling_state()
    engine.advance(0.05)
    engine.import_coupling_state(snapshot)
    restored = engine.export_coupling_state().by_agent["digger"]
    assert restored.battery_soc_j == snapshot.by_agent["digger"].battery_soc_j
    assert restored.pose == snapshot.by_agent["digger"].pose


# -- determinism (TOLERANCE: in-process reproducibility) ------------------------------------


def test_same_seed_runs_reproduce_exactly() -> None:
    def run() -> tuple[float, np.ndarray]:
        engine = _build(_scenario(tool_speed_mps=0.08), seed=3)
        engine.apply_actions(_excavate(None))
        for _ in range(6):
            engine.advance(0.05)
        return engine.tool_reaction_force_n("digger"), engine.particles("digger")[0].copy()

    r1, p1 = run()
    r2, p2 = run()
    assert r1 == r2
    assert np.array_equal(p1, p2)


def test_dem_engine_runs_and_reproduces_in_a_real_episode() -> None:
    # Drive the DEM engine through the full episode loop (reset/step/Trace) — this exercises the
    # scheduler / provenance / coupling kind-routing wiring and the TOLERANCE reproducibility gate.
    from astro_mine.sim.runtime import run_episode

    scenario = _scenario(n_particles=24, settle_substeps=120, bed_width_m=0.3)
    t1 = run_episode(scenario, seed=1, engine_factory=dem_granular_engine_factory)
    t2 = run_episode(scenario, seed=1, engine_factory=dem_granular_engine_factory)
    assert t1.content_hash == t2.content_hash


# -- solver invariants ----------------------------------------------------------------------


def test_pairwise_collision_conserves_momentum_and_recovers_restitution() -> None:
    p = build_params(
        n_particles=2,
        particle_radius_m=0.02,
        regolith_density_kg_m3=1500.0,
        contact_stiffness_n_m=5.0e4,
        restitution=0.9,
        friction_coeff=0.6,
        gravity_m_s2=0.0,  # free space
        bed_width_m=1.0e6,  # walls far away
        tool_x0_m=1.0e9,
        tool_height_m=0.0,
        tool_speed_mps=0.0,
    )
    pos = np.array([[1.0, 5.0], [1.038, 5.0]])  # head-on along x, clear of floor/walls
    bed = DemBed(
        pos=pos.copy(), vel=np.array([[0.5, 0.0], [-0.5, 0.0]]), pos0=pos.copy(), tool_x_m=1e9
    )
    m0 = bed.total_momentum(p.mass_kg).copy()
    for _ in range(60):
        substep(bed, p, p.dt_internal_s, tool_active=False)
    assert np.abs(bed.total_momentum(p.mass_kg) - m0).max() < 1e-9  # exact conservation
    # restitution ~0.9: the pair rebounds to ~±0.45 (equal masses, head-on).
    assert bed.vel[0, 0] < -0.4 and bed.vel[1, 0] > 0.4


def test_separated_particles_feel_no_contact_force() -> None:
    p = build_params(
        n_particles=2,
        particle_radius_m=0.02,
        regolith_density_kg_m3=1500.0,
        contact_stiffness_n_m=5.0e4,
        restitution=0.3,
        friction_coeff=0.6,
        gravity_m_s2=0.0,
        bed_width_m=1.0e6,
        tool_x0_m=1.0e9,
        tool_height_m=0.0,
        tool_speed_mps=0.0,
    )
    pos = np.array([[1.0, 5.0], [2.0, 5.0]])  # 1 m apart >> 2r
    bed = DemBed(pos=pos.copy(), vel=np.zeros((2, 2)), pos0=pos.copy(), tool_x_m=1e9)
    substep(bed, p, p.dt_internal_s, tool_active=False)
    assert np.array_equal(bed.vel, np.zeros((2, 2)))  # no contact, no gravity → no force
