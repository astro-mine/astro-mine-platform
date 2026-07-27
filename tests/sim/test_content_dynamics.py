"""RM-P0-SIM-03 — engine dynamics sourced from the resolved Worlds/Fleet content a scenario pins.

The gap this closes: ``runtime/scenario.py``'s dynamics parameters — mass, regolith density, DEM
numerics, tool geometry — were hand-authored scenario placeholders, explicitly commented "plumbed
later". They are now read from the content the scenario actually pins: the **asset** side from the
resolved Core ``Asset`` (Fleet SADF), the **world** side from the resolved Core ``WorldProvider``
(Worlds) sampled at the agent's site.

Sim still imports only Core: a SADF asset and a WorldProvider are Core types, so nothing here
reaches into ``astro_mine.fleet`` or ``astro_mine.worlds`` (conventions.md §1.1).
"""

from __future__ import annotations

import math

import pytest

from astro_mine.core.sadf.enums import ContactElementKind
from astro_mine.core.sadf.model import (
    Actuator,
    Asset,
    Body,
    ContactElement,
    Identity,
    Inertia,
    Mobility,
    Vec3,
)
from astro_mine.core.units import MOON_BODY_FIXED, Epoch, ReferenceFrame
from astro_mine.core.world import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
)
from astro_mine.sim.runtime import (
    ResolvedAsset,
    dem_granular_dynamics_from_content,
    granular_dynamics_from_content,
    mjx_dynamics_from_content,
    mobility_dynamics_from_content,
    mujoco_dynamics_from_content,
    site_conditions,
)
from astro_mine.sim.runtime.scenario import (
    LUNAR_GRAVITY_M_S2,
    LUNAR_REGOLITH_DENSITY_KG_M3,
)

# The pinned world's terramechanics — deliberately unlike the reduced-order lunar defaults, so a
# test that reads them cannot pass by accident against a hard-coded constant.
_DENSITY = 1740.0
_FRICTION_DEG = 41.0
_BEARING_PA = 9.5e4
_GRAVITY = 1.315
# The pinned asset's properties, likewise distinct from the defaults.
_CHASSIS_KG = 180.0
_PAYLOAD_KG = 95.0
_WHEEL_RADIUS = 0.33
_TOP_SPEED = 0.42
_WHEEL_TORQUE = 77.0
_TOOL_WIDTH = 0.55
_TOOL_HEIGHT = 0.22


class _PinnedWorld:
    """A Core ``WorldProvider`` stand-in for a resolved Worlds bundle."""

    @property
    def frame(self) -> ReferenceFrame:
        return MOON_BODY_FIXED

    def sample(
        self, position: tuple[float, float, float], *, epoch: Epoch | None = None
    ) -> SurfacePoint:
        return SurfacePoint(
            frame=MOON_BODY_FIXED,
            elevation_m=0.0,
            surface_normal=(0.0, 0.0, 1.0),
            gravity=(0.0, 0.0, -_GRAVITY),
            illumination=Illumination(state=IlluminationState.LIT, solar_flux_w_m2=1361.0),
            temperature_k=120.0,
            regolith=RegolithParams(
                bulk_density_kg_m3=_DENSITY,
                friction_angle_deg=_FRICTION_DEG,
                bearing_capacity_pa=_BEARING_PA,
            ),
        )


class _BareWorld(_PinnedWorld):
    """A world that models terrain but declares *no* regolith — the partial-coverage case."""

    def sample(
        self, position: tuple[float, float, float], *, epoch: Epoch | None = None
    ) -> SurfacePoint:
        base = super().sample(position, epoch=epoch)
        return SurfacePoint(
            frame=base.frame,
            elevation_m=base.elevation_m,
            surface_normal=base.surface_normal,
            gravity=base.gravity,
            illumination=base.illumination,
            temperature_k=base.temperature_k,
            regolith=RegolithParams(),  # every field None
        )


def _inertia() -> Inertia:
    return Inertia(ixx=1.0, iyy=1.0, izz=1.0)


def _excavator_asset() -> ResolvedAsset:
    """A Fleet SADF excavator-rover: two rigid bodies, drive actuators, a wheel, and a dig tool."""
    asset = Asset(
        identity=Identity(
            id="astro-mine.fleet.excavator", name="Excavator", version="0.1.0", kind="rover"
        ),
        root_frame="body",
        bodies=[
            Body(
                name="chassis",
                frame="body",
                mass_kg=_CHASSIS_KG,
                center_of_mass_m=Vec3(x=0.0, y=0.0, z=0.0),
                inertia_kg_m2=_inertia(),
            ),
            Body(
                name="payload",
                frame="body",
                mass_kg=_PAYLOAD_KG,
                center_of_mass_m=Vec3(x=0.0, y=0.0, z=0.2),
                inertia_kg_m2=_inertia(),
            ),
        ],
        actuators=[
            Actuator(name="drive", velocity=_TOP_SPEED, torque_nm=_WHEEL_TORQUE),
        ],
        mobility=Mobility(
            contact=[
                ContactElement(
                    kind=ContactElementKind.WHEEL,
                    dimensions_m=Vec3(x=0.12, y=0.12, z=_WHEEL_RADIUS),
                ),
                ContactElement(
                    kind=ContactElementKind.TOOL,
                    dimensions_m=Vec3(x=_TOOL_WIDTH, y=0.1, z=_TOOL_HEIGHT),
                ),
            ]
        ),
    )
    return ResolvedAsset(asset=asset, content_hash="sha256:" + "ab" * 32)


def _bare_asset() -> ResolvedAsset:
    """An asset declaring none of the physical detail — the fallback case."""
    asset = Asset(
        identity=Identity(id="bare", name="Bare", version="0.1.0", kind="rover"),
        root_frame="body",
    )
    return ResolvedAsset(asset=asset, content_hash="sha256:" + "cd" * 32)


# --- the world side --------------------------------------------------------------


def test_site_conditions_come_from_the_pinned_world() -> None:
    site = site_conditions(_PinnedWorld(), (10.0, 20.0, 0.0))
    assert site.regolith_density_kg_m3 == _DENSITY
    assert site.friction_angle_deg == _FRICTION_DEG
    assert site.bearing_capacity_pa == _BEARING_PA
    assert site.gravity_m_s2 == pytest.approx(_GRAVITY)  # magnitude of the world's gravity vector


def test_an_unpinned_scenario_still_gets_the_reduced_order_defaults() -> None:
    # No world pinned ⇒ the documented Phase-0 lunar constants, so an existing scenario is
    # unaffected.
    site = site_conditions(None, (0.0, 0.0, 0.0))
    assert site.regolith_density_kg_m3 == LUNAR_REGOLITH_DENSITY_KG_M3
    assert site.gravity_m_s2 == LUNAR_GRAVITY_M_S2


def test_a_world_that_models_no_regolith_degrades_field_by_field() -> None:
    # A partial world still yields a usable block: it contributes what it models (gravity) and falls
    # back only where it does not (regolith).
    site = site_conditions(_BareWorld(), (0.0, 0.0, 0.0))
    assert site.gravity_m_s2 == pytest.approx(_GRAVITY)  # the world's
    assert site.regolith_density_kg_m3 == LUNAR_REGOLITH_DENSITY_KG_M3  # the fallback


# --- the asset side --------------------------------------------------------------


def test_mass_is_the_sum_of_the_assets_sadf_bodies() -> None:
    dynamics = mobility_dynamics_from_content(_excavator_asset(), world=_PinnedWorld())
    assert dynamics.mass_kg == pytest.approx(_CHASSIS_KG + _PAYLOAD_KG)  # not a scenario literal


def test_the_drawbar_pull_limit_is_derived_from_the_pinned_world_and_asset() -> None:
    # max_traction_n is no longer typed in: it is the friction cone of the asset's weight on the
    # pinned regolith — mu * m * g, with mu = tan(phi). So the traction limit follows the terrain.
    dynamics = mobility_dynamics_from_content(_excavator_asset(), world=_PinnedWorld())
    mass = _CHASSIS_KG + _PAYLOAD_KG
    expected = math.tan(math.radians(_FRICTION_DEG)) * mass * _GRAVITY
    assert dynamics.max_traction_n == pytest.approx(expected)
    assert dynamics.max_speed_mps == pytest.approx(_TOP_SPEED)  # the SADF actuator's limit


def test_the_mujoco_contact_block_is_fully_content_sourced() -> None:
    dynamics = mujoco_dynamics_from_content(_excavator_asset(), world=_PinnedWorld())
    # Asset side (Fleet SADF).
    assert dynamics.mass_kg == pytest.approx(_CHASSIS_KG + _PAYLOAD_KG)
    assert dynamics.wheel_radius_m == pytest.approx(_WHEEL_RADIUS)
    assert dynamics.max_speed_mps == pytest.approx(_TOP_SPEED)
    assert dynamics.wheel_torque_nm == pytest.approx(_WHEEL_TORQUE)
    # World side (Worlds RegolithParams + gravity) — the friction cone the rover drives on.
    assert dynamics.friction_angle_deg == pytest.approx(_FRICTION_DEG)
    assert dynamics.bearing_capacity_pa == pytest.approx(_BEARING_PA)
    assert dynamics.gravity_m_s2 == pytest.approx(_GRAVITY)


def test_the_mjx_block_describes_the_same_machine_as_the_mujoco_block() -> None:
    # The CPU and GPU contact tiers must step the same rover, so they source it identically.
    cpu = mujoco_dynamics_from_content(_excavator_asset(), world=_PinnedWorld())
    gpu = mjx_dynamics_from_content(_excavator_asset(), world=_PinnedWorld(), batch_size=8)
    assert (gpu.mass_kg, gpu.wheel_radius_m, gpu.friction_angle_deg, gpu.gravity_m_s2) == (
        cpu.mass_kg,
        cpu.wheel_radius_m,
        cpu.friction_angle_deg,
        cpu.gravity_m_s2,
    )
    assert gpu.batch_size == 8


def test_the_granular_block_takes_its_regolith_density_from_the_world() -> None:
    # regolith_density_kg_m3 turns excavated volume into excavated *mass*, so it feeds Bench's
    # water_mass metric directly — it must come from the pinned world, not a scenario constant.
    dynamics = granular_dynamics_from_content(_excavator_asset(), world=_PinnedWorld())
    assert dynamics.regolith_density_kg_m3 == pytest.approx(_DENSITY)


def test_the_dem_block_sources_terramechanics_gravity_and_tool_geometry() -> None:
    # The DEM tier shipped with three "plumbed later" gaps; all three now resolve from content.
    dynamics = dem_granular_dynamics_from_content(_excavator_asset(), world=_PinnedWorld())
    assert dynamics.regolith_density_kg_m3 == pytest.approx(_DENSITY)  # Worlds RegolithParams
    assert dynamics.friction_coeff == pytest.approx(math.tan(math.radians(_FRICTION_DEG)))
    assert dynamics.gravity_m_s2 == pytest.approx(_GRAVITY)  # Worlds gravity model
    assert dynamics.tool_height_m == pytest.approx(_TOOL_HEIGHT)  # Fleet excavator SADF tool
    # The bed is sized around the pinned blade rather than a fixed literal.
    assert dynamics.bed_width_m == pytest.approx(max(_TOOL_WIDTH * 2.0, 0.6))
    # The DEM *numerics* stay on the block: they size the reference bed, which is a solver choice,
    # not a property of the world or the asset.
    assert dynamics.n_particles > 0 and dynamics.contact_stiffness_n_m > 0.0


def test_a_bare_asset_and_no_world_reproduce_the_reduced_order_defaults() -> None:
    # The fallback path: content that declares nothing yields exactly the Phase-0 hand-authored
    # values, so adopting the builders never perturbs an un-pinned scenario.
    dynamics = mujoco_dynamics_from_content(_bare_asset(), world=None)
    assert dynamics.mass_kg == 250.0
    assert dynamics.gravity_m_s2 == LUNAR_GRAVITY_M_S2
    assert dynamics.friction_angle_deg == 31.0
    dem = dem_granular_dynamics_from_content(_bare_asset(), world=None)
    assert dem.regolith_density_kg_m3 == LUNAR_REGOLITH_DENSITY_KG_M3


def test_the_content_sourced_block_actually_drives_a_contact_engine() -> None:
    # End-to-end: a block built from pinned content is a valid dynamics block a real engine steps.
    pytest.importorskip("mujoco")
    from astro_mine.core.units import MOON_BODY_FIXED as FRAME
    from astro_mine.sim.engines import mujoco_mobility_engine_factory
    from astro_mine.sim.runtime import AgentSpec, RngStreams, Scenario

    dynamics = mujoco_dynamics_from_content(_excavator_asset(), world=_PinnedWorld())
    scenario = Scenario(
        name="content-sourced",
        dt_s=0.5,
        horizon_steps=2,
        agents=(
            AgentSpec(
                agent_id="exc",
                frame=FRAME,
                velocity_mps=(0.3, 0.0, 0.0),
                battery_soc_j=1.0e6,
                dynamics=dynamics,
            ),
        ),
    )
    engine = mujoco_mobility_engine_factory(scenario, RngStreams(0))
    for _ in range(4):
        engine.advance(0.5)
    assert engine.export_coupling_state().by_agent["exc"].pose.translation_m.x > 0.1
