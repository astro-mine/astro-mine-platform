"""Scenario specification + loader (RM-P0-SIM-01).

A :class:`Scenario` is the reproducible declaration of *what* an episode contains — its agents and
their initial state, the base timestep, the horizon, and the ``seed`` / ``start_epoch`` that fix
determinism. It is a validated Pydantic model (``extra="forbid"``, the Core house style) so a typo'd
field fails loudly, and it loads from a JSON document so a run is captured by file.

Each agent declares a typed ``dynamics`` block — a discriminated union over the regime engines
(RM-P0-SIM-03): ``kinematic`` (the reference default), ``orbital``, ``mobility``, ``manipulation``,
``granular``. The block carries the *minimal physical parameters* the agent's engine needs; the
**Worlds / Prospect / Fleet data plumbing** that would source those fields from real world/asset
models is deliberately still *out of scope* here (deferred per the RM-P0-SIM-03 scope), so the
parameters live on the scenario for now.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astro_mine.core.sadf.enums import JointType
from astro_mine.core.sadf.model import FidelityProfile, PowerBudget, Sensor, ThermalBudget
from astro_mine.core.units import J2000_EPOCH, MOON_BODY_FIXED, Epoch, ReferenceFrame
from astro_mine.sim.engines._rover_mjcf import DEFAULT_WHEEL_RADIUS_M
from astro_mine.sim.scheduler import FidelityPolicy

__all__ = [
    "AgentSpec",
    "BraxContactDynamics",
    "DemGranularDynamics",
    "Dynamics",
    "GranularDynamics",
    "IsruSpec",
    "JointSpec",
    "KinematicDynamics",
    "ManipulationDynamics",
    "MjxContactDynamics",
    "MobilityDynamics",
    "MujocoMobilityDynamics",
    "OrbitalDynamics",
    "OrekitOrbitalDynamics",
    "Scenario",
    "load_scenario",
]

#: The default operating modes in which an ISRU asset extracts (RM-P1-SIM-02); mirrors
#: :data:`astro_mine.sim.isru.DEFAULT_EXTRACTION_MODES` as a stable, sorted tuple for the schema.
_DEFAULT_EXTRACTION_MODES: tuple[str, ...] = ("dig", "drill", "excavate", "extract", "isru")

#: A 3-vector literal (SI metres or m/s) as it appears in a scenario document.
Vec3Spec = tuple[float, float, float]

#: Moon GM (gravitational parameter), m³/s² — the reduced-order orbital default (DE440-class value).
#: The authoritative source is the resolved Worlds gravity model, which
#: :func:`~astro_mine.sim.runtime.content.dynamics_from_content` now reads when a scenario pins one;
#: these constants are the fallback for a scenario that pins no content.
MOON_MU_M3_S2 = 4.902800118e12
#: Bulk regolith density, kg/m³ — a reduced-order lunar-mare default for the granular model (the
#: authoritative source is the resolved Worlds ``RegolithParams``, RM-P0-WORLDS-05).
LUNAR_REGOLITH_DENSITY_KG_M3 = 1500.0
#: Lunar surface gravity, m/s² (the authoritative source is the resolved Worlds gravity model).
LUNAR_GRAVITY_M_S2 = 1.62
#: The Moon's J2 zonal coefficient — the oblateness term the Orekit orbital tier perturbs with
#: (GRGM1200-class value; pure two-body motion cannot represent it at all).
LUNAR_J2 = 2.03e-4
#: The Moon's mean radius, m — the reference radius the J2 term is normalized against.
MOON_RADIUS_M = 1_737_400.0


class _Spec(BaseModel):
    """Base for scenario models: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


# --- per-agent dynamics: the engine-physics parameters (RM-P0-SIM-03) ------------
# Each variant carries the minimal SI parameters its regime engine needs and is tagged by
# ``kind`` so a scenario document selects the engine declaratively. The defaults are
# reduced-order lunar-anchor values; the authoritative sources (Worlds regolith/gravity,
# Fleet SADF mass/power) are plumbed in a later item.


class KinematicDynamics(_Spec):
    """Constant-velocity reference dynamics — the SIM-02 ``KinematicEngine`` default."""

    kind: Literal["kinematic"] = "kinematic"


class OrbitalDynamics(_Spec):
    """Two-body orbital dynamics for an orbiter (the relay). The initial position/velocity
    are the agent's ``initial_position_m`` / ``velocity_mps``, resolved in the agent's
    (inertial) frame; ``mu_m3_s2`` is the central body's gravitational parameter."""

    kind: Literal["orbital"] = "orbital"
    mu_m3_s2: float = Field(default=MOON_MU_M3_S2, gt=0.0)
    substeps: int = Field(default=8, gt=0)
    station_keeping_power_w: float = Field(default=0.0, ge=0.0)


class MobilityDynamics(_Spec):
    """Mid-fidelity wheeled-rover mobility. Acceleration is capped by a terramechanics
    drawbar-pull limit (``max_traction_n`` / ``mass_kg``) and speed by ``max_speed_mps``;
    battery draw is ``idle_power_w`` plus ``drive_power_w_per_mps`` * speed."""

    kind: Literal["mobility"] = "mobility"
    mass_kg: float = Field(gt=0.0)
    max_speed_mps: float = Field(gt=0.0)
    max_traction_n: float = Field(gt=0.0)
    idle_power_w: float = Field(default=0.0, ge=0.0)
    drive_power_w_per_mps: float = Field(default=0.0, ge=0.0)


class JointSpec(_Spec):
    """One joint of a reduced-order articulated linkage. ``axis`` is the joint axis in the
    parent link frame; ``link_length_m`` is the offset (along local +x at rest) to the next joint;
    ``rate_limit`` bounds motion toward a setpoint (rad/s for revolute, m/s for
    prismatic). Non-revolute/prismatic types are treated as fixed."""

    name: str
    joint_type: JointType
    axis: Vec3Spec = (0.0, 0.0, 1.0)
    link_length_m: float = Field(default=0.0, ge=0.0)
    rate_limit: float = Field(gt=0.0)
    lower: float | None = None
    upper: float | None = None
    initial: float = 0.0


class ManipulationDynamics(_Spec):
    """Reduced-order articulated linkage (excavator arm). Forward kinematics over an ordered
    joint chain maps joint coordinates to the end-effector pose."""

    kind: Literal["manipulation"] = "manipulation"
    joints: tuple[JointSpec, ...]
    base_offset_m: Vec3Spec = (0.0, 0.0, 0.0)
    actuation_power_w: float = Field(default=0.0, ge=0.0)

    @field_validator("joints")
    @classmethod
    def _non_empty(cls, joints: tuple[JointSpec, ...]) -> tuple[JointSpec, ...]:
        if not joints:
            raise ValueError("a manipulation linkage needs at least one joint")
        return joints


class GranularDynamics(_Spec):
    """Reduced-order granular excavation. A dig removes volume at ``max_dig_rate_m3_s``;
    the excavated mass is ``regolith_density_kg_m3`` * volume and the battery draw is
    ``specific_energy_j_per_m3`` * volume (a reduced resistive-force/energy model)."""

    kind: Literal["granular"] = "granular"
    regolith_density_kg_m3: float = Field(default=LUNAR_REGOLITH_DENSITY_KG_M3, gt=0.0)
    specific_energy_j_per_m3: float = Field(default=1.0e5, gt=0.0)
    max_dig_rate_m3_s: float = Field(gt=0.0)


class DemGranularDynamics(_Spec):
    """High-fidelity DEM granular excavation (RM-P1-SIM-06) — the ground-truth oracle tier.

    A 2D soft-sphere particle bed a blade excavates (:mod:`astro_mine.sim.engines.dem`). The
    **terramechanics** fields (density, friction, restitution) source from Worlds'
    ``RegolithParams`` and the **tool** fields (width position, height, speed) from the Fleet
    excavator SADF — carried on the scenario for now, like the reduced-order params above. The **DEM
    numerics** (particle count/radius, contact stiffness, bed width, settle steps) size the
    reference bed; keep them modest — the engine is CPU-bound and sub-millisecond per step.
    Selecting this tier requires the ``astro-mine-sim[dem]`` extra (numpy).
    """

    kind: Literal["dem_granular"] = "dem_granular"
    # Terramechanics (Worlds RegolithParams, plumbed later).
    regolith_density_kg_m3: float = Field(default=LUNAR_REGOLITH_DENSITY_KG_M3, gt=0.0)
    friction_coeff: float = Field(default=0.6, ge=0.0)
    restitution: float = Field(default=0.3, gt=0.0, lt=1.0)
    # DEM numerics.
    n_particles: int = Field(default=90, gt=0)
    particle_radius_m: float = Field(default=0.02, gt=0.0)
    contact_stiffness_n_m: float = Field(default=5.0e4, gt=0.0)
    bed_width_m: float = Field(default=0.6, gt=0.0)
    gravity_m_s2: float = Field(default=LUNAR_GRAVITY_M_S2, gt=0.0)
    settle_substeps: int = Field(default=1200, ge=0)
    # Tool geometry/motion (Fleet excavator SADF, plumbed later).
    tool_x0_m: float = Field(default=0.04, ge=0.0)
    tool_height_m: float = Field(default=0.10, gt=0.0)
    tool_speed_mps: float = Field(default=0.04, gt=0.0)


class BraxContactDynamics(_Spec):
    """JAX (Brax/MJX) GPU-vectorizable surface mobility/contact (RM-P1-SIM-04) — the training tier.

    A low-fidelity traction-limited, speed-capped rover integrated by a ``jax.numpy`` kernel that
    ``jax.vmap`` batches across agents and across thousands of parallel envs — the fast-contact
    engine [Learn](learn.md) trains swarm-scale policies on (sim.md §8, §11). Behaviourally it is
    the reduced-order :class:`MobilityDynamics` model (so the drawbar-pull oracle cross-checks it);
    its value is the XLA-compiled batched step. Selecting this tier requires the
    ``astro-mine-sim[brax]`` extra (jax/brax/mujoco); the Ray fan-out also needs ``[ray]``.

    ``batch_size`` is the default number of parallel envs the vectorized rollout builds;
    ``init_speed_jitter_mps`` (0 by default) is the std-dev of a seeded per-env/per-agent initial-
    velocity perturbation for swarm-scale domain randomization — at 0 the tier reduces exactly to
    the deterministic mobility model."""

    kind: Literal["brax_contact"] = "brax_contact"
    mass_kg: float = Field(gt=0.0)
    max_speed_mps: float = Field(gt=0.0)
    max_traction_n: float = Field(gt=0.0)
    idle_power_w: float = Field(default=0.0, ge=0.0)
    drive_power_w_per_mps: float = Field(default=0.0, ge=0.0)
    #: Default parallel-env count for the GPU-batched rollout (sim.md §8).
    batch_size: int = Field(default=1024, gt=0)
    #: Seeded domain-randomization std-dev for initial velocity (m/s); 0 ⇒ deterministic.
    init_speed_jitter_mps: float = Field(default=0.0, ge=0.0)


class OrekitOrbitalDynamics(_Spec):
    """Orekit higher-fidelity orbital dynamics (RM-P0-SIM-03) — the flight-grade orbital tier.

    The same orbital regime as :class:`OrbitalDynamics`, propagated by Orekit's **adaptive
    Dormand-Prince 8(5,3)** integrator under a real force model: Newtonian central gravity plus the
    central body's **J2 oblateness** term, which pure two-body motion cannot represent. Set ``j2``
    to ``0.0`` to recover pure two-body dynamics (how the tier is regressed against the closed-form
    Keplerian oracle). Selecting this tier requires the ``astro-mine-sim[orekit]`` extra; it needs
    no ``orekit-data`` download.
    """

    kind: Literal["orekit_orbital"] = "orekit_orbital"
    mu_m3_s2: float = Field(default=MOON_MU_M3_S2, gt=0.0)
    #: The central body's J2 zonal coefficient (lunar default). ``0.0`` ⇒ pure two-body.
    j2: float = Field(default=LUNAR_J2, ge=0.0)
    #: The reference radius (m) the J2 term is normalized against (the body's equatorial radius).
    reference_radius_m: float = Field(default=MOON_RADIUS_M, gt=0.0)
    #: Adaptive-integrator bounds + per-step tolerances (the error control the RK4 tier lacks).
    min_step_s: float = Field(default=1.0e-3, gt=0.0)
    max_step_s: float = Field(default=300.0, gt=0.0)
    position_tolerance_m: float = Field(default=1.0e-6, gt=0.0)
    velocity_tolerance_mps: float = Field(default=1.0e-9, gt=0.0)
    station_keeping_power_w: float = Field(default=0.0, ge=0.0)


class MujocoMobilityDynamics(_Spec):
    """MuJoCo articulated wheel-soil-contact mobility (RM-P0-SIM-03) — the ARTICULATED tier.

    Where :class:`MobilityDynamics` is a closed-form velocity ramp under an ``a = F/m`` cap, this
    block describes a **physical machine**: a chassis and four torque-driven wheels in frictional
    contact with a compliant regolith plane. Traction is limited by the friction cone (``mu =
    tan(friction_angle_deg)``), so the rover can slip, sink, and pitch.

    The **chassis/wheel** fields are the asset's (Fleet SADF bodies + contact elements + actuators)
    and the **terrain** fields the world's (Worlds ``RegolithParams`` + gravity) — both are sourced
    from the resolved content a scenario pins via
    :func:`~astro_mine.sim.runtime.content.mujoco_dynamics_from_content`, rather than being
    hand-authored. The defaults below are the reduced-order lunar-anchor values for a scenario that
    pins no content. Selecting this tier requires the ``astro-mine-sim[mujoco]`` extra.
    """

    kind: Literal["mujoco_mobility"] = "mujoco_mobility"
    # Asset (Fleet SADF).
    mass_kg: float = Field(gt=0.0)
    max_speed_mps: float = Field(gt=0.0)
    body_half_extents_m: Vec3Spec = (0.5, 0.35, 0.15)
    wheel_radius_m: float = Field(default=DEFAULT_WHEEL_RADIUS_M, gt=0.0)
    wheel_width_m: float = Field(default=0.10, gt=0.0)
    wheel_mass_kg: float = Field(default=5.0, gt=0.0)
    wheel_torque_nm: float = Field(default=40.0, gt=0.0)
    idle_power_w: float = Field(default=0.0, ge=0.0)
    drive_power_w_per_mps: float = Field(default=0.0, ge=0.0)
    # World (Worlds gravity + RegolithParams).
    gravity_m_s2: float = Field(default=LUNAR_GRAVITY_M_S2, gt=0.0)
    friction_angle_deg: float = Field(default=31.0, gt=0.0, lt=90.0)
    bearing_capacity_pa: float = Field(default=4.0e4, gt=0.0)
    # Solver numerics: contact is stiff, so it sub-steps well below the scenario's macro dt.
    timestep_s: float = Field(default=0.002, gt=0.0)


class MjxContactDynamics(_Spec):
    """Brax/MJX GPU-vectorized **contact** mobility (RM-P1-SIM-04) — the batched contact tier.

    The same articulated wheel-soil machine as :class:`MujocoMobilityDynamics`, compiled through
    **MuJoCo MJX** and ``jax.vmap``-batched across thousands of parallel envs on a GPU — real
    contact physics at training scale, where :class:`BraxContactDynamics` is the *cheaper*
    reduced-order JAX kernel (algebraically the kinematic mobility model) kept for very large
    sweeps.

    ``batch_size`` is the default number of parallel envs the vectorized rollout builds;
    ``init_speed_jitter_mps`` (0 by default) is the std-dev of a seeded per-env initial-velocity
    perturbation for domain randomization. Selecting this tier requires ``astro-mine-sim[brax]``
    (jax/brax/mujoco); the Ray fan-out also needs ``[ray]``.
    """

    kind: Literal["mjx_contact"] = "mjx_contact"
    # Asset (Fleet SADF) — mirrors MujocoMobilityDynamics, so the two tiers step the same machine.
    mass_kg: float = Field(gt=0.0)
    max_speed_mps: float = Field(gt=0.0)
    body_half_extents_m: Vec3Spec = (0.5, 0.35, 0.15)
    wheel_radius_m: float = Field(default=DEFAULT_WHEEL_RADIUS_M, gt=0.0)
    wheel_width_m: float = Field(default=0.10, gt=0.0)
    wheel_mass_kg: float = Field(default=5.0, gt=0.0)
    wheel_torque_nm: float = Field(default=40.0, gt=0.0)
    idle_power_w: float = Field(default=0.0, ge=0.0)
    drive_power_w_per_mps: float = Field(default=0.0, ge=0.0)
    # World (Worlds gravity + RegolithParams).
    gravity_m_s2: float = Field(default=LUNAR_GRAVITY_M_S2, gt=0.0)
    friction_angle_deg: float = Field(default=31.0, gt=0.0, lt=90.0)
    bearing_capacity_pa: float = Field(default=4.0e4, gt=0.0)
    # MJX solver numerics + the batched-rollout shape.
    timestep_s: float = Field(default=0.004, gt=0.0)
    #: Default parallel-env count for the GPU-batched contact rollout (sim.md §8).
    batch_size: int = Field(default=64, gt=0)
    #: Seeded domain-randomization std-dev for initial velocity (m/s); 0 ⇒ deterministic.
    init_speed_jitter_mps: float = Field(default=0.0, ge=0.0)


#: The per-agent dynamics block — a discriminated union over the regime engines, tagged by ``kind``
#: (RM-P0-SIM-03). New engines extend it append-only.
Dynamics = Annotated[
    KinematicDynamics
    | OrbitalDynamics
    | OrekitOrbitalDynamics
    | MobilityDynamics
    | MujocoMobilityDynamics
    | ManipulationDynamics
    | GranularDynamics
    | DemGranularDynamics
    | BraxContactDynamics
    | MjxContactDynamics,
    Field(discriminator="kind"),
]


class IsruSpec(_Spec):
    """The reduced-order ISRU extraction/storage parameters of an agent (RM-P1-SIM-02).

    When an agent declares ``isru``, the Simulator evolves its stored-water mass + extraction energy
    alongside power/thermal: while the agent is in one of ``extraction_modes`` it produces
    ``extraction_rate_kg_s`` (scaled by the local water abundance, sampled from the injected
    resource field, or ``nominal_abundance`` when none) at ``specific_energy_j_per_kg`` per kg,
    bounded by an optional tank ``capacity_kg``. Sim-owned schema (no Core/SADF change)."""

    extraction_rate_kg_s: float = Field(default=1.0e-3, ge=0.0)
    specific_energy_j_per_kg: float = Field(default=2.0e6, ge=0.0)
    capacity_kg: float | None = Field(default=None, ge=0.0)
    extraction_modes: tuple[str, ...] = _DEFAULT_EXTRACTION_MODES
    nominal_abundance: float = Field(default=1.0, ge=0.0, le=1.0)


class AgentSpec(_Spec):
    """One agent's initial declaration.

    ``initial_position_m`` / ``velocity_mps`` resolve in the agent's
    :class:`~astro_mine.core.units.ReferenceFrame`: ``frame`` when set, else the scenario default. A
    per-agent frame lets heterogeneous assets coexist — e.g. an inertial-frame relay orbiter
    alongside body-fixed surface rovers (RM-P0-SIM-03/04). ``battery_floor_j`` is the
    state-of-charge floor below which the agent terminates; the linear placeholder draw is a
    stand-in until real power/thermal evolution (RM-P0-SIM-07). ``dynamics`` selects and
    parameterizes the agent's regime engine (RM-P0-SIM-03), defaulting to the
    constant-velocity reference engine so a bare agent stays valid."""

    agent_id: str
    initial_position_m: Vec3Spec = (0.0, 0.0, 0.0)
    velocity_mps: Vec3Spec = (0.0, 0.0, 0.0)
    battery_soc_j: float = 0.0
    battery_floor_j: float = 0.0
    mode: str = "idle"
    frame: ReferenceFrame | None = None
    dynamics: Dynamics = Field(default_factory=KinematicDynamics)
    #: The agent's available multi-fidelity ladder (Fleet's profiles, RM-P0-FLEET-05), consumed by
    #: the multi-fidelity scheduler (RM-P0-SIM-05). Empty defaults to the single tier the agent's
    #: regime engine declares; the Fleet-sourced plumbing lands later, like the dynamics parameters
    #: above.
    fidelity_profiles: tuple[FidelityProfile, ...] = ()
    #: The agent's SADF sensor suite (RM-P0-FLEET-04), rendered into per-tick observations by the
    #: sensor models (RM-P0-SIM-06). Empty means the agent reports no sensor readings.
    sensors: tuple[Sensor, ...] = ()
    #: The agent's SADF power / thermal budgets (RM-P0-FLEET-04). When ``power`` is set the
    #: Simulator evolves the asset's battery SoC and temperature via power/thermal evolution
    #: (RM-P0-SIM-07); when it is ``None`` the agent keeps its engine's placeholder battery draw.
    #: ``initial_temperature_k`` seeds the thermal state (defaults to the operating-range midpoint).
    power: PowerBudget | None = None
    thermal: ThermalBudget | None = None
    initial_temperature_k: float | None = None
    #: The agent's reduced-order ISRU extraction/storage parameters (RM-P1-SIM-02). When set, the
    #: Simulator evolves its stored-water mass + extraction energy, reported by a
    #: ``resource_storage`` sensor; ``None`` means the agent does no ISRU.
    isru: IsruSpec | None = None
    #: How much regolith this agent can carry (kg), from its SADF payload slots that accept it.
    #: A hauler declares one; everything else carries nothing. ``None`` means "not a carrier" and
    #: is distinct from a declared capacity of zero (#64).
    cargo_capacity_kg: float | None = Field(default=None, ge=0.0)


class Scenario(_Spec):
    """A loadable, reproducible episode declaration.

    ``seed`` + ``start_epoch`` fix determinism; ``dt_s`` is the clock base rate and
    ``horizon_steps`` the truncation horizon. ``frame`` defaults to the Moon body-fixed frame and
    ``start_epoch`` to the J2000 TDB epoch — the lunar anchor scenario's defaults — but both are
    explicit and overridable (no implicit Earth/WGS84 anywhere).
    """

    name: str
    agents: tuple[AgentSpec, ...]
    seed: int = 0
    dt_s: float = Field(default=1.0, gt=0.0)
    horizon_steps: int = Field(default=8, gt=0)
    start_epoch: Epoch = J2000_EPOCH
    frame: ReferenceFrame = MOON_BODY_FIXED
    #: The rule-based multi-fidelity policy (RM-P0-SIM-05) the scheduler resolves over the agents;
    #: the default selects the coarsest available tier per agent.
    fidelity: FidelityPolicy = Field(default_factory=FidelityPolicy)

    @field_validator("agents")
    @classmethod
    def _non_empty_unique(cls, agents: tuple[AgentSpec, ...]) -> tuple[AgentSpec, ...]:
        if not agents:
            raise ValueError("a scenario needs at least one agent")
        ids = [a.agent_id for a in agents]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate agent ids: {duplicates}")
        return agents

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Scenario:
        """Validate a scenario from an in-memory mapping (e.g. a parsed document)."""
        return cls.model_validate(dict(data))

    @classmethod
    def from_json(cls, text: str) -> Scenario:
        """Validate a scenario from a JSON document string."""
        return cls.from_mapping(json.loads(text))


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a :class:`Scenario` from a JSON file."""
    return Scenario.from_json(Path(path).read_text(encoding="utf-8"))
