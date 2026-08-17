# SPDX-License-Identifier: Apache-2.0
"""Terramechanics validation against analytic cases (RM-P0-SIM-10; sim.md §2.9, §10).

The Phase-0 surface engines are reduced-order, so their oracle is the **closed form of the very
model they implement** — no lab data needed for this tier (terrestrial-analog rover-swarm validation
is Phase 2). Each gate carries an explicit error budget:

- **Mobility (drawbar-pull):** a rover from rest under a velocity command ramps at the
  terramechanics-limited acceleration ``a = max_traction_n / mass_kg`` and saturates at
  ``max_speed_mps`` — i.e. its speed profile is exactly ``min(a·t, v_max)``. The engine integrates
  velocity to that profile (position is forward-Euler, an ``O(dt)`` quadrature on top), so the speed
  match is the tight, model-faithful oracle. :func:`validate_mobility_engine`.
- **Granular (excavation):** an excavator removes volume at ``max_dig_rate_m3_s`` until a target,
  accruing mass ``rho·V`` and work ``specific_energy_j_per_m3·V`` — a bit-exact mass/energy balance.
  :func:`validate_granular_engine`.
- **DEM granular (high-fidelity):** the ground-truth soft-sphere engine (RM-P1-SIM-06) is
  ``TOLERANCE``, so it is gated by a *physical* analytic case rather than its own closed form: a
  settled bed's total floor reaction must balance its weight (Newtonian static equilibrium),
  ``Σ floor_normal ≈ N·m·g``, within a loose budget. :func:`validate_dem_granular_engine`.
"""

from __future__ import annotations

import math

from astro_mine.core.messages.enums import (
    ActionKind,
    ControlMode,
    ExcavationPattern,
    ExcavationTool,
    TaskKind,
)
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    ExcavateTask,
    TaskDirective,
    Vec3,
    Volume,
)
from astro_mine.sim.engines.granular import GranularEngine, granular_engine_factory
from astro_mine.sim.engines.mobility import MobilityEngine, mobility_engine_factory
from astro_mine.sim.runtime.rng import RngStreams
from astro_mine.sim.runtime.scenario import (
    AgentSpec,
    DemGranularDynamics,
    GranularDynamics,
    MobilityDynamics,
    Scenario,
)
from astro_mine.sim.validation._report import OracleReport, validate_against_oracle

__all__ = [
    "drawbar_pull_speed",
    "validate_dem_granular_engine",
    "validate_granular_engine",
    "validate_mobility_engine",
]


def drawbar_pull_speed(
    t_s: float, max_traction_n: float, mass_kg: float, max_speed_mps: float
) -> float:
    """The analytic rover speed at time ``t_s``: ``min(a·t, v_max)`` with ``a = traction/mass``.

    The closed form of the drawbar-pull-limited, top-speed-capped mobility model — a rover from rest
    accelerates at the traction limit until it saturates at the speed cap.
    """
    return min((max_traction_n / mass_kg) * t_s, max_speed_mps)


def _mobility_engine(mass_kg: float, max_traction_n: float, max_speed_mps: float) -> MobilityEngine:
    scenario = Scenario(
        name="mobility-oracle",
        agents=(
            AgentSpec(
                agent_id="rover",
                battery_soc_j=1.0e6,
                dynamics=MobilityDynamics(
                    mass_kg=mass_kg, max_speed_mps=max_speed_mps, max_traction_n=max_traction_n
                ),
            ),
        ),
        horizon_steps=1,
    )
    return mobility_engine_factory(scenario, RngStreams(0))


def validate_mobility_engine(
    *,
    mass_kg: float,
    max_traction_n: float,
    max_speed_mps: float,
    dt_s: float,
    steps: int,
    budget: float = 1e-9,
) -> OracleReport:
    """Validate the mobility engine's speed profile against the analytic drawbar-pull case.

    Commands a rover (from rest) a velocity above the speed cap and checks that its speed at each
    step matches ``min(a·t, v_max)`` — the traction-limited acceleration and the top-speed cap, in
    one profile — within ``budget`` (relative). Position is a forward-Euler quadrature of this speed
    and is validated separately by the engine's own tests.
    """
    engine = _mobility_engine(mass_kg, max_traction_n, max_speed_mps)
    command = ActionBatch(
        actions=[
            Action(
                agent_id="rover",
                kind=ActionKind.ACTUATOR,
                actuator=ActuatorCommand(
                    target="base",
                    control_mode=ControlMode.VELOCITY,
                    setpoint=[2.0 * max_speed_mps, 0.0, 0.0],  # above the cap → full ramp then hold
                ),
            )
        ]
    )
    engine.apply_actions(command)  # the command persists across advances
    actual: list[tuple[float, ...]] = []
    reference: list[tuple[float, ...]] = []
    for k in range(1, steps + 1):
        engine.advance(dt_s)
        sample = engine.export_coupling_state().by_agent["rover"]
        velocity = sample.linear_velocity_mps
        assert velocity is not None
        actual.append((velocity.x, velocity.y, velocity.z))
        reference.append(
            (drawbar_pull_speed(k * dt_s, max_traction_n, mass_kg, max_speed_mps), 0.0, 0.0)
        )
    return validate_against_oracle(
        actual, reference, budget=budget, name="mobility-drawbar-pull", relative=True
    )


def _granular_engine(
    regolith_density_kg_m3: float, specific_energy_j_per_m3: float, max_dig_rate_m3_s: float
) -> GranularEngine:
    scenario = Scenario(
        name="granular-oracle",
        agents=(
            AgentSpec(
                agent_id="digger",
                battery_soc_j=1.0e9,
                dynamics=GranularDynamics(
                    regolith_density_kg_m3=regolith_density_kg_m3,
                    specific_energy_j_per_m3=specific_energy_j_per_m3,
                    max_dig_rate_m3_s=max_dig_rate_m3_s,
                ),
            ),
        ),
        horizon_steps=1,
    )
    return granular_engine_factory(scenario, RngStreams(0))


def validate_granular_engine(
    *,
    regolith_density_kg_m3: float,
    specific_energy_j_per_m3: float,
    max_dig_rate_m3_s: float,
    target_volume_m3: float,
    dt_s: float,
    steps: int,
    budget: float = 1e-9,
) -> OracleReport:
    """Validate the granular engine's excavated-volume profile against the analytic rate model.

    Commands an excavate to ``target_volume_m3`` and checks the cumulative excavated volume at each
    step matches ``min(k·rate·dt, target)`` within ``budget``. The bit-exact mass (``rho·V``) and
    work (``specific_energy·V``) balances follow from the volume and are asserted by engine tests.
    """
    engine = _granular_engine(regolith_density_kg_m3, specific_energy_j_per_m3, max_dig_rate_m3_s)
    command = ActionBatch(
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
                        target_volume_m3=target_volume_m3,
                    ),
                ),
            )
        ]
    )
    engine.apply_actions(command)
    actual: list[tuple[float, ...]] = []
    reference: list[tuple[float, ...]] = []
    for k in range(1, steps + 1):
        engine.advance(dt_s)
        actual.append((engine.excavated_volume_m3("digger"),))
        reference.append((min(k * max_dig_rate_m3_s * dt_s, target_volume_m3),))
    return validate_against_oracle(
        actual, reference, budget=budget, name="granular-excavation", relative=True
    )


def validate_dem_granular_engine(
    *,
    n_particles: int = 90,
    particle_radius_m: float = 0.02,
    regolith_density_kg_m3: float = 1500.0,
    contact_stiffness_n_m: float = 5.0e4,
    friction_coeff: float = 0.6,
    restitution: float = 0.3,
    gravity_m_s2: float = 1.62,
    bed_width_m: float = 0.6,
    settle_substeps: int = 1500,
    seed: int = 0,
    budget: float = 0.05,
) -> OracleReport:
    """Validate the DEM engine against static equilibrium: settled floor reaction ≈ bed weight.

    The ground-truth soft-sphere engine has no closed form to test against; the physical
    analytic case is Newton's static balance — a bed at rest transmits its full weight
    ``N*m*g`` (``m = rho*pi*r^2`` per unit depth) to the floor as normal reaction. Building the
    engine settles the bed (the factory runs ``settle_substeps``); we compare the total floor
    reaction to the weight within a loose ``budget`` (relative), the tolerance appropriate to a
    ``TOLERANCE`` engine (residual settling motion + finite contact stiffness). Requires the
    ``[dem]`` extra (numpy), imported lazily here.
    """
    from astro_mine.sim.engines.dem import dem_granular_engine_factory

    scenario = Scenario(
        name="dem-granular-oracle",
        agents=(
            AgentSpec(
                agent_id="digger",
                battery_soc_j=1.0e9,
                dynamics=DemGranularDynamics(
                    n_particles=n_particles,
                    particle_radius_m=particle_radius_m,
                    regolith_density_kg_m3=regolith_density_kg_m3,
                    contact_stiffness_n_m=contact_stiffness_n_m,
                    friction_coeff=friction_coeff,
                    restitution=restitution,
                    gravity_m_s2=gravity_m_s2,
                    bed_width_m=bed_width_m,
                    settle_substeps=settle_substeps,
                ),
            ),
        ),
        horizon_steps=1,
    )
    engine = dem_granular_engine_factory(scenario, RngStreams(seed))
    mass_kg = regolith_density_kg_m3 * math.pi * particle_radius_m**2
    weight_n = n_particles * mass_kg * gravity_m_s2
    floor_reaction_n = engine.floor_reaction_n("digger")  # type: ignore[attr-defined]
    return validate_against_oracle(
        [(floor_reaction_n,)], [(weight_n,)], budget=budget, name="dem-granular-force-balance"
    )
