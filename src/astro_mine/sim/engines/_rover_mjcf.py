# SPDX-License-Identifier: Apache-2.0
"""The articulated wheeled-rover MJCF model — the shared contact model (RM-P0-SIM-03, RM-P1-SIM-04).

One place that describes the rover **as a physical machine** rather than as a point mass: a chassis
body on a free joint, four hinge-driven wheels with real geometry and mass, and a ground plane whose
friction and stiffness stand in for regolith. It is the model *both* contact tiers step —

- the **MuJoCo** articulated mobility tier (:mod:`astro_mine.sim.engines.mujoco`, ``[mujoco]``),
  which steps it on the CPU through ``mujoco.mj_step``; and
- the **Brax/MJX** GPU-vectorized contact tier (:mod:`astro_mine.sim.engines.brax`, ``[brax]``),
  which compiles the *same* model through ``mujoco.mjx`` and ``jax.vmap``s it across thousands of
  parallel envs —

so the two tiers cannot silently disagree about what a rover *is*. The distinction between them is
purely the execution substrate (CPU step vs. XLA-compiled batched step), which is exactly the claim
sim.md §11 makes ("MuJoCo/MJX default, Brax for differentiable/JAX-native massively parallel
rollouts").

**Wheel-soil contact.** MuJoCo's contact model is Coulomb friction over a soft (spring-damper)
contact. Regolith is not a rigid surface, so the plane's contact parameters are derived from the
terramechanics the world/scenario declares — the friction coefficient from the regolith's internal
friction angle (``mu = tan(phi)``), and the contact softness (``solref``/``solimp``) from its
bearing stiffness — rather than being hard-coded. That is a genuine (if reduced-order) wheel-soil
contact: the wheels *roll*, they can *slip*, and drawbar pull is limited by the friction cone, not
by an ``a = F/m`` cap written into a kinematic formula.

This module builds only the **XML string**; it imports neither MuJoCo nor JAX, so it is free to
import anywhere (the descriptors and the scenario schema reference its defaults).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DEFAULT_WHEEL_RADIUS_M",
    "RoverModelSpec",
    "rover_mjcf",
]

#: Default wheel radius (m) — a mid-size prospecting rover.
DEFAULT_WHEEL_RADIUS_M = 0.25


@dataclass(frozen=True, slots=True)
class RoverModelSpec:
    """The physical description a wheeled rover's contact model is built from (SI).

    The **chassis/wheel** fields are the asset's (a Fleet SADF body/contact-element declaration);
    the **terrain** fields are the world's (a Worlds ``RegolithParams`` / gravity sample). Both are
    sourced from resolved content when a scenario pins it — see
    :func:`astro_mine.sim.runtime.content.mujoco_dynamics_from_content`."""

    mass_kg: float
    #: Chassis half-extents (m) — the box the mass is distributed over.
    body_half_extents_m: tuple[float, float, float] = (0.5, 0.35, 0.15)
    wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M
    wheel_width_m: float = 0.10
    wheel_mass_kg: float = 5.0
    #: Peak torque (N·m) one wheel motor can apply — the actuator limit Guard enforces.
    wheel_torque_nm: float = 40.0
    #: The rover's top speed (m/s); it bounds the wheel-velocity setpoint the controller tracks.
    max_speed_mps: float = 1.0
    #: Surface gravity magnitude (m/s²) — lunar by default (Worlds' gravity model when pinned).
    gravity_m_s2: float = 1.62
    #: Regolith internal friction angle (deg). The wheel-soil Coulomb friction coefficient is
    #: ``tan(phi)``, so this — not a hand-written acceleration cap — is what limits drawbar pull.
    friction_angle_deg: float = 31.0
    #: Regolith bearing stiffness (Pa). Drives the contact's ``solref`` stiffness: soft ground
    #: yields under the wheel (a shallow sinkage) instead of behaving like a rigid floor.
    bearing_capacity_pa: float = 4.0e4
    #: Integrator step (s) the contact solver runs at. Contact is stiff, so it sub-steps well below
    #: the scenario's macro dt (sim.md §4: "granular sub-millisecond; mobility milliseconds").
    timestep_s: float = 0.002

    @property
    def friction_coeff(self) -> float:
        """The wheel-soil Coulomb friction coefficient, ``tan(phi)`` — the drawbar-pull limit."""
        return math.tan(math.radians(self.friction_angle_deg))

    @property
    def contact_time_constant_s(self) -> float:
        """The contact's ``solref`` time constant, from the ground's bearing stiffness.

        A stiffer soil resolves a contact faster (a shorter time constant); a soft one lets the
        wheel settle into it. Clamped to at least two solver steps, which is MuJoCo's stability
        floor for a
        soft contact."""
        # A reduced-order map: stiffer ground (higher bearing capacity) -> faster contact response.
        stiffness_ref = 4.0e4
        scale = math.sqrt(stiffness_ref / max(self.bearing_capacity_pa, 1.0))
        return max(0.02 * scale, 2.0 * self.timestep_s)


def rover_mjcf(spec: RoverModelSpec) -> str:
    """The MJCF (MuJoCo XML) description of one articulated wheeled rover on regolith.

    A free-jointed chassis box with four hinge wheels, each driven by a velocity actuator (the wheel
    motor), rolling on a friction plane whose Coulomb coefficient and contact softness come from the
    declared terramechanics. Gravity is the body's, not Earth's."""
    hx, hy, hz = spec.body_half_extents_m
    r = spec.wheel_radius_m
    friction = spec.friction_coeff
    solref = spec.contact_time_constant_s
    # Wheel anchors at the four corners of the chassis, dropped to the axle line.
    anchors = {
        "fl": (+hx * 0.8, +hy, 0.0),
        "fr": (+hx * 0.8, -hy, 0.0),
        "rl": (-hx * 0.8, +hy, 0.0),
        "rr": (-hx * 0.8, -hy, 0.0),
    }
    wheels = "\n".join(
        f"""      <body name="wheel_{name}" pos="{x} {y} {z}">
        <joint name="drive_{name}" type="hinge" axis="0 1 0" damping="0.5"/> <geom
        name="tyre_{name}" type="cylinder" size="{r} {spec.wheel_width_m / 2}"
              quat="0.7071068 0.7071068 0 0" mass="{spec.wheel_mass_kg}" friction="{friction} 0.01
              0.001" solref="{solref} 1" solimp="0.9 0.95 0.001" rgba="0.3 0.3 0.35 1"/>
      </body>"""
        for name, (x, y, z) in anchors.items()
    )
    actuators = "\n".join(
        f"""    <velocity name="motor_{name}" joint="drive_{name}"
              kv="{spec.wheel_torque_nm}" ctrlrange="{-spec.max_speed_mps / r} {
            spec.max_speed_mps / r
        }"
              forcerange="{-spec.wheel_torque_nm} {spec.wheel_torque_nm}"/>"""
        for name in anchors
    )
    # The chassis starts one wheel-radius up so the wheels rest on the plane at t=0.
    return f"""<mujoco model="astro-mine-rover">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{spec.timestep_s}" gravity="0 0 {-spec.gravity_m_s2}"
          integrator="implicitfast" cone="elliptic"/>
  <default>
    <geom condim="4"/>
  </default>
  <worldbody>
    <geom name="regolith" type="plane" size="0 0 1" pos="0 0 0"
          friction="{friction} 0.01 0.001" solref="{solref} 1" solimp="0.9 0.95 0.001"
          rgba="0.45 0.42 0.40 1"/>
    <body name="chassis" pos="0 0 {r}">
      <freejoint name="chassis_free"/>
      <geom name="hull" type="box" size="{hx} {hy} {hz}" mass="{spec.mass_kg}"
            friction="{friction} 0.01 0.001" rgba="0.75 0.72 0.68 1"/>
{wheels}
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""
