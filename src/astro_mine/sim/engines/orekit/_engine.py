"""The Orekit orbital :class:`RegimeEngine` — the higher-fidelity orbital tier (RM-P0-SIM-03).

The flight-grade orbital backend RM-P0-SIM-03 names, behind the *same* ``RegimeEngine`` waist as the
reduced-order RK4 two-body engine (sim.md §4, §11: "Basilisk + Orekit ... SPICE for frames/time,
GMAT/STK as oracles only"). Selecting it is **configuration** — a scenario's ``dynamics.kind`` — not
a Sim code change, and no Orekit type ever leaks past the adapter: the engine's public surface is
Core messages (sim.md §2.1).

What makes it the higher tier:

- **Error-controlled integration.** Orekit's adaptive **Dormand-Prince 8(5,3)** integrator with
  per-step absolute/relative tolerances, rather than a fixed-step RK4 whose truncation error is
  whatever the sub-step count happens to buy.
- **A richer force model.** Newtonian central gravity **plus the central body's J2 oblateness term**
  — a real perturbation (it drives nodal regression and apsidal precession of a lunar orbit) that a
  pure two-body propagator cannot represent at all. Set ``j2`` to 0 to recover pure two-body motion,
  which is how the tier is regressed against the closed-form Keplerian oracle.

**The JVM lives here, never at package import.** :mod:`astro_mine.sim.engines.orekit` imports this
module only inside its factory, so the base wheel — and ``builtins.py``'s manifest registration —
stay JVM-free (the ``[orekit]`` extra). The Cartesian/Keplerian propagation path needs **no
``orekit-data`` bundle**: it touches no leap-second table, no Earth-orientation history, and no
gravity-field file, so the tier runs fully offline (CX-LOCAL) with nothing to download.

Determinism is ``TOLERANCE`` (an adaptive step sequence and the JVM's reductions are not
bit-portable across builds); same-inputs runs reproduce in-process, and the tier is admitted against
the analytic oracle's explicit error budget rather than a golden hash (sim.md §11).

**No maneuvers.** A Δv/targeting capability is gated out of the open commons
(``operational_targeting``), so ``apply_actions`` honors only a mode command — exactly like the
reduced-order orbital engine. Orekit's maneuver/targeting machinery is deliberately not wired in.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.sim.engines.actuation import actions_by_agent
from astro_mine.sim.engines.adapter import CouplingState, EngineDescriptor
from astro_mine.sim.engines.orekit._descriptor import OREKIT_ORBITAL_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from collections.abc import Iterable

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.core.units import ReferenceFrame
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import OrekitOrbitalDynamics, Scenario

__all__ = ["OrekitOrbitalEngine", "build_orekit_orbital_engine", "start_jvm"]

_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)

#: The JVM is process-global and may only be started once; guard the boot so repeated engine
#: construction (a Bench sweep builds one engine per episode) is safe and race-free.
_JVM_LOCK = threading.Lock()
_JVM_STARTED = False


def start_jvm() -> None:
    """Boot the JVM the Orekit binding runs on — idempotent and thread-safe.

    ``jdk4py`` ships a JVM with the ``[orekit]`` extra, so the tier needs no system Java: if
    ``JAVA_HOME`` is unset we point it at that bundled runtime. A host that has already configured
    its own JVM (a different JDK, a tuned heap) keeps it — we never override an explicit
    ``JAVA_HOME``."""
    global _JVM_STARTED
    with _JVM_LOCK:
        if _JVM_STARTED:
            return
        if not os.environ.get("JAVA_HOME"):
            import jdk4py

            os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
        import orekit_jpype

        orekit_jpype.initVM()
        _JVM_STARTED = True


@dataclass
class _OrekitState:
    """Mutable per-orbiter state: the live Orekit propagator plus the Sim-side accounting the
    ``RegimeEngine`` contract owns (battery, mode) — which Orekit knows nothing about."""

    agent_id: str
    frame: ReferenceFrame
    propagator: Any  # org.orekit.propagation.numerical.NumericalPropagator
    epoch: Any  # org.orekit.time.AbsoluteDate — the date the propagator has reached
    #: The scenario's dynamics block, kept Sim-side so a coupling re-seed rebuilds the propagator
    #: with the identical integrator + force model (a Java object cannot carry a Python attribute).
    dynamics: OrekitOrbitalDynamics
    mu_m3_s2: float
    station_keeping_power_w: float
    battery_soc_j: float
    battery_floor_j: float
    mode: str | None

    def sample(self) -> StateSample:
        """The current state as a frame-explicit Core coupling sample (position + velocity)."""
        pv = self.propagator.getPVCoordinates(self.epoch, self.propagator.getFrame())
        p, v = pv.getPosition(), pv.getVelocity()
        return StateSample(
            agent_id=self.agent_id,
            frame=self.frame,
            pose=Transform(
                translation_m=Vec3(x=float(p.getX()), y=float(p.getY()), z=float(p.getZ())),
                rotation_quat_xyzw=_IDENTITY_QUAT,  # a point mass carries no attitude
            ),
            linear_velocity_mps=Vec3(x=float(v.getX()), y=float(v.getY()), z=float(v.getZ())),
            battery_soc_j=self.battery_soc_j,
            mode=self.mode,
        )


class OrekitOrbitalEngine:
    """The Orekit-backed orbital :class:`~astro_mine.sim.engines.RegimeEngine`.

    Owns one Orekit ``NumericalPropagator`` per orbiter and advances each by the macro step; the
    only actuation is a mode command (no maneuvers in the open commons). Battery/mode accounting is
    Sim's, exactly as in the reduced-order tier — the backend swap is invisible above the waist."""

    def __init__(self, states: dict[str, _OrekitState]) -> None:
        self._states = states
        self._elapsed_s = 0.0

    @property
    def descriptor(self) -> EngineDescriptor:
        return OREKIT_ORBITAL_ENGINE_DESCRIPTOR

    def apply_actions(self, actions: ActionBatch) -> None:
        """Honor a mode command for an owned orbiter; ignore everything else (no maneuvers)."""
        for agent_id, action in actions_by_agent(actions).items():
            state = self._states.get(agent_id)
            if state is None:
                continue
            if action.kind is ActionKind.MODE and action.mode is not None:
                state.mode = action.mode.mode

    def advance(self, dt_s: float) -> None:
        """Propagate every live orbiter forward ``dt_s`` through its Orekit propagator."""
        self._elapsed_s += dt_s
        for state in self._states.values():
            target = state.epoch.shiftedBy(float(dt_s))
            state.propagator.propagate(target)
            state.epoch = target
            drained = state.battery_soc_j - state.station_keeping_power_w * dt_s
            state.battery_soc_j = max(state.battery_floor_j, drained)

    def export_coupling_state(self) -> CouplingState:
        return CouplingState(
            sim_time_s=self._elapsed_s,
            samples=tuple(state.sample() for state in self._states.values()),
        )

    def import_coupling_state(self, state: CouplingState) -> None:
        """Overwrite live orbiters from a boundary snapshot — re-seeding the Orekit propagator with
        the incoming Cartesian state at the current date (the coupler's cross-engine handoff)."""
        incoming = state.by_agent
        for agent_id, current in self._states.items():
            sample = incoming.get(agent_id)
            if sample is None:
                continue
            t = sample.pose.translation_m
            velocity = sample.linear_velocity_mps
            v = (
                (velocity.x, velocity.y, velocity.z)
                if velocity is not None
                else _velocity_of(current)
            )
            current.propagator = _build_propagator(
                position=(t.x, t.y, t.z),
                velocity=v,
                epoch=current.epoch,
                mu_m3_s2=current.mu_m3_s2,
                dynamics=current.dynamics,
            )
            if sample.battery_soc_j is not None:
                current.battery_soc_j = sample.battery_soc_j
            if sample.mode is not None:
                current.mode = sample.mode
        self._elapsed_s = state.sim_time_s

    def retire(self, agent_ids: Iterable[str]) -> None:
        for agent_id in agent_ids:
            self._states.pop(agent_id, None)


def _velocity_of(state: _OrekitState) -> tuple[float, float, float]:
    pv = state.propagator.getPVCoordinates(state.epoch, state.propagator.getFrame())
    v = pv.getVelocity()
    return (float(v.getX()), float(v.getY()), float(v.getZ()))


def _build_propagator(
    *,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    epoch: Any,
    mu_m3_s2: float,
    dynamics: OrekitOrbitalDynamics,
) -> Any:
    """An Orekit ``NumericalPropagator`` seeded with a Cartesian state.

    Adaptive Dormand-Prince 8(5,3) over the Cartesian orbit type, with Newtonian central gravity
    plus (when ``j2`` is non-zero) the central body's J2 oblateness term. Every input is Cartesian
    and the frame is the fixed inertial EME2000, so nothing here needs an ``orekit-data`` bundle."""
    start_jvm()  # idempotent; a coupling re-seed can reach here before the factory has run
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.forces.gravity import J2OnlyPerturbation, NewtonianAttraction
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import CartesianOrbit, OrbitType
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.utils import PVCoordinates

    frame = FramesFactory.getEME2000()
    pv = PVCoordinates(
        Vector3D(float(position[0]), float(position[1]), float(position[2])),
        Vector3D(float(velocity[0]), float(velocity[1]), float(velocity[2])),
    )
    orbit = CartesianOrbit(pv, frame, epoch, float(mu_m3_s2))
    integrator = DormandPrince853Integrator(
        float(dynamics.min_step_s),
        float(dynamics.max_step_s),
        float(dynamics.position_tolerance_m),
        float(dynamics.velocity_tolerance_mps),
    )
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(SpacecraftState(orbit))
    propagator.addForceModel(NewtonianAttraction(float(mu_m3_s2)))
    if dynamics.j2:
        # The oblateness term the reduced-order two-body tier cannot represent at all.
        propagator.addForceModel(
            J2OnlyPerturbation(
                float(mu_m3_s2),
                float(dynamics.reference_radius_m),
                float(dynamics.j2),
                frame,
            )
        )
    return propagator


def build_orekit_orbital_engine(scenario: Scenario, rng: RngStreams) -> OrekitOrbitalEngine:
    """Build an :class:`OrekitOrbitalEngine` for the scenario's ``orekit_orbital`` agents.

    Agents whose ``dynamics`` is not ``orekit_orbital`` are skipped, so the engine owns only its
    regime's assets (the heterogeneous co-step is RM-P0-SIM-04). The propagator is deterministic, so
    the seeded ``rng`` is unused — same as the reduced-order orbital tier."""
    start_jvm()  # must precede any `org.*` import: the Java packages exist only once the JVM is up
    from org.orekit.time import AbsoluteDate

    # The scenario's start epoch is TDB seconds past J2000 — which is exactly the offset Orekit's
    # own J2000 epoch is shifted by, so the two clocks agree without a leap-second table.
    start = AbsoluteDate.J2000_EPOCH.shiftedBy(float(scenario.start_epoch.tdb_seconds))
    states: dict[str, _OrekitState] = {}
    for spec in scenario.agents:
        dyn = spec.dynamics
        if dyn.kind != "orekit_orbital":
            continue
        propagator = _build_propagator(
            position=spec.initial_position_m,
            velocity=spec.velocity_mps,
            epoch=start,
            mu_m3_s2=dyn.mu_m3_s2,
            dynamics=dyn,
        )
        state = _OrekitState(
            agent_id=spec.agent_id,
            frame=spec.frame or scenario.frame,
            propagator=propagator,
            epoch=start,
            dynamics=dyn,
            mu_m3_s2=dyn.mu_m3_s2,
            station_keeping_power_w=dyn.station_keeping_power_w,
            battery_soc_j=spec.battery_soc_j,
            battery_floor_j=spec.battery_floor_j,
            mode=spec.mode,
        )
        states[spec.agent_id] = state
    return OrekitOrbitalEngine(states)
