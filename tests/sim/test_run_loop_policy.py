"""The Core-Policy-driven run loop — Sim records a Bench-scorable MCAP (RM-P0-SIM-11).

Sim's Environment accepts an **injected Core Policy** and records an MCAP that Bench scores across
the artifact boundary; ``astro_mine.sim`` imports only Core (conventions.md §1.1). The policy and
the scenario below are **test fixtures** — a policy is autonomy's (Mind/Learn), the anchor scenario
is Bench's — exercising the run-loop injection; Sim ships neither. Covers: the policy is driven
(open-loop differs), determinism, policy-sensitivity, and the recorded MCAP carrying the discovery
(neutron), comms (``earth_contact``), and survival (battery/temperature) channels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astro_mine.core.messages.enums import ActionKind, ControlMode, NodeRole
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    ContactInterval,
    ContactNode,
    ContactPlan,
    Observation,
)
from astro_mine.core.policy import DecisionContext, Policy, check_policy
from astro_mine.core.resource import ResourceField
from astro_mine.core.sadf.enums import PowerSourceKind, SensorKind
from astro_mine.core.sadf.model import (
    ModeLoad,
    ObservationModel,
    PowerBudget,
    PowerSource,
    PowerStorage,
    Range,
    ResourceTarget,
    Sensor,
    ThermalBudget,
)
from astro_mine.core.units import INERTIAL_J2000, MOON_BODY_FIXED
from astro_mine.sim.comms import ConnectivitySource, ReferenceConnectivitySampler
from astro_mine.sim.coupling import coupled_engine_factory
from astro_mine.sim.recording import read_recording, record_episode
from astro_mine.sim.runtime import (
    AgentSpec,
    MobilityDynamics,
    OrbitalDynamics,
    Scenario,
    run_episode,
)
from astro_mine.sim.sensors import ReferenceResourceField

_SPECIES = "water_equivalent_hydrogen"
_TARGET_M = (30.0, 0.0, 0.0)
_DT_S = 60.0
_HORIZON = 16


@dataclass(frozen=True)
class _SurveyDriver:
    """A test Core Policy: drive surface (body-fixed) agents toward the deposit, skip orbital ones.

    A fixture, not shipped by Sim — a policy is autonomy's; here it just exercises the injected
    decision loop. Deterministic: same observations + context ⇒ the same batch."""

    target_m: tuple[float, float, float] = _TARGET_M
    speed_mps: float = 0.5

    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        actions: list[Action] = []
        for agent_id in sorted(observations):
            state = observations[agent_id].self_state
            if state.frame != MOON_BODY_FIXED:
                continue  # orbital assets (the relay) propagate under their own dynamics
            position = state.pose.translation_m
            dx = self.target_m[0] - position.x
            dy = self.target_m[1] - position.y
            dz = self.target_m[2] - position.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance > 1e-9:
                scale = self.speed_mps / distance
                setpoint = [dx * scale, dy * scale, dz * scale]
            else:
                setpoint = [0.0, 0.0, 0.0]
            actions.append(
                Action(
                    agent_id=agent_id,
                    kind=ActionKind.ACTUATOR,
                    actuator=ActuatorCommand(
                        target="base", control_mode=ControlMode.VELOCITY, setpoint=setpoint
                    ),
                )
            )
        return ActionBatch(actions=actions)


def _scenario(seed: int = 1001) -> Scenario:
    """A prospecting rover (neutron + power/thermal) and a relay orbiter — a test fixture."""
    rover = AgentSpec(
        agent_id="rover",
        initial_position_m=(0.0, 0.0, 0.0),
        battery_soc_j=8.0e5,
        battery_floor_j=1.0e5,
        mode="prospect",
        dynamics=MobilityDynamics(
            mass_kg=200.0,
            max_speed_mps=0.5,
            max_traction_n=400.0,
            idle_power_w=10.0,
            drive_power_w_per_mps=20.0,
        ),
        sensors=(
            Sensor(
                name="neutron",
                kind=SensorKind.NEUTRON_SPECTROMETER,
                frame="body",
                observation_model=ObservationModel(noise_sigma=0.003),
                resource=ResourceTarget(species=_SPECIES, si_unit="mass_fraction"),
            ),
        ),
        power=PowerBudget(
            sources=[PowerSource(name="solar", kind=PowerSourceKind.SOLAR, nominal_power_w=200.0)],
            storage=[
                PowerStorage(
                    name="battery", capacity_j=1.0e6, max_charge_w=150.0, max_discharge_w=300.0
                )
            ],
            floor_w=25.0,
            loads_by_mode=[
                ModeLoad(mode="idle", power_w=30.0),
                ModeLoad(mode="prospect", power_w=90.0),
            ],
        ),
        thermal=ThermalBudget(
            operating_range_k=Range(min=120.0, max=330.0),
            survival_range_k=Range(min=95.0, max=360.0),
            dissipation_w=60.0,
            radiator_area_m2=0.4,
            heater_power_w=30.0,
            surface_coupling=True,
        ),
        initial_temperature_k=250.0,
    )
    relay = AgentSpec(
        agent_id="relay",
        initial_position_m=(1_837_400.0, 0.0, 0.0),
        velocity_mps=(0.0, 1633.4, 0.0),
        battery_soc_j=5.0e5,
        frame=INERTIAL_J2000,
        mode="relay",
        dynamics=OrbitalDynamics(),
    )
    return Scenario(
        name="lunar-polar-ice-prospecting-v1",
        agents=(rover, relay),
        seed=seed,
        dt_s=_DT_S,
        horizon_steps=_HORIZON,
        frame=MOON_BODY_FIXED,
    )


def _field() -> ResourceField:
    return ReferenceResourceField(
        species=_SPECIES, unit="mass_fraction", center_m=_TARGET_M, peak=0.12, length_scale_m=20.0
    )


def _connectivity() -> ConnectivitySource:
    span = _HORIZON * _DT_S  # J2000 starts at tdb = 0, so the span is [0, N*dt)
    half = span / 2.0
    plan = ContactPlan(
        nodes=[
            ContactNode(id="rover", role=NodeRole.SPACE),
            ContactNode(id="relay", role=NodeRole.SPACE),
            ContactNode(id="dsn", role=NodeRole.GROUND),
        ],
        intervals=[
            ContactInterval(node_a="rover", node_b="dsn", start_tdb_s=0.0, end_tdb_s=half),
            ContactInterval(node_a="rover", node_b="relay", start_tdb_s=half, end_tdb_s=span),
        ],
    )
    return ReferenceConnectivitySampler(plan)


def _record(path: Path, *, policy: Policy | None, seed: int = 1001) -> Any:
    return record_episode(
        _scenario(seed),
        path,
        seed=seed,
        resource_field=_field(),
        connectivity=_connectivity(),
        policy=policy,
        engine_factory=coupled_engine_factory(),
    )


def _rover_steps(path: Path) -> list[dict[str, Any]]:
    recording = read_recording(path)
    return [
        frame["observations"]["rover"]
        for frame in recording.frames
        if frame.get("kind") == "step" and "rover" in frame.get("observations", {})
    ]


# --- the injected policy is a Core contract -----------------------------------------------------


def test_driver_is_a_core_policy_that_drives_surface_agents() -> None:
    driver = _SurveyDriver()
    assert isinstance(driver, Policy)
    observations = {aid: obs for aid, obs in _record_reset_observations().items()}
    batch = check_policy(driver, observations, DecisionContext())  # validates the ActionBatch
    commanded = {action.agent_id for action in batch.actions}
    assert commanded == {"rover"}  # the orbital relay is skipped


def _record_reset_observations() -> Mapping[str, Observation]:
    from astro_mine.sim.runtime import Simulator

    sim = Simulator(_scenario(), engine_factory=coupled_engine_factory())
    return sim.reset().observations


def test_driver_holds_still_at_the_target() -> None:
    from astro_mine.sim.runtime import Simulator

    sim = Simulator(_scenario(), engine_factory=coupled_engine_factory())
    observations = sim.reset().observations
    at_target = _SurveyDriver(target_m=(0.0, 0.0, 0.0))  # the rover starts at the origin
    batch = at_target.decide(observations, DecisionContext())
    rover_action = next(a for a in batch.actions if a.agent_id == "rover")
    assert rover_action.actuator is not None
    assert rover_action.actuator.setpoint == [0.0, 0.0, 0.0]


# --- the run-loop injection ---------------------------------------------------------------------


def test_policy_driven_run_differs_from_open_loop(tmp_path: Path) -> None:
    open_loop = _record(tmp_path / "open.mcap", policy=None)
    driven = _record(tmp_path / "driven.mcap", policy=_SurveyDriver())
    assert open_loop.content_hash != driven.content_hash  # the injected loop changed the trace


def test_policy_driven_run_is_deterministic(tmp_path: Path) -> None:
    first = _record(tmp_path / "a.mcap", policy=_SurveyDriver())
    second = _record(tmp_path / "b.mcap", policy=_SurveyDriver())
    assert first.content_hash == second.content_hash


def test_run_is_policy_sensitive(tmp_path: Path) -> None:
    toward = _record(tmp_path / "t.mcap", policy=_SurveyDriver(target_m=(30.0, 0.0, 0.0)))
    away = _record(tmp_path / "w.mcap", policy=_SurveyDriver(target_m=(-30.0, 0.0, 0.0)))
    assert toward.content_hash != away.content_hash


def test_run_episode_accepts_a_policy() -> None:
    # the batch path (no recorder) also drives the injected policy
    trace = run_episode(
        _scenario(),
        seed=1001,
        resource_field=_field(),
        connectivity=_connectivity(),
        policy=_SurveyDriver(),
        engine_factory=coupled_engine_factory(),
    )
    assert trace.content_hash and len(trace.frames) == _HORIZON + 1


# --- the recorded MCAP is Bench-scorable --------------------------------------------------------


def test_recorded_mcap_carries_the_metric_channels(tmp_path: Path) -> None:
    _record(tmp_path / "run.mcap", policy=_SurveyDriver())
    steps = _rover_steps(tmp_path / "run.mcap")
    # discovery: the neutron spectrometer reports resource readings
    assert any(
        reading["sensor"] == "neutron" and reading["values"]
        for observation in steps
        for reading in observation.get("sensors", [])
    )
    # comms: earth_contact toggles (direct DSN window, then PSR denial)
    earth = [o["comms"]["earth_contact"] for o in steps if o.get("comms") is not None]
    assert earth and 0 < sum(earth) < len(earth)
    # survival: battery + temperature evolve
    assert all(o["self_state"]["battery_soc_j"] is not None for o in steps)
    assert all(o["self_state"]["temperature_k"] is not None for o in steps)
    # the injected policy drove the rover toward the deposit
    assert steps[-1]["self_state"]["pose"]["translation_m"]["x"] > 0.0
