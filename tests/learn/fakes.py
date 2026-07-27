"""In-repo conformant fake Core world + heterogeneous SADF assets (RM-P1-LEARN-01 tests).

Learn's tests stay *waist-pure*: they exercise the SwarmEnv adapter against a fake that
implements the Core :class:`~astro_mine.core.env.protocol.Environment` protocol directly,
with **no** ``astro_mine.sim`` import. The fake is deterministic (a seeded, pure function
of tick), heterogeneous (three agents with different ``CapabilityTag`` / ``Sensor`` sets),
and exercises the surfaces the adapter must handle: partial observability (a masked
agent), a comms mask with reachable/denied peers, monotonic agent attrition (one agent
terminates), and a horizon truncation. It passes
:func:`astro_mine.core.env.check_environment` (asserted in the tests).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from astro_mine.core.env.model import AgentId, ResetResult, StepResult
from astro_mine.core.messages.model import (
    ActionBatch,
    CommsObservationMask,
    Observation,
    PeerLink,
    Quat,
    SensorReading,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.sadf.enums import CapabilityTag, CommsBand, NodeRole, SensorKind
from astro_mine.core.sadf.model import Asset, Comms, Identity, Sensor
from astro_mine.core.units import MOON_BODY_FIXED

#: The three heterogeneous agents (distinct capability + sensor sets).
AGENTS: tuple[AgentId, ...] = ("rover", "digger", "relay")


def make_fake_swarm_env() -> Any:
    """A zero-arg :class:`~astro_mine.learn.envs.SwarmEnv` factory over the fake world.

    An importable ``module:attr`` env factory — the shape ``train/run.py``'s ``--env-factory``
    resolves (a real run points it at a Sim-backed Core Environment factory instead)."""
    from astro_mine.learn.envs import make_swarm_env

    return make_swarm_env(FakeSwarmWorld(), build_assets())


def build_assets() -> dict[AgentId, Asset]:
    """Three SADF assets with deliberately different capability/sensor suites."""
    return {
        "rover": Asset(
            identity=Identity(id="rover", name="Prospecting Rover", version="0.1.0", kind="rover"),
            capabilities=[
                CapabilityTag.MOBILITY_WHEELED,
                CapabilityTag.PROSPECTING_NEUTRON,
                CapabilityTag.SENSING_IMAGING,
                CapabilityTag.COMMS_DIRECT_TO_EARTH,
            ],
            root_frame="body",
            sensors=[
                Sensor(name="cam", kind=SensorKind.IMAGING, frame="body"),
                Sensor(name="neutron", kind=SensorKind.NEUTRON_SPECTROMETER, frame="body"),
            ],
            comms=[Comms(name="radio", band=CommsBand.S_BAND, node_role=NodeRole.SPACE)],
        ),
        "digger": Asset(
            identity=Identity(id="digger", name="Excavator", version="0.1.0", kind="excavator"),
            capabilities=[
                CapabilityTag.MOBILITY_TRACKED,
                CapabilityTag.EXCAVATION_BUCKET,
                CapabilityTag.SENSING_LIDAR,
            ],
            root_frame="body",
            sensors=[Sensor(name="lidar", kind=SensorKind.LIDAR, frame="body")],
        ),
        "relay": Asset(
            identity=Identity(id="relay", name="Relay Orbiter", version="0.1.0", kind="orbiter"),
            capabilities=[
                CapabilityTag.MOBILITY_ORBITER,
                CapabilityTag.COMMS_RELAY,
                CapabilityTag.SENSING_IMAGING,
            ],
            root_frame="body",
            sensors=[Sensor(name="cam", kind=SensorKind.IMAGING, frame="body")],
            comms=[Comms(name="xband", band=CommsBand.X_BAND, relay=True)],
        ),
    }


class FakeSwarmWorld:
    """A deterministic, conformant Core Environment for exercising the adapter.

    ``relay`` is unobservable on ticks where ``tick % 3 == 2`` (partial observability);
    ``digger`` terminates at ``terminate_at`` (monotonic attrition); every remaining
    agent truncates at ``horizon``. Actions are ignored — dynamics are a pure function
    of the tick — so the trace is byte-identical under a fixed seed."""

    def __init__(self, *, horizon: int = 12, terminate_at: int = 3) -> None:
        self._horizon = horizon
        self._terminate_at = terminate_at
        self._tick = 0
        self._seed = 0
        self._active: tuple[AgentId, ...] = AGENTS

    @property
    def possible_agents(self) -> tuple[AgentId, ...]:
        return AGENTS

    @property
    def agents(self) -> tuple[AgentId, ...]:
        return self._active

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        self._seed = 0 if seed is None else seed
        self._tick = 0
        self._active = AGENTS
        observations = {agent: self._observe(agent, 0) for agent in self._active}
        return ResetResult(observations=observations)

    def step(self, actions: ActionBatch) -> StepResult:
        self._tick += 1
        tick = self._tick
        active = self._active  # pre-retirement: a terminating agent still reports this tick
        observations = {agent: self._observe(agent, tick) for agent in active}
        terminations = {
            agent: (agent == "digger" and tick >= self._terminate_at) for agent in active
        }
        truncated = tick >= self._horizon
        truncations = {agent: truncated for agent in active}

        retired = {agent for agent, done in terminations.items() if done}
        self._active = () if truncated else tuple(a for a in active if a not in retired)
        return StepResult(
            observations=observations,
            sim_time_s=float(tick),
            terminations=terminations,
            truncations=truncations,
            dt_s=1.0,
        )

    # --- deterministic observation synthesis ---------------------------------------

    def _observe(self, agent: AgentId, tick: int) -> Observation:
        observable = not (agent == "relay" and tick % 3 == 2)
        state = self._state(agent, tick)
        neighbors = [self._state(other, tick) for other in self._active if other != agent]
        return Observation(
            tick=tick,
            sim_time_s=float(tick),
            agent_id=agent,
            observable=observable,
            self_state=state,
            sensors=self._sensors(agent, tick),
            comms=self._comms(agent, tick),
            neighbors=neighbors,
        )

    def _state(self, agent: AgentId, tick: int) -> StateSample:
        idx = AGENTS.index(agent)
        return StateSample(
            agent_id=agent,
            frame=MOON_BODY_FIXED,
            pose=Transform(
                translation_m=Vec3(x=float(idx * 100 + tick), y=float(tick * 2), z=0.0),
                rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            battery_soc_j=1.0e5 - tick * 10.0 + self._seed * 0.0,
            temperature_k=200.0 + tick,
            mode="idle",
        )

    def _sensors(self, agent: AgentId, tick: int) -> list[SensorReading]:
        if agent == "rover":
            return [
                SensorReading(sensor="cam", values=[tick * 1.0 + 0.1], unit="dn"),
                SensorReading(
                    sensor="neutron",
                    values=[tick * 0.5 + 0.2, 42.0],
                    resource_species="water_equivalent_hydrogen",
                ),
            ]
        if agent == "digger":
            return [SensorReading(sensor="lidar", values=[tick * 2.0])]
        return [SensorReading(sensor="cam", values=[tick * 1.0 + 0.3])]

    def _comms(self, agent: AgentId, tick: int) -> CommsObservationMask | None:
        if agent == "rover":
            return CommsObservationMask(
                agent_id="rover",
                links=[
                    PeerLink(
                        peer="relay", reachable=True, rate_bps=1.0e6, latency_s=0.1, margin_db=3.0
                    ),
                    PeerLink(peer="digger", reachable=False),
                ],
                earth_contact=(tick % 4 == 0),
            )
        if agent == "relay":
            return CommsObservationMask(
                agent_id="relay",
                links=[PeerLink(peer="rover", reachable=True, rate_bps=2.0e6)],
                earth_contact=False,
            )
        return None
