# SPDX-License-Identifier: Apache-2.0
"""The untrusted falsification plant: a double-integrator rollout driving a Core policy.

The minimal, **stdlib-only** (no numpy — Guard carries none; no Sim import) closed-loop the
adversarial falsification suite runs against the shield (RM-P1-GUARD-05; guard.md §10 "falsification
is the central validation strategy"). Each tick it builds a Core
:class:`~astro_mine.core.messages.Observation` from the plant state, calls a Core
:class:`~astro_mine.core.policy.protocol.Policy` — a :class:`~astro_mine.guard.wrap.PolicyShield`
(shielded) or the raw untrusted policy (the deliberately **unshielded control**) — reads the
certified action off the returned :class:`~astro_mine.core.messages.ActionBatch`, and integrates the
point-mass **double integrator** ``pos += v·dt; v += a·dt`` that matches the CBF's commanded-
acceleration model (guard.md §3).

The plant is *untrusted tooling that attacks the TCB from outside* (issue #5, "the falsification
search harness is untrusted tooling … never part of the TCB"): it only reads the shield's public
:meth:`~astro_mine.guard.wrap.PolicyShield.decide` output and the best-effort verdict stream, never
the trusted core's internals. Scalar safety signals (energy, thermal, torque, speed) ride on the
observation as one-value sensor readings, so the shield's default
:class:`~astro_mine.guard.wrap.DefaultSignalResolver` resolves them; the adversary evolves them tick
to tick to drive the state toward the constraint boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    Observation,
    Quat,
    SensorReading,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.units import ReferenceFrame
from astro_mine.core.units.enums import FrameClass
from astro_mine.guard.audit.sink import CollectingSink

if TYPE_CHECKING:
    from astro_mine.core.policy import AgentId, Policy
    from astro_mine.guard.audit.model import SafetyVerdict
    from astro_mine.guard.falsify.adversary import Adversary
    from astro_mine.guard.wrap import PolicyShield

__all__ = [
    "DEFAULT_DT",
    "DEFAULT_FRAME",
    "DEFAULT_U_MAX",
    "AdversaryPolicy",
    "PlantState",
    "RolloutStep",
    "control_rollout",
    "shielded_rollout",
]

#: The plant integration step. Fine enough that the forward-Euler discretization of the CBF's
#: commanded-acceleration model stays inside the safe set (the one-step overshoot bound is
#: ``u_max·dt²``; see :mod:`astro_mine.guard.falsify.oracle`).
DEFAULT_DT = 0.05
#: The per-component commanded-acceleration bound (matches ``CoreConfig.u_max``).
DEFAULT_U_MAX = 20.0
#: The body-fixed lunar frame the anchor keep-out geometry is expressed in (LUNAR-TR-001).
DEFAULT_FRAME = "MOON_ME"

_AGENT: str = "rover"


@dataclass(frozen=True, slots=True)
class PlantState:
    """One tick of the double-integrator plant: position, velocity, and the safety signal vector."""

    position: tuple[float, ...]
    velocity: tuple[float, ...]
    signals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RolloutStep:
    """The record the oracle inspects: the state *before* this tick's action, the certified action
    the policy returned, and (shielded only) the per-tick :class:`SafetyVerdict`."""

    index: int
    state: PlantState
    certified_action: tuple[float, ...]
    verdict: SafetyVerdict | None


def _observation(agent_id: str, state: PlantState, frame_name: str, index: int) -> Observation:
    """Build a per-agent Observation from the plant state (signals ride as sensor readings)."""
    frame = ReferenceFrame(name=frame_name, frame_class=FrameClass.BODY_FIXED, center="MOON")
    pos = _fit(state.position, 3)
    vel = _fit(state.velocity, 3)
    return Observation(
        tick=index,
        sim_time_s=float(index),
        agent_id=agent_id,
        self_state=StateSample(
            agent_id=agent_id,
            frame=frame,
            pose=Transform(
                translation_m=Vec3(x=pos[0], y=pos[1], z=pos[2]),
                rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            linear_velocity_mps=Vec3(x=vel[0], y=vel[1], z=vel[2]),
        ),
        sensors=[SensorReading(sensor=k, values=[v]) for k, v in state.signals.items()],
    )


def _effort_action(agent_id: str, accel: Sequence[float]) -> Action:
    """An ACTUATOR/EFFORT action carrying a commanded-acceleration setpoint (shieldable)."""
    return Action(
        agent_id=agent_id,
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="body", control_mode=ControlMode.EFFORT, setpoint=[float(x) for x in accel]
        ),
    )


def _fit(values: Sequence[float], n: int) -> list[float]:
    """Pad with zeros / truncate ``values`` to exactly ``n`` elements."""
    return [float(x) for x in (list(values) + [0.0] * n)[:n]]


class AdversaryPolicy:
    """Presents an :class:`~astro_mine.guard.falsify.adversary.Adversary`'s proposed action as a
    Core :class:`~astro_mine.core.policy.protocol.Policy` — the *untrusted wrapped policy* the
    shield treats as adversarial input (guard.md §9.1).

    Reads only position/velocity off each observation (never the shield's internals), asks the
    adversary for a proposed commanded acceleration, and emits it as an EFFORT action. Unshielded,
    this policy *is* the control: its raw proposals drive the plant straight into violations."""

    def __init__(self, adversary: Adversary, *, spatial_dim: int, agent_id: str = _AGENT) -> None:
        self._adversary = adversary
        self._spatial_dim = spatial_dim
        self._agent_id = agent_id
        self._index = 0

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        actions: list[Action] = []
        for agent_id, obs in observations.items():
            t = obs.self_state.pose.translation_m
            v = obs.self_state.linear_velocity_mps
            position = _fit([t.x, t.y, t.z], self._spatial_dim)
            velocity = _fit([v.x, v.y, v.z] if v is not None else [], self._spatial_dim)
            proposed = self._adversary.action(self._index, position, velocity, self._spatial_dim)
            actions.append(_effort_action(str(agent_id), proposed))
        self._index += 1
        return ActionBatch(actions=actions)


def _integrate(
    state: PlantState,
    certified: Sequence[float],
    disturbance: Sequence[float],
    next_signals: Mapping[str, float],
    *,
    dt: float,
    spatial_dim: int,
) -> PlantState:
    """Advance the double integrator one step: ``pos += v·dt`` then ``v += a·dt`` (explicit Euler),
    with the adversary's bounded external acceleration added to the certified command."""
    pos = _fit(state.position, spatial_dim)
    vel = _fit(state.velocity, spatial_dim)
    cert = _fit(certified, spatial_dim)
    dist = _fit(disturbance, spatial_dim)
    new_pos = [pos[i] + vel[i] * dt for i in range(spatial_dim)]
    new_vel = [vel[i] + (cert[i] + dist[i]) * dt for i in range(spatial_dim)]
    return PlantState(tuple(new_pos), tuple(new_vel), dict(next_signals))


def _rollout(
    policy: Policy,
    adversary: Adversary,
    *,
    spatial_dim: int,
    initial: PlantState,
    horizon: int,
    dt: float,
    agent_id: str,
    frame_name: str,
    sink: CollectingSink | None,
) -> list[RolloutStep]:
    context = DecisionContext()
    state = initial
    steps: list[RolloutStep] = []
    for index in range(horizon):
        obs = _observation(agent_id, state, frame_name, index)
        before = len(sink.verdicts) if sink is not None else 0
        batch = policy.decide({agent_id: obs}, context)
        certified = tuple(float(x) for x in batch.actions[0].actuator.setpoint)  # type: ignore[union-attr]
        verdict = sink.verdicts[-1] if sink is not None and len(sink.verdicts) > before else None
        steps.append(
            RolloutStep(index=index, state=state, certified_action=certified, verdict=verdict)
        )
        disturbance = adversary.accel_disturbance(
            index, list(state.position), list(state.velocity), spatial_dim
        )
        next_signals = adversary.next_signals(index, dict(state.signals))
        state = _integrate(
            state, certified, disturbance, next_signals, dt=dt, spatial_dim=spatial_dim
        )
    return steps


def shielded_rollout(
    shield: PolicyShield,
    adversary: Adversary,
    *,
    initial: PlantState,
    horizon: int,
    sink: CollectingSink,
    dt: float = DEFAULT_DT,
    agent_id: str = _AGENT,
    frame_name: str = DEFAULT_FRAME,
) -> list[RolloutStep]:
    """Drive a constructed :class:`~astro_mine.guard.wrap.PolicyShield` through the plant, capturing
    each tick's certified action and :class:`SafetyVerdict` (the shield must have been built with
    ``sink``). The wrapped policy may be an :class:`AdversaryPolicy` or a learned
    :class:`~astro_mine.core.policy.OnnxPolicy`; the disturbances/signal evolution come from
    ``adversary``."""
    return _rollout(
        shield,
        adversary,
        spatial_dim=shield.spatial_dim,
        initial=initial,
        horizon=horizon,
        dt=dt,
        agent_id=agent_id,
        frame_name=frame_name,
        sink=sink,
    )


def control_rollout(
    adversary: Adversary,
    *,
    spatial_dim: int,
    initial: PlantState,
    horizon: int,
    dt: float = DEFAULT_DT,
    agent_id: str = _AGENT,
    frame_name: str = DEFAULT_FRAME,
) -> list[RolloutStep]:
    """The **unshielded control**: drive the raw adversarial policy (no shield) through the same
    plant. Its uncorrected proposals must produce violations — proving the falsification search is
    real, not vacuous (issue #5)."""
    return _rollout(
        AdversaryPolicy(adversary, spatial_dim=spatial_dim, agent_id=agent_id),
        adversary,
        spatial_dim=spatial_dim,
        initial=initial,
        horizon=horizon,
        dt=dt,
        agent_id=agent_id,
        frame_name=frame_name,
        sink=None,
    )
