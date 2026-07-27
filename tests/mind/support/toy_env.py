"""A deterministic multi-agent Environment for exercising the Mind spine.

Implements the Core :class:`~astro_mine.core.env.protocol.Environment` contract without any
sibling-package import (Mind never imports Sim; the real Sim environment is injected
downstream through this same Core contract). Agents integrate the velocity setpoints the
reference controller emits over a fixed timestep toward their assigned prospect regions —
a toy stand-in for the lunar-polar-prospecting anchor scenario, just enough to demonstrate
the composed hierarchy stepping an environment and to gate the trace for determinism.

Fully deterministic: no clock or RNG (the ``seed`` is accepted and ignored — positions are a
pure function of the actions), so the same actions always yield the same observations, as
:func:`~astro_mine.core.env.conformance.check_environment` requires. ``comms_denied_ticks``
optionally drops Earth contact on chosen ticks (the input the RM-P1-MIND-06 comms-regime
work builds on).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from astro_mine.core.env.model import AgentId, ResetResult, StepResult
from astro_mine.core.messages.model import (
    ActionBatch,
    CommsObservationMask,
    Observation,
    Quat,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.units import MOON_BODY_FIXED

__all__ = ["ToyProspectingEnv"]


class ToyProspectingEnv:
    """A deterministic toy prospecting environment (Core Environment contract)."""

    def __init__(
        self,
        *,
        n_agents: int = 2,
        dt_s: float = 1.0,
        horizon: int = 8,
        terminate_at: int | None = None,
        comms_denied_ticks: Iterable[int] = (),
    ) -> None:
        self._ids: tuple[AgentId, ...] = tuple(f"rover-{i}" for i in range(n_agents))
        self._dt_s = dt_s
        self._horizon = horizon
        self._terminate_at = terminate_at
        self._comms_denied = frozenset(comms_denied_ticks)
        self._pos: dict[AgentId, list[float]] = {}
        self._tick = 0

    @property
    def possible_agents(self) -> tuple[AgentId, ...]:
        return self._ids

    @property
    def agents(self) -> tuple[AgentId, ...]:
        return self._ids

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        self._tick = 0
        self._pos = {agent_id: [0.0, 0.0] for agent_id in self._ids}
        return ResetResult(observations=self._observe())

    def step(self, actions: ActionBatch) -> StepResult:
        commands = {action.agent_id: action for action in actions.actions}
        for agent_id in self._ids:
            action = commands.get(agent_id)
            if (
                action is not None
                and action.actuator is not None
                and len(action.actuator.setpoint) >= 2
            ):
                self._pos[agent_id][0] += action.actuator.setpoint[0] * self._dt_s
                self._pos[agent_id][1] += action.actuator.setpoint[1] * self._dt_s
        self._tick += 1
        truncated = self._tick >= self._horizon
        terminated = self._terminate_at is not None and self._tick >= self._terminate_at
        return StepResult(
            observations=self._observe(),
            sim_time_s=self._tick * self._dt_s,
            terminations={agent_id: terminated for agent_id in self._ids},
            truncations={agent_id: truncated for agent_id in self._ids},
            dt_s=self._dt_s,
        )

    def _observe(self) -> dict[AgentId, Observation]:
        sim_time_s = self._tick * self._dt_s
        earth_contact = self._tick not in self._comms_denied
        observations: dict[AgentId, Observation] = {}
        for agent_id in self._ids:
            x, y = self._pos[agent_id]
            state = StateSample(
                agent_id=agent_id,
                frame=MOON_BODY_FIXED,
                pose=Transform(
                    translation_m=Vec3(x=x, y=y, z=0.0),
                    rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
            observations[agent_id] = Observation(
                tick=self._tick,
                sim_time_s=sim_time_s,
                agent_id=agent_id,
                observable=True,
                self_state=state,
                comms=CommsObservationMask(
                    agent_id=agent_id, links=[], earth_contact=earth_contact
                ),
            )
        return observations
