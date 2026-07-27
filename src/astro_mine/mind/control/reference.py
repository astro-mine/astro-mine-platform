"""Reference classical controllers — the baselines that always work (RM-P1-MIND-03).

Two pure, deterministic controllers behind Core's
:class:`~astro_mine.core.policy.protocol.Controller` sub-interface, each closing the loop toward
the agent's GOTO target (from ``DecisionContext.upstream``) with a clamped velocity setpoint:

- :class:`PidController` — proportional feedback on the position error. The reference is
  memoryless (integral/derivative gains default to zero) so it stays a pure function of the
  current observation, preserving cross-run reproducibility; a stateful PID binds through the
  same contract once a per-run reset seam lands.
- :class:`MpcController` — a receding-horizon controller for the velocity-integrator model:
  the constant velocity that reaches the target in ``horizon_s`` (closed-form for the linear
  model), clamped to the speed limit. Stateless and deterministic.

Both are Guard-wrapped like any tier — a controller cannot emit an un-shielded action.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from typing import Any

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    Observation,
    Vec3,
)
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry.loader import load_manifest
from astro_mine.core.registry.model import PluginManifest
from astro_mine.mind.registry.registry import TierPlugin

__all__ = ["MpcController", "PidController", "mpc_control_plugin", "pid_control_plugin"]


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _target_for(action: Action | None, fallback: Vec3) -> Vec3:
    if action is not None and action.task is not None and action.task.goto is not None:
        return action.task.goto.target_pose.translation_m
    return fallback


#: Dimensionality of the commanded velocity vector. The reference controllers are planar: the
#: toy env integrates a 2-vector, and the reference stacks (and their golden traces) are authored
#: against that. A **real** shield is not so relaxed — Guard's TCB certifies a command only when
#: ``len(setpoint) == spatial_dim`` of the compiled SafetySpec, and a spec with real keep-out
#: geometry is 3-D because the world is. A 2-vector handed to a 3-D model is *not certifiable*, and
#: the shield correctly falls back to a zero-effort hold — i.e. a planar controller under a real
#: 3-D safety contract yields a frozen swarm, not a shielded one.
#:
#: So dimensionality is a controller parameter (``dim``), not a constant. It defaults to 2 — the
#: reference stacks and goldens are unchanged — and the anchor stack, which binds the real Guard
#: shield, sets 3. Surface agents command no vertical velocity, so the third component is 0.0: the
#: vector is widened to match the safety model, never given new authority.
_DEFAULT_DIM = 2


def _velocity_action(agent_id: AgentId, vx: float, vy: float, dim: int = _DEFAULT_DIM) -> Action:
    setpoint = [vx, vy] if dim == 2 else [vx, vy, 0.0]
    return Action(
        agent_id=agent_id,
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="base", control_mode=ControlMode.VELOCITY, setpoint=setpoint, unit="m/s"
        ),
    )


def _dim(params: Mapping[str, Any]) -> int:
    """The commanded-vector dimensionality from a stack spec's ``params`` (2 or 3)."""
    dim = int(params.get("dim", _DEFAULT_DIM))
    if dim not in (2, 3):
        raise ValueError(f"control 'dim' must be 2 or 3, got {dim}")
    return dim


class PidController:
    """Proportional position controller (memoryless reference; integral/derivative = 0)."""

    def __init__(
        self, *, kp: float = 0.5, max_speed_mps: float = 2.0, dim: int = _DEFAULT_DIM
    ) -> None:
        self._kp = kp
        self._max_speed_mps = max_speed_mps
        self._dim = dim

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        by_agent = {action.agent_id: action for action in upstream.actions}
        actions = []
        for agent_id in sorted(observations):
            position = observations[agent_id].self_state.pose.translation_m
            target = _target_for(by_agent.get(agent_id), position)
            vx = _clamp((target.x - position.x) * self._kp, self._max_speed_mps)
            vy = _clamp((target.y - position.y) * self._kp, self._max_speed_mps)
            actions.append(_velocity_action(agent_id, vx, vy, self._dim))
        return ActionBatch(actions=actions)


class MpcController:
    """Receding-horizon controller: reach the target in ``horizon_s`` (velocity integrator)."""

    def __init__(
        self, *, horizon_s: float = 2.0, max_speed_mps: float = 2.0, dim: int = _DEFAULT_DIM
    ) -> None:
        self._horizon_s = horizon_s
        self._max_speed_mps = max_speed_mps
        self._dim = dim

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        by_agent = {action.agent_id: action for action in upstream.actions}
        actions = []
        for agent_id in sorted(observations):
            position = observations[agent_id].self_state.pose.translation_m
            target = _target_for(by_agent.get(agent_id), position)
            vx = _clamp((target.x - position.x) / self._horizon_s, self._max_speed_mps)
            vy = _clamp((target.y - position.y) / self._horizon_s, self._max_speed_mps)
            actions.append(_velocity_action(agent_id, vx, vy, self._dim))
        return ActionBatch(actions=actions)


def _manifest(filename: str) -> PluginManifest:
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("manifests", filename)
        .read_text(encoding="utf-8")
    )
    return load_manifest(text).manifest


def _pid_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"dim": _dim(params)}
    if "kp" in params:
        out["kp"] = float(params["kp"])
    if "max_speed_mps" in params:
        out["max_speed_mps"] = float(params["max_speed_mps"])
    return out


def _mpc_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"dim": _dim(params)}
    if "horizon_s" in params:
        out["horizon_s"] = float(params["horizon_s"])
    if "max_speed_mps" in params:
        out["max_speed_mps"] = float(params["max_speed_mps"])
    return out


def pid_control_plugin() -> TierPlugin:
    """Provider for the reference PID controller (entry point)."""
    return TierPlugin(
        manifest=_manifest("pid_control.yaml"),
        factory=lambda params: PidController(**_pid_params(params)),
    )


def mpc_control_plugin() -> TierPlugin:
    """Provider for the reference MPC controller (entry point)."""
    return TierPlugin(
        manifest=_manifest("mpc_control.yaml"),
        factory=lambda params: MpcController(**_mpc_params(params)),
    )
