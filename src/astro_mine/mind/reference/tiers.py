"""Reference tier plugins — replaceable examples, not privileged internals.

Per conventions.md §1.3 ("reference implementations ship as replaceable examples"), these
are the minimal, deterministic tier/shield policies that make the spine runnable and
testable end-to-end without the heavyweight backends (PDDL/OMPL/ONNX arrive as drop-in
replacements in RM-P1-MIND-03). They implement a toy lunar-polar-prospecting loop:

- :class:`ScriptedMissionPlanner` assigns each agent a prospect region (strategic);
- :class:`ScriptedTampPlanner` turns the assigned region into a GOTO toward its centre
  (tactical), reading the mission tier's output from ``context.upstream``;
- :class:`ScriptedController` closes the loop with a clamped velocity setpoint toward the
  GOTO target, reading the agent's current pose from the observation (reactive);
- :class:`PassthroughShield` is the local-dev stand-in for Guard's ``PolicyShield`` — it
  passes the proposed action through unchanged. RM-P1-MIND-05 registers the real shield.

Each tier is a Core :class:`~astro_mine.core.policy.protocol.Policy` (``decide``); every one
is deterministic — no clock or RNG of its own — so a seeded run reproduces exactly. The
providers at the bottom bundle each policy with its Core manifest for the registry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib import resources
from typing import Any

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ActionKind, ControlMode, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    GotoTask,
    Observation,
    ProspectTask,
    Quat,
    TaskDirective,
    Transform,
    Vec3,
    Volume,
)
from astro_mine.core.policy.guardrail import InterventionKind, ShieldReport
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry.loader import load_manifest
from astro_mine.core.registry.model import PluginManifest
from astro_mine.core.registry.tier import TierPlugin

__all__ = [
    "ConstraintShield",
    "PassthroughShield",
    "ScriptedController",
    "ScriptedMissionPlanner",
    "ScriptedTampPlanner",
    "constraint_shield_plugin",
    "control_plugin",
    "mission_plugin",
    "shield_plugin",
    "tamp_plugin",
]

#: The body-fixed frame the toy scenario resolves geometry in.
FRAME = "body"
_IDENTITY_ROTATION = Quat(x=0.0, y=0.0, z=0.0, w=1.0)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class ScriptedMissionPlanner:
    """Assigns each agent a fixed prospect region along the x-axis (deterministic)."""

    def __init__(self, *, spacing_m: float = 10.0) -> None:
        self._spacing_m = spacing_m

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        actions = []
        for index, agent_id in enumerate(sorted(observations)):
            center = Vec3(x=self._spacing_m * (index + 1), y=0.0, z=0.0)
            region = Volume(frame=FRAME, center_m=center, dimensions_m=Vec3(x=2.0, y=2.0, z=2.0))
            actions.append(
                Action(
                    agent_id=agent_id,
                    kind=ActionKind.TASK,
                    task=TaskDirective(
                        task_kind=TaskKind.PROSPECT, prospect=ProspectTask(region=region)
                    ),
                )
            )
        return ActionBatch(actions=actions)


class ScriptedTampPlanner:
    """Turns each agent's assigned prospect region (from ``context.upstream``) into a GOTO
    toward the region centre."""

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        by_agent = {action.agent_id: action for action in upstream.actions}
        actions = []
        for agent_id in sorted(observations):
            assigned = by_agent.get(agent_id)
            if (
                assigned is not None
                and assigned.task is not None
                and assigned.task.prospect is not None
            ):
                center = assigned.task.prospect.region.center_m
            else:
                center = Vec3(x=0.0, y=0.0, z=0.0)
            target = Transform(translation_m=center, rotation_quat_xyzw=_IDENTITY_ROTATION)
            actions.append(
                Action(
                    agent_id=agent_id,
                    kind=ActionKind.TASK,
                    task=TaskDirective(
                        task_kind=TaskKind.GOTO,
                        goto=GotoTask(target_frame=FRAME, target_pose=target),
                    ),
                )
            )
        return ActionBatch(actions=actions)


class ScriptedController:
    """Emits a clamped velocity setpoint toward each agent's GOTO target (from
    ``context.upstream``), using the agent's current pose from the observation."""

    def __init__(self, *, gain: float = 0.5, max_speed_mps: float = 2.0) -> None:
        self._gain = gain
        self._max_speed_mps = max_speed_mps

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        by_agent = {action.agent_id: action for action in upstream.actions}
        actions = []
        for agent_id in sorted(observations):
            position = observations[agent_id].self_state.pose.translation_m
            assigned = by_agent.get(agent_id)
            if (
                assigned is not None
                and assigned.task is not None
                and assigned.task.goto is not None
            ):
                target = assigned.task.goto.target_pose.translation_m
            else:
                target = position
            vx = _clamp((target.x - position.x) * self._gain, self._max_speed_mps)
            vy = _clamp((target.y - position.y) * self._gain, self._max_speed_mps)
            actions.append(
                Action(
                    agent_id=agent_id,
                    kind=ActionKind.ACTUATOR,
                    actuator=ActuatorCommand(
                        target="base",
                        control_mode=ControlMode.VELOCITY,
                        setpoint=[vx, vy],
                        unit="m/s",
                    ),
                )
            )
        return ActionBatch(actions=actions)


class PassthroughShield:
    """Local-dev stand-in for Guard's ``PolicyShield``: returns the proposed action
    unchanged. Explicitly a non-enforcing placeholder — the enforcing reference shield is
    :class:`ConstraintShield`, and the real Guard shield (RM-P1-MIND-05) enforces hard
    constraints independently of any tier."""

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        return context.upstream if context.upstream is not None else ActionBatch()


class ConstraintShield:
    """Reference *enforcing* shield — a deterministic stand-in for Guard's ``PolicyShield``
    (RM-P1-MIND-05). It projects every actuator VELOCITY setpoint onto a hard
    speed-magnitude ceiling (a ``kinematic_limit`` constraint, RFC-0004): a setpoint over the
    ceiling is scaled back onto it (a ``shield_edit``), and the intervention is surfaced
    through the :class:`~astro_mine.core.policy.guardrail.ReportingShield` seam so it lands in
    the decision trace with its clause id. Independence holds structurally: the ceiling is the
    shield's, not the controller's — a learned or classical tier cannot raise it. Deterministic
    (a pure function of the proposed batch), so a seeded run reproduces its interventions."""

    #: The ``SafetySpec`` constraint id this reference clause stands in for (RFC-0004).
    CLAUSE = "ref.kinematic_limit.base_speed"

    def __init__(self, *, max_speed_mps: float = 1.5) -> None:
        self._max_speed_mps = max_speed_mps
        self._last_report: ShieldReport | None = None

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        emitted: list[Action] = []
        intervened = False
        for action in upstream.actions:
            shielded = action
            actuator = action.actuator
            if actuator is not None and actuator.control_mode is ControlMode.VELOCITY:
                speed = math.hypot(*actuator.setpoint[:2]) if actuator.setpoint else 0.0
                if speed > self._max_speed_mps:
                    scale = self._max_speed_mps / speed
                    clamped = [v * scale for v in actuator.setpoint]
                    shielded = action.model_copy(
                        update={"actuator": actuator.model_copy(update={"setpoint": clamped})}
                    )
                    intervened = True
            emitted.append(shielded)
        self._last_report = ShieldReport(
            intervened=intervened,
            kind=InterventionKind.SHIELD_EDIT if intervened else None,
            clauses=(self.CLAUSE,) if intervened else (),
        )
        return ActionBatch(actions=emitted)

    def report(self) -> ShieldReport | None:
        return self._last_report


def _manifest(filename: str) -> PluginManifest:
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("manifests", filename)
        .read_text(encoding="utf-8")
    )
    return load_manifest(text).manifest


def mission_plugin() -> TierPlugin:
    """Provider for the reference mission planner (entry point)."""
    return TierPlugin(
        manifest=_manifest("mission.yaml"),
        factory=lambda params: ScriptedMissionPlanner(**_mission_params(params)),
    )


def tamp_plugin() -> TierPlugin:
    """Provider for the reference TAMP planner (entry point)."""
    return TierPlugin(manifest=_manifest("tamp.yaml"), factory=lambda params: ScriptedTampPlanner())


def control_plugin() -> TierPlugin:
    """Provider for the reference controller (entry point)."""
    return TierPlugin(
        manifest=_manifest("control.yaml"),
        factory=lambda params: ScriptedController(**_control_params(params)),
    )


def shield_plugin() -> TierPlugin:
    """Provider for the reference pass-through shield (entry point)."""
    return TierPlugin(manifest=_manifest("shield.yaml"), factory=lambda params: PassthroughShield())


def constraint_shield_plugin() -> TierPlugin:
    """Provider for the reference enforcing :class:`ConstraintShield` (entry point)."""
    return TierPlugin(
        manifest=_manifest("constraint_shield.yaml"),
        factory=lambda params: ConstraintShield(**_constraint_shield_params(params)),
    )


def _mission_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {"spacing_m": float(params["spacing_m"])} if "spacing_m" in params else {}


def _control_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "gain" in params:
        out["gain"] = float(params["gain"])
    if "max_speed_mps" in params:
        out["max_speed_mps"] = float(params["max_speed_mps"])
    return out


def _constraint_shield_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {"max_speed_mps": float(params["max_speed_mps"])} if "max_speed_mps" in params else {}
