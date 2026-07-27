"""Reference TAMP backend — symbolic task interleaved with sampling motion (RM-P1-MIND-03).

The default behind Core's :class:`~astro_mine.core.policy.protocol.TaskMotionPlanner`
sub-interface: each tick it selects each agent's tactical target from the mission decomposition
(:class:`~astro_mine.mind.tamp.task.reference.ReferenceTaskPlanner`), plans a collision-free path
to it (:class:`~astro_mine.mind.tamp.motion.reference.ReferenceMotionPlanner`), and emits a GOTO
toward the first waypoint — the PDDLStream-style interleaving of symbolic task and geometric
feasibility (mind.md §4). Deterministic given the seed (the motion RNG is seeded from
``DecisionContext.seed``); an OMPL/FCL backend drops in behind the same motion contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import resources
from random import Random

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    GotoTask,
    Observation,
    Quat,
    TaskDirective,
    Transform,
    Vec3,
)
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry.loader import load_manifest
from astro_mine.core.registry.model import PluginManifest
from astro_mine.mind.registry.registry import TierPlugin
from astro_mine.mind.tamp.motion.protocol import MotionPlanner
from astro_mine.mind.tamp.motion.reference import Obstacle, ReferenceMotionPlanner
from astro_mine.mind.tamp.task.reference import ReferenceTaskPlanner

__all__ = ["ReferenceTampPlanner", "sampling_tamp_plugin"]

#: The body-fixed frame the toy scenario resolves geometry in.
FRAME = "body"
_IDENTITY_ROTATION = Quat(x=0.0, y=0.0, z=0.0, w=1.0)


class ReferenceTampPlanner:
    """Interleaves symbolic task selection with sampling motion feasibility."""

    def __init__(
        self,
        *,
        obstacles: Sequence[Obstacle] = (),
        motion: MotionPlanner | None = None,
        task: ReferenceTaskPlanner | None = None,
    ) -> None:
        self._obstacles = tuple(obstacles)
        self._motion = motion or ReferenceMotionPlanner()
        self._task = task or ReferenceTaskPlanner()

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        rng = Random(context.seed if context.seed is not None else 0)
        targets = self._task.targets(observations, upstream)
        actions = []
        for agent_id in sorted(observations):
            pose = observations[agent_id].self_state.pose.translation_m
            target = targets[agent_id]
            path = self._motion.plan((pose.x, pose.y), (target.x, target.y), self._obstacles, rng)
            waypoint = path[0] if path else (target.x, target.y)
            goto_pose = Transform(
                translation_m=Vec3(x=waypoint[0], y=waypoint[1], z=target.z),
                rotation_quat_xyzw=_IDENTITY_ROTATION,
            )
            actions.append(
                Action(
                    agent_id=agent_id,
                    kind=ActionKind.TASK,
                    task=TaskDirective(
                        task_kind=TaskKind.GOTO,
                        goto=GotoTask(target_frame=FRAME, target_pose=goto_pose),
                    ),
                )
            )
        return ActionBatch(actions=actions)


def _manifest(filename: str) -> PluginManifest:
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("manifests", filename)
        .read_text(encoding="utf-8")
    )
    return load_manifest(text).manifest


def sampling_tamp_plugin() -> TierPlugin:
    """Provider for the reference sampling TAMP planner (entry point)."""
    return TierPlugin(
        manifest=_manifest("sampling_tamp.yaml"), factory=lambda params: ReferenceTampPlanner()
    )
