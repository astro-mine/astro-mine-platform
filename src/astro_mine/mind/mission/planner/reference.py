# SPDX-License-Identifier: Apache-2.0
"""Reference PDDL/temporal mission planner (RM-P1-MIND-03).

The deterministic pure-Python default behind Core's
:class:`~astro_mine.core.policy.protocol.MissionPlanner` sub-interface: each replan it generates
a PDDL problem from the belief (the observed agents → a prospect region apiece), "solves" the
assignment deterministically, and emits the global decomposition as per-agent ``PROSPECT`` tasks.
A replaceable example (conventions.md §1.3) — a ``unified-planning`` engine consuming the same
:func:`~astro_mine.mind.mission.planner.pddl.render_problem` text drops in behind the same
sub-interface (``[pddl]`` extra). No clock or RNG of its own, so a seeded run reproduces exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from typing import Any

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    Observation,
    ProspectTask,
    TaskDirective,
    Vec3,
    Volume,
)
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry.loader import load_manifest
from astro_mine.core.registry.model import PluginManifest
from astro_mine.core.registry.tier import TierPlugin
from astro_mine.mind.mission.planner.pddl import PddlProblem, region_name, render_problem

__all__ = ["FRAME", "ReferenceMissionPlanner", "pddl_mission_plugin", "region_volume"]

#: The body-fixed frame the toy scenario resolves geometry in.
FRAME = "body"

#: Default along-track spacing between consecutive prospect regions.
DEFAULT_SPACING_M = 10.0


def region_volume(index: int, spacing_m: float = DEFAULT_SPACING_M) -> Volume:
    """The geometry of the ``index``-th prospect region (the region ``region_name(index)`` names).

    Shared by the reference planner and the native ``unified-planning`` adapter so a region means
    the same patch of ground whichever backend decomposed onto it.
    """
    center = Vec3(x=spacing_m * (index + 1), y=0.0, z=0.0)
    return Volume(frame=FRAME, center_m=center, dimensions_m=Vec3(x=2.0, y=2.0, z=2.0))


class ReferenceMissionPlanner:
    """Assigns each observed agent a prospect region via a generated PDDL problem."""

    def __init__(self, *, spacing_m: float = DEFAULT_SPACING_M) -> None:
        self._spacing_m = spacing_m

    def problem(self, observations: Mapping[AgentId, Observation]) -> PddlProblem:
        """The PDDL problem this planner generates for ``observations`` (one region/agent)."""
        agents = tuple(sorted(observations))
        regions = tuple(region_name(index) for index in range(len(agents)))
        return PddlProblem(name="lunar-prospecting-replan", agents=agents, regions=regions)

    def pddl(self, observations: Mapping[AgentId, Observation]) -> str:
        """The canonical PDDL problem text for ``observations`` (deterministic; hashable into
        plan provenance, RM-P1-MIND-07)."""
        return render_problem(self.problem(observations))

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        # Generating the problem each replan is the belief->PDDL step (mind.md §5); the
        # reference solve is the deterministic region assignment below.
        problem = self.problem(observations)
        actions = []
        for index, agent_id in enumerate(problem.agents):
            region = region_volume(index, self._spacing_m)
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


def _manifest(filename: str) -> PluginManifest:
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("manifests", filename)
        .read_text(encoding="utf-8")
    )
    return load_manifest(text).manifest


def _params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {"spacing_m": float(params["spacing_m"])} if "spacing_m" in params else {}


def pddl_mission_plugin() -> TierPlugin:
    """Provider for the reference PDDL mission planner (entry point)."""
    return TierPlugin(
        manifest=_manifest("pddl_mission.yaml"),
        factory=lambda params: ReferenceMissionPlanner(**_params(params)),
    )
