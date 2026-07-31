"""The ``unified-planning`` mission backend (RM-P1-MIND-03) — the ``[pddl]`` extra.

The production realization of the PDDL/temporal mission tier that mind.md §4/§11 names: a real
symbolic planner solves the problem :mod:`~astro_mine.mind.mission.planner.pddl` generates from
the belief, and the engine's plan — not a hand-rolled assignment — becomes the mission
decomposition.

**unified-planning** is a backend-agnostic façade: the engine is resolved by *problem kind*, so
whichever solver is installed (``up-fast-downward``, ``up-enhsp``, OPTIC, …) is used without a
code change; ``engine_name`` pins one explicitly when a scenario must be reproducible against a
specific solver. This is exactly the "commit to the interface, not the backend" posture
(principle 2) one level down: Mind commits to PDDL, not to Fast Downward.

**How the decomposition is derived.** The generated domain models the *assignment decision*
itself (``assign`` binds a free agent to an unassigned region; ``prospect`` requires that
binding), so the engine derives who prospects what from the goal "every region prospected". The
adapter reads the ``assign`` steps out of the returned plan and maps each bound region back to
its :func:`~astro_mine.mind.mission.planner.reference.region_volume` geometry. An agent the plan
leaves unbound simply gets no action this tick — a defined, safe result.

**Determinism.** Fast Downward's search is deterministic given the same problem text and engine
build, so a replan reproduces within a pinned environment (``determinism_class: bit_exact``). It
is *not* guaranteed byte-identical across engine versions — which is why the pure-Python
reference planner, not this one, is the CI-tested default and the golden-trace baseline.

Every ``unified_planning`` import is deferred into the call: the base wheel ships without the
extra, and this module is imported by the entry-point provider on every registry construction.
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
)
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry.loader import load_manifest
from astro_mine.core.registry.model import PluginManifest
from astro_mine.core.registry.tier import TierPlugin
from astro_mine.mind.mission.planner.pddl import (
    PddlProblem,
    prospecting_domain,
    region_name,
    render_problem,
)
from astro_mine.mind.mission.planner.reference import DEFAULT_SPACING_M, region_volume

__all__ = ["UnifiedPlanningMissionPlanner", "up_mission_plugin"]

#: The domain action whose grounded arguments carry the agent→region decomposition.
_ASSIGN = "assign"


class MissionPlanningError(Exception):
    """Raised when the ``[pddl]`` extra is absent or the engine returns no plan."""


class UnifiedPlanningMissionPlanner:
    """Mission tier backed by a real PDDL engine via ``unified-planning``.

    ``engine_name`` pins a specific solver (e.g. ``"fast-downward"``); ``None`` lets
    unified-planning pick by problem kind. ``spacing_m`` fixes the region geometry (shared with
    the reference planner). ``timeout_s`` bounds the solve — a mission replan runs under a
    deadline (mind.md §8), and an exhausted deadline degrades to *no assignment this tick*
    rather than an exception (principle 4).
    """

    def __init__(
        self,
        *,
        engine_name: str | None = None,
        spacing_m: float = DEFAULT_SPACING_M,
        timeout_s: float | None = 30.0,
    ) -> None:
        self._engine_name = engine_name
        self._spacing_m = spacing_m
        self._timeout_s = timeout_s

    def problem(self, observations: Mapping[AgentId, Observation]) -> PddlProblem:
        """The PDDL problem generated for ``observations`` (one region per observed agent)."""
        agents = tuple(sorted(observations))
        regions = tuple(region_name(index) for index in range(len(agents)))
        return PddlProblem(name="lunar-prospecting-replan", agents=agents, regions=regions)

    def pddl(self, observations: Mapping[AgentId, Observation]) -> tuple[str, str]:
        """The ``(domain, problem)`` PDDL text handed to the engine."""
        return prospecting_domain(), render_problem(self.problem(observations))

    def solve(self, observations: Mapping[AgentId, Observation]) -> dict[AgentId, int]:
        """Solve the replan; return the engine's ``agent → region index`` decomposition.

        Raises :class:`MissionPlanningError` if the ``[pddl]`` extra is not installed.
        """
        try:
            from unified_planning.engines import PlanGenerationResultStatus
            from unified_planning.io import PDDLReader
            from unified_planning.shortcuts import OneshotPlanner, get_environment
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissionPlanningError(
                "the unified-planning mission backend needs the [pddl] extra: "
                "pip install 'astro-mine-platform[mind-pddl]'"
            ) from exc

        # unified-planning prints an engine credits banner on first use; silence it so a Mind
        # run's stdout stays the decision trace and nothing else.
        get_environment().credits_stream = None

        domain, problem_text = self.pddl(observations)
        problem = PDDLReader().parse_problem_string(domain, problem_text)
        with OneshotPlanner(name=self._engine_name, problem_kind=problem.kind) as engine:
            result = engine.solve(problem, timeout=self._timeout_s)
        solved = {
            PlanGenerationResultStatus.SOLVED_SATISFICING,
            PlanGenerationResultStatus.SOLVED_OPTIMALLY,
        }
        if result.status not in solved or result.plan is None:
            return {}
        return self._decomposition(result.plan)

    @staticmethod
    def _decomposition(plan: Any) -> dict[AgentId, int]:
        """Read the ``assign(agent, region)`` steps out of the engine's plan."""
        assignment: dict[AgentId, int] = {}
        for action in plan.actions:
            if action.action.name != _ASSIGN:
                continue
            agent, region = (str(param) for param in action.actual_parameters)
            assignment[agent] = int(region.removeprefix("r"))
        return assignment

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        assignment = self.solve(observations)
        actions = [
            Action(
                agent_id=agent_id,
                kind=ActionKind.TASK,
                task=TaskDirective(
                    task_kind=TaskKind.PROSPECT,
                    prospect=ProspectTask(region=region_volume(index, self._spacing_m)),
                ),
            )
            for agent_id, index in sorted(assignment.items())
        ]
        return ActionBatch(actions=actions)


def _manifest(filename: str) -> PluginManifest:
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("manifests", filename)
        .read_text(encoding="utf-8")
    )
    return load_manifest(text).manifest


def _params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "engine_name" in params:
        out["engine_name"] = str(params["engine_name"])
    if "spacing_m" in params:
        out["spacing_m"] = float(params["spacing_m"])
    if "timeout_s" in params:
        out["timeout_s"] = float(params["timeout_s"])
    return out


def up_mission_plugin() -> TierPlugin:
    """Provider for the unified-planning mission backend (entry point).

    The factory defers construction (and therefore the ``unified_planning`` import) until a
    stack actually binds this plugin, so discovering it costs nothing without the extra.
    """
    return TierPlugin(
        manifest=_manifest("up_mission.yaml"),
        factory=lambda params: UnifiedPlanningMissionPlanner(**_params(params)),
    )
