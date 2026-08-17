# SPDX-License-Identifier: Apache-2.0
"""Reference greedy allocator — the delegated-to stand-in for Astro-Mine-Allocate (RM-P1-MIND-04).

A deterministic solver behind the Core :class:`~astro_mine.core.policy.protocol.Allocator`
sub-interface: it reads the :class:`~astro_mine.core.plan.allocation.AllocationRequest` the
adapter published under ``allocation.request``, assigns each asset to its nearest unassigned task
(greedy, stable), and returns the assignment as per-agent ``PROSPECT`` directives — TAMP's
roles/regions. It answers immediately with an ``incumbent`` (the anytime contract's good-enough
result) and records its solver name + seed as provenance so a delegated decision reproduces. A
replaceable example (conventions.md §1.3): the real ``astro_mine.allocate.AllocationPlanner``
(CP-SAT / OR-Tools) binds behind the same adapter via the companion entry-point shim. Mind embeds
no solver — this is the *stand-in for* the solver, wired through the registry.
"""

from __future__ import annotations

import math
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
)
from astro_mine.core.plan.allocation import (
    ALLOCATION_REQUEST_KEY,
    Allocation,
    AllocationAdapter,
    AllocationProvenance,
    AllocationRequest,
    Assignment,
)
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry.loader import load_manifest
from astro_mine.core.registry.model import PluginManifest
from astro_mine.core.registry.tier import TierPlugin

__all__ = ["GreedyReferenceAllocator", "greedy_allocator_plugin"]


class GreedyReferenceAllocator:
    """Nearest-task greedy assignment over the delegated request (deterministic incumbent)."""

    SOLVER = "mind.allocate.greedy"

    def __init__(self) -> None:
        self._last: Allocation | None = None

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        request = context.extras.get(ALLOCATION_REQUEST_KEY)
        provenance = AllocationProvenance(solver=self.SOLVER, seed=context.seed)
        if not isinstance(request, AllocationRequest):
            self._last = Allocation(assignments=(), provenance=provenance)
            return ActionBatch()
        unassigned = list(request.tasks)
        assignments: list[Assignment] = []
        by_agent: dict[AgentId, str] = {}
        actions = []
        for asset in request.assets:
            if not unassigned:
                break
            best = min(unassigned, key=lambda task: _distance(asset.position, task.center))
            unassigned.remove(best)
            assignments.append(Assignment(agent_id=asset.agent_id, task_id=best.task_id))
            by_agent[asset.agent_id] = best.task_id
            actions.append(
                Action(
                    agent_id=asset.agent_id,
                    kind=ActionKind.TASK,
                    task=TaskDirective(
                        task_kind=TaskKind.PROSPECT, prospect=ProspectTask(region=best.region)
                    ),
                )
            )
        self._last = Allocation(
            assignments=tuple(assignments), provenance=provenance, incumbent=True, by_agent=by_agent
        )
        return ActionBatch(actions=actions)

    def allocation(self) -> Allocation | None:
        return self._last


def _distance(a: Vec3, b: Vec3) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _manifest(filename: str) -> PluginManifest:
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("manifests", filename)
        .read_text(encoding="utf-8")
    )
    return load_manifest(text).manifest


def _deadline(params: Mapping[str, Any]) -> float | None:
    return float(params["deadline_s"]) if "deadline_s" in params else None


def greedy_allocator_plugin() -> TierPlugin:
    """Provider for the reference allocator (entry point) — the adapter wrapping the greedy
    stand-in solver, exactly as the Guard/Allocate shims wrap the real ones."""
    return TierPlugin(
        manifest=_manifest("greedy_allocator.yaml"),
        factory=lambda params: AllocationAdapter(
            GreedyReferenceAllocator(), deadline_s=_deadline(params)
        ),
    )
