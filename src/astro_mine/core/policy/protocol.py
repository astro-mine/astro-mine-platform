# SPDX-License-Identifier: Apache-2.0
"""Policy/Planner API v0.1 — the contract (RM-P0-CORE-03).

One uniform "observations + context -> actions/assignments" contract: a policy maps the
per-agent :class:`~astro_mine.core.messages.Observation`\\ s the Environment API yields
to an :class:`~astro_mine.core.messages.ActionBatch` the Environment API consumes —
``env.step(policy.decide(observations, context))`` closes the loop. The output spans
both *control* (Action ``ACTUATOR``/``MODE``) and *assignment* (Action ``TASK`` carrying
a :class:`~astro_mine.core.messages.TaskDirective`) in one type, so controllers and
allocators speak the same contract.

The four tiers (charter §5.4; mind.md, allocate.md) are **sub-interfaces of this one
contract** — nominal markers that document a tier's role and what ``Action.kind`` it
emits; they share ``decide`` so any layer composes with the others without rewriting
them (:mod:`astro_mine.core.policy.compose`). Determinism, not statelessness, is the
contract: given the same observations and ``context`` (incl. ``seed``) a policy is
reproducible; the runtime owns the clock/RNG (sim.md; bench determinism gates).

Core owns the *contract*; the implementations live in Phase-1 siblings — Mind (the
hierarchy), Allocate (the solver), Learn (learned policies), Guard (the wrapping shield).
Guard needs no special hook: a shield is itself a :class:`Policy` that wraps another.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.model import DecisionContext

__all__ = ["Allocator", "Controller", "MissionPlanner", "Policy", "TaskMotionPlanner"]


@runtime_checkable
class Policy(Protocol):
    """Maps per-agent observations + context to an action/assignment batch.

    The single uniform decision contract every autonomy layer implements. Multi-agent by
    construction (the single-agent case is one entry); a swarm-level planner/allocator
    decides for many agents at once, a per-agent controller for one.
    """

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        """Compute the actions/assignments for this decision step."""
        ...


class MissionPlanner(Policy, Protocol):
    """Strategic tier: assigns roles/regions to agents and groups, emitted as high-level
    ``TASK`` assignments. Delegates the combinatorial assignment to an :class:`Allocator`."""


class TaskMotionPlanner(Policy, Protocol):
    """Tactical tier (TAMP): turns an assigned role/task into a concrete sequence of
    parameterized, motion-feasible ``TASK``/``ACTUATOR`` actions for one agent."""


class Allocator(Policy, Protocol):
    """Combinatorial tier: decides who does what, when, and where under coupled
    constraints, emitted as scheduled ``TASK`` assignments (allocate.md)."""


class Controller(Policy, Protocol):
    """Reactive tier: closes the loop, emitting low-level ``ACTUATOR``/``MODE`` setpoints
    (optionally consuming an upstream tier's assignments via ``context.upstream``)."""
