# SPDX-License-Identifier: Apache-2.0
"""PDDL problem generation from a belief view (RM-P1-MIND-03).

The mission tier's symbolic-planning input: a PDDL2.1 *problem* generated per replan from the
belief view (mind.md §5, "PDDL problem files generated per-replan from the belief"), over a
fixed prospecting *domain*. This module is the pure, deterministic text generator — it names
the objects (agents, regions), the initial facts, and the goal — shared by the reference
symbolic planner and any drop-in ``unified-planning`` engine (Fast Downward / OPTIC / ENHSP)
that consumes the same problem. Keeping generation separate from solving is the PDDLStream-style
seam: the reference planner "solves" it deterministically; a native engine parses the identical
text.

The output is canonical (sorted objects, fixed clause order), so a given belief yields
byte-identical PDDL — its content hash is part of the plan provenance (RM-P1-MIND-07).

**The generated problem is genuinely solvable.** The domain models the *assignment decision*
itself: an ``assign`` action binds a free agent to an unassigned region, and ``prospect`` then
requires that binding. So a real engine (Fast Downward / OPTIC / ENHSP, via the ``[native]``
:mod:`~astro_mine.mind.mission.planner.native` adapter) *derives* the agent→region decomposition
from the goal "every region prospected" rather than being handed one — which is what makes the
PDDL backend a real mission planner and not a rubber stamp on an assignment Mind already made.
The pure-Python reference planner solves the same problem deterministically by index.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PddlProblem", "prospecting_domain", "region_name", "render_problem"]

#: The fixed prospecting domain. ``assign`` is the decision the engine makes (a free agent takes
#: an unassigned region); ``prospect`` consumes that binding to achieve the goal.
_DOMAIN = """\
(define (domain lunar-prospecting)
  (:requirements :strips :typing)
  (:types agent region)
  (:predicates (free ?a - agent) (unassigned ?r - region)
    (assigned ?a - agent ?r - region) (prospected ?r - region))
  (:action assign
    :parameters (?a - agent ?r - region)
    :precondition (and (free ?a) (unassigned ?r))
    :effect (and (assigned ?a ?r) (not (free ?a)) (not (unassigned ?r))))
  (:action prospect
    :parameters (?a - agent ?r - region)
    :precondition (assigned ?a ?r)
    :effect (prospected ?r)))
"""


def region_name(index: int) -> str:
    """The canonical PDDL object name for the ``index``-th prospect region.

    The single source of the region vocabulary: the reference planner, the native
    ``unified-planning`` adapter, and the geometry that resolves a region to a
    :class:`~astro_mine.core.messages.model.Volume` all address regions through this name, so
    both backends decompose over the *same* regions.
    """
    return f"r{index}"


@dataclass(frozen=True, slots=True)
class PddlProblem:
    """A generated prospecting problem: the ``agents`` and ``regions`` in play and the goal
    (every region prospected). ``name`` identifies the replan."""

    name: str
    agents: tuple[str, ...]
    regions: tuple[str, ...]


def prospecting_domain() -> str:
    """The fixed PDDL prospecting domain text."""
    return _DOMAIN


def render_problem(problem: PddlProblem) -> str:
    """Render ``problem`` to canonical PDDL problem text (deterministic given the inputs)."""
    agents = sorted(problem.agents)
    regions = sorted(problem.regions)
    init_terms = [f"(free {a})" for a in agents] + [f"(unassigned {r})" for r in regions]
    init = " ".join(init_terms)
    goal_terms = " ".join(f"(prospected {r})" for r in regions)
    goal = f"(and {goal_terms})" if len(regions) != 1 else goal_terms
    return (
        f"(define (problem {problem.name})\n"
        f"  (:domain lunar-prospecting)\n"
        f"  (:objects {' '.join(agents)} - agent {' '.join(regions)} - region)\n"
        f"  (:init {init})\n"
        f"  (:goal {goal}))\n"
    )
