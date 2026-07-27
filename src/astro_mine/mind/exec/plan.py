"""Plan behavior over the Core-owned plan schema (RFC-0006; RM-P1-MIND-06).

RFC-0006 ratified ``Plan``/``ContingentPlan`` as **Core-owned message schemas**
(:mod:`astro_mine.core.plan`, canonical ``plan.schema.json``), superseding the Mind-local
dataclasses ``astro_mine.mind.plan`` shipped as the pre-migration realization. Core carries the
*vocabulary* and nothing more — "schema only; planning (Mind) and evaluation (Bench/Ops) live
above Core" — so the small amount of **behavior** those dataclasses carried as methods lives
here, re-expressed as free functions over the Core models:

- :func:`is_valid` / :func:`is_stale` — the act-while-fresh test and its negation. An agent MAY
  keep acting on a stale plan under comms loss (act-while-stale, principle 5); the decision trace
  records that it did.
- :func:`branch_for` — the contingency branch a fired trigger resolves through.
- :func:`build_contingent_plan` — the validity-horizoned :class:`ContingentPlan` a tier issues
  each replan (assumptions + ``comms_lost``/``plan_expired`` branches), the delay-tolerant
  artifact :mod:`~astro_mine.mind.exec.degrade` hands the swarm.
- :func:`plan_document` — the plan wrapped in Core's versioned ``PlanDocument`` envelope, the
  form that round-trips through Core's canonical JSON Schema (``load_plan``/``validate_plan``).

Pure and deterministic — no clock and no RNG of its own; the caller supplies ``now_s`` — so a
seeded run reproduces exactly.
"""

from __future__ import annotations

from typing import Literal

from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.plan import (
    Assumption,
    ContingencyBranch,
    ContingentPlan,
    Plan,
    PlanDocument,
    PlanValidity,
)
from astro_mine.mind.compose.graph import TierNode

__all__ = [
    "PLAN_VERSION",
    "branch_for",
    "build_contingent_plan",
    "expires_at_s",
    "is_stale",
    "is_valid",
    "plan_document",
]

#: The plan-schema minor Mind emits. Core types ``PlanDocument.plan_version`` as a closed
#: ``Literal``, so this is re-declared (rather than re-exported from Core, whose ``PLAN_VERSION``
#: widens to ``str``) to keep the envelope statically checked. A test pins it to Core's constant,
#: so a Core schema bump fails loudly here rather than silently emitting a stale envelope.
PLAN_VERSION: Literal["0.1"] = "0.1"


def expires_at_s(validity: PlanValidity) -> float | None:
    """When the plan expires, or ``None`` for a standing plan (no horizon)."""
    if validity.horizon_s is None:
        return None
    return validity.issued_at_s + validity.horizon_s


def is_valid(validity: PlanValidity, now_s: float) -> bool:
    """Whether the plan is still fresh at ``now_s`` — the act-while-fresh test."""
    expiry = expires_at_s(validity)
    return expiry is None or now_s < expiry


def is_stale(validity: PlanValidity, now_s: float) -> bool:
    """Whether the plan's horizon has elapsed at ``now_s`` (the negation of :func:`is_valid`)."""
    return not is_valid(validity, now_s)


def branch_for(plan: ContingentPlan, trigger: str) -> ContingencyBranch | None:
    """The branch handling ``trigger``, or ``None`` if ``plan`` declares none."""
    return next((branch for branch in plan.branches if branch.trigger == trigger), None)


def build_contingent_plan(
    node: TierNode, batch: ActionBatch, sim_time_s: float, *, comms_denied: bool
) -> ContingentPlan:
    """The :class:`ContingentPlan` ``node`` issues for ``batch`` at ``sim_time_s``.

    The plan is valid for the tier's ``validity_horizon_s`` (``None`` ⇒ standing), rests on the
    ``earth_contact`` assumption whose last known truth is the current comms state, and declares
    the two degrade-not-collapse branches: ``comms_lost`` → hold the cached intent,
    ``plan_expired`` → reconcile once the horizon lapses and comms is back.
    """
    plan = Plan(
        plan_id=f"{node.role.value}@{sim_time_s}",
        tier=node.role.value,
        validity=PlanValidity(issued_at_s=sim_time_s, horizon_s=node.validity_horizon_s),
        actions=batch,
        assumptions=[
            Assumption(
                key="earth_contact",
                description="ground link to the mission tier",
                holds=not comms_denied,
            )
        ],
    )
    branches = [
        ContingencyBranch(
            trigger="comms_lost",
            action="hold_cached",
            description="act on cached intent while comms is denied",
        ),
        ContingencyBranch(
            trigger="plan_expired",
            action="reconcile",
            description="replan when the horizon expires and comms is up",
        ),
    ]
    return ContingentPlan(base=plan, branches=branches)


def plan_document(plan: ContingentPlan) -> PlanDocument:
    """Wrap ``plan`` in Core's versioned :class:`PlanDocument` envelope — the serialized form
    Ops replays, View renders, and Bench scores, validated by Core's canonical JSON Schema."""
    return PlanDocument(plan_version=PLAN_VERSION, plan=plan)
