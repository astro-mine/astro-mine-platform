# SPDX-License-Identifier: Apache-2.0
"""Stack-spec closed vocabularies (Mind-owned, RM-P1-MIND-01).

The small, closed enumerations the stack spec authors against. Like Core's SADF /
registry vocabularies (conventions.md §3) they are **append-only** — a member is never
removed or repurposed, and new members grow the vocabulary by RFC / follow-on issue. In
particular the single-member ``ExecutionKind`` / ``CoordinationKind`` today reserve the
switch points that later Mind work fills in additively:

- ``ExecutionKind`` gains ``behavior_tree`` in **RM-P1-MIND-02** (the BehaviorTree.CPP
  scaffold), so a stack can select the BT execution representation over the default
  direct composition without a schema break;
- ``CoordinationKind`` gains ``decentralized`` / ``hybrid`` in **RM-P1-MIND-06** (the
  degrade-not-collapse `coord/` work), so a stack can pick the comms-regime posture.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["CoordinationKind", "ExecutionKind", "ReplanTriggerKind", "TierRole"]


class TierRole(StrEnum):
    """The tier a plugin fills in the three-tier hierarchy (mind.md §1, §3).

    ``mission`` (strategic: roles/regions), ``allocator`` (who-does-what-when-where, delegated
    to Astro-Mine-Allocate; **RM-P1-MIND-04**), ``tamp`` (tactical: task-and-motion), and
    ``control`` (reactive: closed-loop setpoints). A stack MAY collapse the hierarchy to a
    subset (principle 3) — a single end-to-end ``control`` policy is a valid one-tier stack,
    and a stack that does not delegate assignment simply omits ``allocator`` — but whichever
    roles appear are composed in the canonical :data:`TIER_ORDER`.
    """

    MISSION = "mission"
    ALLOCATOR = "allocator"
    TAMP = "tamp"
    CONTROL = "control"


#: Canonical composition order of the tiers — the mission tier decides first and threads its
#: output downstream to the allocator (who-does-what, RM-P1-MIND-04), then TAMP, then control
#: (Core ``ComposedPolicy`` ``upstream`` semantics). The composer orders a stack's tiers by
#: this regardless of authoring order; a stack MAY omit any role (principle 3), e.g. no
#: ``allocator`` tier when assignment is not delegated.
TIER_ORDER: tuple[TierRole, ...] = (
    TierRole.MISSION,
    TierRole.ALLOCATOR,
    TierRole.TAMP,
    TierRole.CONTROL,
)


class ReplanTriggerKind(StrEnum):
    """What makes the executive re-invoke a tier rather than act on its cached decision.

    ``plan_expired`` fires when the tier's decision is older than its
    ``validity_horizon_s``; ``periodic`` fires every ``every_ticks`` ticks; ``on_fallback``
    forces a fresh decision on the tick after a fallback activated (reconcile-on-recovery,
    the pre-`ContingentPlan` form of principle 5). ``comms_lost`` (**RM-P1-MIND-06**) fires a
    conservative local re-decision the tick comms transitions to denied; under a comms
    blackout the comms-aware strategy conversely *suppresses* ``plan_expired`` on a tier whose
    input source is unreachable, so agents act on cached intent (act-while-stale).
    """

    PLAN_EXPIRED = "plan_expired"
    PERIODIC = "periodic"
    ON_FALLBACK = "on_fallback"
    COMMS_LOST = "comms_lost"


class ExecutionKind(StrEnum):
    """How the executive turns the composed hierarchy into an action each tick.

    ``composition`` runs the tiers directly in canonical order, threading each tier's
    ``ActionBatch`` into the next tier's ``DecisionContext.upstream`` (Core
    ``ComposedPolicy``). ``behavior_tree`` (**RM-P1-MIND-02**) instead ticks an authored
    Groot-compatible behavior tree (named by ``ExecutionSpec.behavior_tree_ref``) that
    sequences and guards the same tier policies, with explicit selector/decorator degradation
    branches.
    """

    COMPOSITION = "composition"
    BEHAVIOR_TREE = "behavior_tree"


class CoordinationKind(StrEnum):
    """The coordination posture across agents (mind.md §11).

    ``centralized`` decides for all agents in one composed pass. ``decentralized`` engages the
    `coord/` neighbor coordination (gossip / consensus / conflict resolution) and comms-aware
    act-while-stale so agents stay coherent when the global view is stale; ``hybrid`` is the
    recommended posture (mind.md §11) — a centralized mission tier for global coherence that
    *degrades* to decentralized neighbor coordination under comms loss (**RM-P1-MIND-06**).
    """

    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HYBRID = "hybrid"
