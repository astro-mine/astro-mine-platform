# SPDX-License-Identifier: Apache-2.0
"""The composed hierarchy graph (RM-P1-MIND-01).

The validated, runnable artifact the composer produces from a stack spec and the executive
ticks: the ordered tier nodes (each an instantiated :class:`Policy` plus its validity
horizon, replan triggers, and fallback), the mandatory shield node, the execution and
coordination posture, and the decision provenance. Frozen in-memory containers — the
authored artifact is the stack spec; this is its resolved, instantiated form.
"""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.core.policy.protocol import Policy
from astro_mine.core.registry.model import PluginManifest
from astro_mine.mind.bt.model import BehaviorTree
from astro_mine.mind.spec.enums import TierRole
from astro_mine.mind.spec.model import CoordinationSpec, ExecutionSpec, ReplanTrigger
from astro_mine.mind.trace.model import DecisionProvenance

__all__ = ["HierarchyGraph", "ShieldNode", "TierNode"]


@dataclass(frozen=True)
class TierNode:
    """One instantiated tier: its ``role``, the resolved ``policy`` and the ``plugin_name``
    it came from, the ``validity_horizon_s`` and ``replan_triggers`` that govern when the
    executive re-invokes it, and the optional instantiated ``fallback`` policy (with its
    ``fallback_name``). ``manifest`` is the gated Core manifest the policy was built from."""

    role: TierRole
    plugin_name: str
    policy: Policy
    validity_horizon_s: float | None
    replan_triggers: tuple[ReplanTrigger, ...]
    fallback: Policy | None
    fallback_name: str | None
    manifest: PluginManifest


@dataclass(frozen=True)
class ShieldNode:
    """The mandatory shield: the instantiated shield ``policy``, the ``plugin_name`` it came
    from, and its gated ``manifest``. Every emitted action passes through this policy."""

    plugin_name: str
    policy: Policy
    manifest: PluginManifest


@dataclass(frozen=True)
class HierarchyGraph:
    """A runnable autonomy hierarchy: ``tiers`` in canonical composition order, the
    mandatory ``shield``, the ``execution``/``coordination`` posture, and the
    ``provenance`` (pinned plugin set + Core interface versions + seed)."""

    stack_id: str
    tiers: tuple[TierNode, ...]
    shield: ShieldNode
    execution: ExecutionSpec
    coordination: CoordinationSpec
    provenance: DecisionProvenance
    behavior_tree: BehaviorTree | None = None

    def tier(self, role: TierRole) -> TierNode | None:
        """The tier node filling ``role``, or ``None`` if the stack collapsed it."""
        return next((node for node in self.tiers if node.role is role), None)
