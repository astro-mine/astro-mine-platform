"""The composition execution strategy (RM-P1-MIND-01).

How the executive turns the hierarchy into an action each tick under the default
``composition`` execution kind: run the due tiers in canonical order (mission → tamp →
control), threading each tier's :class:`ActionBatch` into the next tier's
``DecisionContext.upstream`` (Core's ``ComposedPolicy`` seam), while reusing a tier's cached
decision as long as it is valid and activating the tier's fallback when it fails. This is
the pluggable seam RM-P1-MIND-02 fills with the behavior-tree execution representation; the
executive holds a strategy and does not otherwise know how a tick is produced.

Replan policy per tier, given it already has a cached decision:

- ``plan_expired`` — the tier's ``validity_horizon_s`` has elapsed. Keeping this an explicit
  trigger (rather than always replanning on expiry) is deliberate: it separates the
  *staleness* fact (the horizon) from the *replan* policy, so RM-P1-MIND-06 can suppress it
  under a comms blackout and let agents act on cached intent (act-while-stale, principle 5).
- ``periodic`` — re-decide every ``every_ticks`` ticks.
- ``on_fallback`` — re-decide the tick after a fallback activated (reconcile-on-recovery).

A tier with neither a horizon nor any trigger is reactive: it decides every tick.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.compose.graph import HierarchyGraph, TierNode
from astro_mine.mind.spec.enums import ReplanTriggerKind, TierRole
from astro_mine.mind.trace.model import TierDecisionRecord

__all__ = ["CompositionStrategy", "Strategy"]


@runtime_checkable
class Strategy(Protocol):
    """How a tick is produced from the composed hierarchy — the seam the executive holds so
    it does not know whether a tick came from direct composition (:class:`CompositionStrategy`)
    or a behavior tree (:class:`~astro_mine.mind.bt.strategy.BehaviorTreeStrategy`). Returns the
    tick's proposed (pre-shield) batch plus the per-tier decision records."""

    def decide(
        self,
        observations: Mapping[AgentId, Observation],
        context: DecisionContext,
        *,
        tick: int,
        sim_time_s: float,
    ) -> tuple[ActionBatch, tuple[TierDecisionRecord, ...]]: ...


@dataclass
class _TierState:
    """Per-tier replan bookkeeping across ticks."""

    cached: ActionBatch | None = None
    decided_at_s: float | None = None
    decided_tick: int | None = None
    last_was_fallback: bool = False


class CompositionStrategy:
    """Direct-composition execution of a :class:`HierarchyGraph`. Stateful across ticks (it
    caches each tier's decision); construct one per episode/run."""

    def __init__(self, graph: HierarchyGraph) -> None:
        self._graph = graph
        self._state: dict[TierRole, _TierState] = {node.role: _TierState() for node in graph.tiers}

    def decide(
        self,
        observations: Mapping[AgentId, Observation],
        context: DecisionContext,
        *,
        tick: int,
        sim_time_s: float,
    ) -> tuple[ActionBatch, tuple[TierDecisionRecord, ...]]:
        """Produce this tick's proposed (pre-shield) batch and the per-tier records."""
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        records: list[TierDecisionRecord] = []
        for node in self._graph.tiers:
            state = self._state[node.role]
            trigger = self._due(node, state, tick=tick, sim_time_s=sim_time_s)
            if trigger is not None:
                batch, fallback_used = self._invoke(
                    node, state, observations, replace(context, upstream=upstream)
                )
                state.cached = batch
                state.decided_at_s = sim_time_s
                state.decided_tick = tick
                state.last_was_fallback = fallback_used
                records.append(
                    TierDecisionRecord(
                        role=node.role.value,
                        replanned=True,
                        trigger=trigger,
                        fallback_used=fallback_used,
                    )
                )
            else:
                assert state.cached is not None  # _due returns a trigger whenever cache is empty
                batch = state.cached
                records.append(
                    TierDecisionRecord(
                        role=node.role.value, replanned=False, trigger=None, fallback_used=False
                    )
                )
            upstream = batch
        return upstream, tuple(records)

    def _due(
        self, node: TierNode, state: _TierState, *, tick: int, sim_time_s: float
    ) -> str | None:
        """The reason the tier must re-decide this tick, or ``None`` to reuse its cache."""
        if state.cached is None:
            return "initial"
        if node.validity_horizon_s is None and not node.replan_triggers:
            return "reactive"
        for trigger in node.replan_triggers:
            if trigger.kind is ReplanTriggerKind.PLAN_EXPIRED:
                assert node.validity_horizon_s is not None  # enforced at load
                assert state.decided_at_s is not None
                if sim_time_s - state.decided_at_s >= node.validity_horizon_s:
                    return ReplanTriggerKind.PLAN_EXPIRED.value
            elif trigger.kind is ReplanTriggerKind.PERIODIC:
                assert trigger.every_ticks is not None  # enforced at load
                assert state.decided_tick is not None
                if tick - state.decided_tick >= trigger.every_ticks:
                    return ReplanTriggerKind.PERIODIC.value
            elif trigger.kind is ReplanTriggerKind.ON_FALLBACK and state.last_was_fallback:
                return ReplanTriggerKind.ON_FALLBACK.value
        return None

    def _invoke(
        self,
        node: TierNode,
        state: _TierState,
        observations: Mapping[AgentId, Observation],
        context: DecisionContext,
    ) -> tuple[ActionBatch, bool]:
        """Run the tier, degrading to its fallback → last cached decision → safe-idle empty
        batch if it raises. A plugin fault must never crash the executive (principle 4)."""
        try:
            return node.policy.decide(observations, context), False
        except Exception:
            if node.fallback is not None:
                try:
                    return node.fallback.decide(observations, context), True
                except Exception:
                    pass
            if state.cached is not None:
                return state.cached, True
            return ActionBatch(), True
