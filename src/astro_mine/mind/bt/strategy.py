"""The behavior-tree execution strategy (RM-P1-MIND-02).

The ``behavior_tree`` counterpart to :class:`~astro_mine.mind.exec.strategy.CompositionStrategy`:
same interface (``decide(observations, context, *, tick, sim_time_s)`` → proposed batch +
per-tier records), so the executive holds either and does not otherwise know how a tick is
produced — the seam the direct-composition strategy's docstring reserved. Where the composition
strategy runs the tiers in fixed canonical order under per-tier validity horizons, the BT
strategy lets an authored :class:`~astro_mine.mind.bt.model.BehaviorTree` sequence and guard the
same tier policies, with explicit selector/decorator degradation branches. Reactive: the tree
is re-ticked each executive tick (the ``tick``/``sim_time_s`` arguments are accepted for
interface parity; timing lives in the tree's conditions/decorators), so a seeded run is
deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.bt.engine import BehaviorTreeEngine
from astro_mine.mind.compose.graph import HierarchyGraph
from astro_mine.mind.trace.model import TierDecisionRecord

__all__ = ["BehaviorTreeStrategy"]


class BehaviorTreeStrategy:
    """Ticks the graph's behavior tree over its composed tier policies each executive tick."""

    def __init__(self, graph: HierarchyGraph) -> None:
        if graph.behavior_tree is None:  # pragma: no cover - guarded by the composer
            raise ValueError(
                f"stack {graph.stack_id!r} selects behavior_tree execution but carries no tree"
            )
        self._engine = BehaviorTreeEngine(
            graph.behavior_tree, {node.role.value: node.policy for node in graph.tiers}
        )

    def decide(
        self,
        observations: Mapping[AgentId, Observation],
        context: DecisionContext,
        *,
        tick: int,
        sim_time_s: float,
    ) -> tuple[ActionBatch, tuple[TierDecisionRecord, ...]]:
        """Produce this tick's proposed (pre-shield) batch and the per-tier records."""
        return self._engine.tick(observations, context)
