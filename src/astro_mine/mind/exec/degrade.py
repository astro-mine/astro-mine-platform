"""Comms-aware degrade-not-collapse strategy (RM-P1-MIND-06).

The execution strategy for the decentralized / hybrid coordination posture: it makes
degrade-not-collapse (principle 4) a runtime property, not a best-effort nicety. Over the direct
composition it adds three behaviours keyed on the belief's comms state (Link LOS/earth-contact,
via the Environment API — never a sibling import):

- **act-while-stale.** When comms to the mission tier is denied, its ``plan_expired`` replan is
  *suppressed* — agents keep acting on the cached mission intent through the blackout rather than
  collapsing to undefined behaviour (a ``comms_stale_hold`` trace note). The reactive tiers
  (TAMP/control) keep running locally.
- **reconcile-on-recovery.** The tick comms is restored, the mission tier is forced to replan
  (``comms_recovered``) so the swarm re-syncs with a fresh global plan.
- **decentralized coordination.** During a blackout the ``coord/`` gossip coordinator resolves
  conflicting task claims among neighbours (lowest agent id keeps the claim; the rest yield and
  hold), so a comms-denied swarm stays coherent without a central view.

Each mission replan issues a validity-horizoned :class:`~astro_mine.core.plan.model.ContingentPlan`
(assumptions + comms_lost/plan_expired branches) — the delay-tolerant artifact agents act on, and
since RFC-0006 a **Core-owned** message schema (:mod:`astro_mine.core.plan`) rather than a
Mind-local type, so Ops replays, View renders, and Bench scores the same plan vocabulary Mind
composes. The behavior over that schema (issue / expiry / branch lookup) lives in
:mod:`astro_mine.mind.exec.plan`. The result is a *defined* safe-productive (act-on-cached) or
safe-idle state under injected comms loss, never undefined behaviour. Deterministic given the seed
and the (fixed) blackout schedule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.plan import ContingentPlan
from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.belief.view import BELIEF_EXTRAS_KEY, BeliefView
from astro_mine.mind.compose.graph import HierarchyGraph, TierNode
from astro_mine.mind.coord.model import GossipCoordinator, Intent
from astro_mine.mind.exec.plan import build_contingent_plan
from astro_mine.mind.exec.strategy import CompositionStrategy, _TierState
from astro_mine.mind.spec.enums import ReplanTriggerKind, TierRole
from astro_mine.mind.trace.model import TierDecisionRecord

__all__ = ["DecentralizedStrategy"]


class DecentralizedStrategy(CompositionStrategy):
    """Direct composition + comms-aware act-while-stale, coord, and reconcile-on-recovery."""

    def __init__(self, graph: HierarchyGraph) -> None:
        super().__init__(graph)
        self._coordinator = GossipCoordinator()
        self._was_denied = False
        self._contingent: dict[TierRole, ContingentPlan] = {}

    def decide(
        self,
        observations: Mapping[AgentId, Observation],
        context: DecisionContext,
        *,
        tick: int,
        sim_time_s: float,
    ) -> tuple[ActionBatch, tuple[TierDecisionRecord, ...]]:
        comms_denied = _comms_denied(context.extras.get(BELIEF_EXTRAS_KEY))
        recovered = self._was_denied and not comms_denied
        just_lost = comms_denied and not self._was_denied

        upstream = context.upstream if context.upstream is not None else ActionBatch()
        records: list[TierDecisionRecord] = []
        for node in self._graph.tiers:
            state = self._state[node.role]
            trigger, note = self._decide_tier(
                node,
                state,
                tick=tick,
                sim_time_s=sim_time_s,
                comms_denied=comms_denied,
                recovered=recovered,
                just_lost=just_lost,
            )
            if trigger is not None:
                batch, fallback_used = self._invoke(
                    node, state, observations, replace(context, upstream=upstream)
                )
                state.cached = batch
                state.decided_at_s = sim_time_s
                state.decided_tick = tick
                state.last_was_fallback = fallback_used
                if node.role is TierRole.MISSION:
                    self._contingent[node.role] = build_contingent_plan(
                        node, batch, sim_time_s, comms_denied=comms_denied
                    )
                records.append(
                    TierDecisionRecord(
                        role=node.role.value,
                        replanned=True,
                        trigger=trigger,
                        fallback_used=fallback_used,
                        note=note,
                    )
                )
            else:
                assert state.cached is not None
                batch = state.cached
                records.append(
                    TierDecisionRecord(
                        role=node.role.value,
                        replanned=False,
                        trigger=None,
                        fallback_used=False,
                        note=note,
                    )
                )
            upstream = batch

        if comms_denied:
            upstream, records = self._coordinate(upstream, records)
        self._was_denied = comms_denied
        return upstream, tuple(records)

    def _decide_tier(
        self,
        node: TierNode,
        state: _TierState,
        *,
        tick: int,
        sim_time_s: float,
        comms_denied: bool,
        recovered: bool,
        just_lost: bool,
    ) -> tuple[str | None, str | None]:
        """The (trigger, note) for one tier, comms-aware. ``trigger`` None ⇒ hold the cache."""
        if state.cached is None:
            return "initial", None
        if node.role is TierRole.MISSION and recovered:
            return "comms_recovered", "comms_recovered"
        if just_lost and any(t.kind is ReplanTriggerKind.COMMS_LOST for t in node.replan_triggers):
            return ReplanTriggerKind.COMMS_LOST.value, None
        base = self._due(node, state, tick=tick, sim_time_s=sim_time_s)
        if base is None:
            return None, None
        if (
            base == ReplanTriggerKind.PLAN_EXPIRED.value
            and comms_denied
            and node.role is TierRole.MISSION
        ):
            # Suppress the mission replan: act on cached intent through the blackout.
            return None, "comms_stale_hold"
        return base, None

    def _coordinate(
        self, batch: ActionBatch, records: list[TierDecisionRecord]
    ) -> tuple[ActionBatch, list[TierDecisionRecord]]:
        """Resolve conflicting task claims among neighbours; yielded agents hold this tick."""
        mission_state = self._state.get(TierRole.MISSION)
        if mission_state is None or mission_state.cached is None:
            return batch, records
        intents: list[Intent] = []
        for action in mission_state.cached.actions:
            task = action.task
            if task is not None and task.prospect is not None:
                center = task.prospect.region.center_m
                intents.append(
                    Intent(agent_id=action.agent_id, task_id=f"{center.x},{center.y},{center.z}")
                )
        resolution = self._coordinator.resolve(intents)
        if not resolution.yielded:
            return batch, records
        yielded = set(resolution.yielded)
        kept = ActionBatch(actions=[a for a in batch.actions if a.agent_id not in yielded])
        annotated = [
            replace(rec, note="coord_yield") if rec.role == TierRole.CONTROL.value else rec
            for rec in records
        ]
        return kept, annotated

    def contingent_plan(self, role: TierRole) -> ContingentPlan | None:
        """The most recent :class:`ContingentPlan` issued for ``role`` (for tests/provenance)."""
        return self._contingent.get(role)


def _comms_denied(belief: object) -> bool:
    if not isinstance(belief, BeliefView):
        return False
    return bool(belief.agents) and not any(belief.earth_contact(agent) for agent in belief.agents)
