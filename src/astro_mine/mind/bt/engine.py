# SPDX-License-Identifier: Apache-2.0
"""The behavior-tree tick engine (RM-P1-MIND-02).

Evaluates a :class:`~astro_mine.mind.bt.model.BehaviorTree` once per executive tick, reactively
(re-ticked from the root each tick — no cross-tick node memory, so a seeded run reproduces).
It threads a running proposal :class:`ActionBatch` through the tree exactly as the direct
composition strategy threads ``upstream``: a planner/policy leaf decides against the current
proposal and replaces it, a primitive leaf emits an SADF action, a condition gates a branch on
the belief. The engine's product is the tick's *proposed* (pre-shield) batch — the executive
still routes it through the mandatory shield, so Guard-wrapped output stays the only output.

Degrade-not-collapse is structural here (principle 4): a leaf that raises is caught and becomes
``failure`` — never a crash — so a ``fallback`` composite moves to its next branch (e.g. a
safe-idle primitive), and a ``fresh_upstream`` condition fails when the tier above produced no
fresh input, routing to the same degradation branch. The engine records each tier a
planner/policy leaf invokes as a :class:`TierDecisionRecord` (uniform with the composition
strategy, so the two execution kinds yield comparable traces), marking ``fallback_used`` when
the leaf was reached through a fallback branch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import Action, ActionBatch, Observation, TaskDirective
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.policy.protocol import Policy
from astro_mine.mind.bt.model import (
    ActionNode,
    BehaviorTree,
    BTNode,
    ConditionKind,
    ConditionNode,
    ControlKind,
    ControlNode,
    DecoratorKind,
    DecoratorNode,
    InvokeKind,
    NodeStatus,
)
from astro_mine.mind.trace.model import TierDecisionRecord

__all__ = ["BehaviorTreeEngine", "PrimitiveError"]

#: BT execution replans reactively every tick; the trigger label recorded for each leaf.
_BT_TRIGGER = "bt_tick"


class PrimitiveError(Exception):
    """Raised when a primitive leaf names an SADF action the reference engine cannot emit."""


@dataclass
class _TickState:
    observations: Mapping[AgentId, Observation]
    context: DecisionContext
    proposal: ActionBatch
    records: list[TierDecisionRecord] = field(default_factory=list)


class BehaviorTreeEngine:
    """Ticks a :class:`BehaviorTree` over a fixed set of tier policies (role → ``Policy``).

    Stateless across ticks; construct once and reuse. ``tiers`` maps the tier role each
    planner/policy leaf references to the composed policy that fills it.
    """

    def __init__(self, tree: BehaviorTree, tiers: Mapping[str, Policy]) -> None:
        self._tree = tree
        self._tiers = dict(tiers)

    def tick(
        self,
        observations: Mapping[AgentId, Observation],
        context: DecisionContext,
    ) -> tuple[ActionBatch, tuple[TierDecisionRecord, ...]]:
        """Evaluate the tree once; return the proposed batch and the per-tier records."""
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        state = _TickState(observations=observations, context=context, proposal=upstream)
        self._eval(self._tree.root, state, under_fallback=False)
        return state.proposal, tuple(state.records)

    def _eval(self, node: BTNode, state: _TickState, *, under_fallback: bool) -> NodeStatus:
        if isinstance(node, ControlNode):
            return self._eval_control(node, state, under_fallback=under_fallback)
        if isinstance(node, DecoratorNode):
            return self._eval_decorator(node, state, under_fallback=under_fallback)
        if isinstance(node, ActionNode):
            return self._eval_action(node, state, under_fallback=under_fallback)
        return self._eval_condition(node, state)

    def _eval_control(
        self, node: ControlNode, state: _TickState, *, under_fallback: bool
    ) -> NodeStatus:
        if node.kind is ControlKind.SEQUENCE:
            for child in node.children:
                status = self._eval(child, state, under_fallback=under_fallback)
                if status is not NodeStatus.SUCCESS:
                    return status
            return NodeStatus.SUCCESS
        # FALLBACK (selector): the first non-failing child wins; children after the first are
        # the reachable degradation branches (reached only because an earlier one failed).
        for index, child in enumerate(node.children):
            status = self._eval(child, state, under_fallback=under_fallback or index > 0)
            if status is not NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE

    def _eval_decorator(
        self, node: DecoratorNode, state: _TickState, *, under_fallback: bool
    ) -> NodeStatus:
        status = self._eval(node.child, state, under_fallback=under_fallback)
        if status is NodeStatus.RUNNING:
            return status
        succeeded = status is NodeStatus.SUCCESS
        if node.kind is DecoratorKind.INVERTER:
            succeeded = not succeeded
        elif node.kind is DecoratorKind.FORCE_SUCCESS:
            succeeded = True
        else:  # FORCE_FAILURE
            succeeded = False
        return NodeStatus.SUCCESS if succeeded else NodeStatus.FAILURE

    def _eval_action(
        self, node: ActionNode, state: _TickState, *, under_fallback: bool
    ) -> NodeStatus:
        if node.invoke is InvokeKind.PRIMITIVE:
            state.proposal = self._primitive(node, state)
            state.records.append(
                TierDecisionRecord(
                    role=f"primitive.{node.ref}",
                    replanned=True,
                    trigger=_BT_TRIGGER,
                    fallback_used=under_fallback,
                )
            )
            return NodeStatus.SUCCESS
        policy = self._tiers.get(node.ref)
        if policy is None:  # unresolved ref; composer validates, so this is defensive only
            return NodeStatus.FAILURE
        try:
            batch = policy.decide(
                state.observations, replace(state.context, upstream=state.proposal)
            )
        except Exception:
            # A tier fault must never crash the executive (principle 4); it degrades to a
            # failure so a fallback branch can take over.
            state.records.append(
                TierDecisionRecord(
                    role=node.ref, replanned=True, trigger=_BT_TRIGGER, fallback_used=True
                )
            )
            return NodeStatus.FAILURE
        state.proposal = batch
        state.records.append(
            TierDecisionRecord(
                role=node.ref, replanned=True, trigger=_BT_TRIGGER, fallback_used=under_fallback
            )
        )
        return NodeStatus.SUCCESS

    def _eval_condition(self, node: ConditionNode, state: _TickState) -> NodeStatus:
        if node.check is ConditionKind.FRESH_UPSTREAM:
            fresh = bool(state.proposal.actions)
            return NodeStatus.SUCCESS if fresh else NodeStatus.FAILURE
        return NodeStatus.FAILURE  # pragma: no cover - closed vocabulary, unreachable

    def _primitive(self, node: ActionNode, state: _TickState) -> ActionBatch:
        if node.ref == "standby":
            return ActionBatch(
                actions=[
                    Action(
                        agent_id=agent_id,
                        kind=ActionKind.TASK,
                        task=TaskDirective(task_kind=TaskKind.STANDBY),
                    )
                    for agent_id in sorted(state.observations)
                ]
            )
        raise PrimitiveError(f"primitive leaf {node.node_id!r}: unknown SADF action {node.ref!r}")
