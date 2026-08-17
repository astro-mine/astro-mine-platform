# SPDX-License-Identifier: Apache-2.0
"""The one place a composed hierarchy turns observations into an emitted action.

Extracted so the two runtimes that drive a stack — the :class:`~astro_mine.mind.exec.Executive`
(Mind owns the loop and holds the ``Environment``) and the
:class:`~astro_mine.mind.exec.policy.StackPolicy` (someone else owns the loop and calls
``decide``) — share a *single* implementation rather than each carrying its own copy.

That matters because of what this function does last: it routes the strategy's proposed batch
through :func:`~astro_mine.mind.guardrail.shield.shield_egress`. Guard-wrapping is the mandatory
single output path (RM-P1-MIND-05, mind.md §2, §7) and it is enforced *structurally* — by there
being exactly one code path from a proposed batch to an emitted one. A second, hand-copied tick
loop would be a second place for that invariant to rot.
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.objective.model import ObjectiveSpec
from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.belief.view import BELIEF_EXTRAS_KEY, BeliefView, assemble_belief
from astro_mine.mind.compose.graph import HierarchyGraph
from astro_mine.mind.exec.strategy import Strategy
from astro_mine.mind.guardrail.shield import shield_egress
from astro_mine.mind.trace.model import TickRecord

__all__ = ["comms_denied", "decide_tick"]


def comms_denied(belief: BeliefView) -> bool:
    """Whether the tick is comms-denied — no observed agent has an Earth/DSN link.

    The input the degrade-not-collapse strategy keys on (RM-P1-MIND-06), recorded on every tick
    so a trace shows *when* the swarm was dark. Sourced from ``Observation.comms`` — the Core
    Environment API's :class:`~astro_mine.core.messages.model.CommsObservationMask` — so it is
    the environment that decides, never a sibling import: a toy env's synthetic schedule and
    Sim's ContactPlan-derived mask reach this function through the same field.
    """
    return bool(belief.agents) and not any(belief.earth_contact(agent) for agent in belief.agents)


def decide_tick(
    graph: HierarchyGraph,
    strategy: Strategy,
    observations: Mapping[AgentId, Observation],
    *,
    tick: int,
    sim_time_s: float,
    seed: int | None,
    objective: ObjectiveSpec | None,
) -> tuple[ActionBatch, TickRecord]:
    """Produce one tick: refresh the belief, ask the strategy, and shield the result.

    Returns the **shielded** batch (the only batch any caller may hand to an environment) and the
    :class:`TickRecord` for the decision trace.
    """
    belief = assemble_belief(observations, tick=tick, sim_time_s=sim_time_s)
    context = DecisionContext(
        sim_time_s=sim_time_s,
        objective=objective,
        seed=seed,
        extras={BELIEF_EXTRAS_KEY: belief},
    )
    proposed, tier_records = strategy.decide(
        observations, context, tick=tick, sim_time_s=sim_time_s
    )
    emitted, shield_record = shield_egress(
        graph.shield.policy,
        observations,
        context,
        proposed,
        shield_name=graph.shield.plugin_name,
    )
    record = TickRecord(
        tick=tick,
        sim_time_s=sim_time_s,
        seed=seed,
        tiers=tier_records,
        shield=shield_record,
        action_batch=emitted,
        comms_denied=comms_denied(belief),
    )
    return emitted, record
