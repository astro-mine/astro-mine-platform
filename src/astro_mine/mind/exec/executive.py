# SPDX-License-Identifier: Apache-2.0
"""The executive — the event-driven loop that ticks a composed hierarchy (RM-P1-MIND-01).

The runtime that steps a :class:`HierarchyGraph` against a Core :class:`Environment`. On
each tick it (1) refreshes the :class:`BeliefView` from the observation and publishes it to
the tiers via ``DecisionContext.extras``; (2) asks the execution strategy for the proposed
action (the strategy owns per-tier validity/replan/fallback); (3) routes that proposal
through the **mandatory shield** — the single egress; (4) records the tick; (5) steps the
environment with the *shielded* batch.

The safety guarantee is structural: **the executive is the only holder of the ``Environment``**.
Tiers and the shield are pure ``decide(obs, ctx)`` policies handed observations and context —
never the env — so no plugin can emit an action directly. The tick itself (belief → strategy →
:func:`~astro_mine.mind.guardrail.shield.shield_egress`) lives in
:func:`~astro_mine.mind.exec._tick.decide_tick`, shared with
:class:`~astro_mine.mind.exec.policy.StackPolicy` so that the shielded path from a proposed batch
to an emitted one exists in exactly one place. RM-P1-MIND-05 verifies it with an adversarial plugin.

The runtime owns the clock (``sim_time_s`` from the env) and the ``seed`` it threads into
every ``DecisionContext`` — so a run is deterministic given a seed, a pinned plugin set, and
a fixed environment (principle 9), which the golden-trace gate checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from astro_mine.core.env.model import AgentId
from astro_mine.core.env.protocol import Environment
from astro_mine.core.messages.model import Observation
from astro_mine.core.objective.model import ObjectiveSpec
from astro_mine.mind.bt.strategy import BehaviorTreeStrategy
from astro_mine.mind.compose.graph import HierarchyGraph
from astro_mine.mind.exec._tick import decide_tick
from astro_mine.mind.exec.degrade import DecentralizedStrategy
from astro_mine.mind.exec.strategy import CompositionStrategy, Strategy
from astro_mine.mind.spec.enums import CoordinationKind, ExecutionKind
from astro_mine.mind.trace.model import DecisionTrace, TickRecord

__all__ = ["Executive", "RunResult", "build_strategy"]


def build_strategy(graph: HierarchyGraph) -> Strategy:
    """The execution strategy the stack selects — a behavior tree (``execution.kind``), the
    comms-aware degrade-not-collapse strategy (decentralized/hybrid ``coordination.kind``,
    RM-P1-MIND-06), or direct composition. A fresh, per-run instance (strategies cache tier
    state across a run)."""
    if graph.execution.kind is ExecutionKind.BEHAVIOR_TREE:
        return BehaviorTreeStrategy(graph)
    if graph.coordination.kind in (CoordinationKind.DECENTRALIZED, CoordinationKind.HYBRID):
        return DecentralizedStrategy(graph)
    return CompositionStrategy(graph)


@dataclass(frozen=True)
class RunResult:
    """The outcome of an executive run: the full :class:`DecisionTrace`, the last
    observations, how many ticks ran, and whether the episode ended by termination or
    truncation (vs hitting ``max_ticks``)."""

    trace: DecisionTrace
    final_observations: Mapping[AgentId, Observation]
    ticks_run: int
    terminated: bool
    truncated: bool


class Executive:
    """Ticks a composed hierarchy against an environment, emitting a decision trace.

    Stateful across a run (the strategy caches tier decisions); construct one per run, or
    call :meth:`run`, which builds a fresh strategy each time.
    """

    def __init__(self, graph: HierarchyGraph) -> None:
        self._graph = graph

    def run(
        self,
        env: Environment,
        *,
        max_ticks: int,
        seed: int | None = None,
        objective: ObjectiveSpec | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Reset ``env`` and tick the hierarchy up to ``max_ticks``, stopping early when
        every active agent has terminated/truncated. Returns the run's :class:`RunResult`."""
        strategy = build_strategy(self._graph)
        reset = env.reset(seed=seed, options=options)
        observations = reset.observations
        sim_time_s = 0.0

        records: list[TickRecord] = []
        terminated = False
        truncated = False
        for tick in range(max_ticks):
            batch, record = decide_tick(
                self._graph,
                strategy,
                observations,
                tick=tick,
                sim_time_s=sim_time_s,
                seed=seed,
                objective=objective,
            )
            records.append(record)

            result = env.step(batch)
            observations = result.observations
            sim_time_s = result.sim_time_s

            active = tuple(observations.keys())
            if not active:
                break
            if all(result.terminations.get(agent, False) for agent in active):
                terminated = True
                break
            if all(result.truncations.get(agent, False) for agent in active):
                truncated = True
                break

        trace = DecisionTrace(
            stack_id=self._graph.stack_id,
            provenance=self._graph.provenance,
            ticks=tuple(records),
        )
        return RunResult(
            trace=trace,
            final_observations=observations,
            ticks_run=len(records),
            terminated=terminated,
            truncated=truncated,
        )
