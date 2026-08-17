# SPDX-License-Identifier: Apache-2.0
"""A composed Mind stack, presented as a plain Core :class:`~astro_mine.core.policy.Policy`.

The :class:`~astro_mine.mind.exec.Executive` drives the loop itself: it holds the
``Environment``, calls ``reset``/``step``, and owns the clock. That is the right shape when Mind
is in charge — and it is what makes the shield the structural single egress.

But the runtimes Mind must *integrate* with invert that control. Bench's ``EpisodeRunner`` seam is
``runner(resolved, policy, seed) -> EpisodeTrace``, and Sim's ``run_episode``/``record_episode``
own the stepping loop and call ``policy.decide(observations, context)`` per tick. Neither will
hand Mind its ``Environment``. So a composed stack cannot be scored on Bench, or stepped through
Sim's own recorded episode, unless it can also present itself as an ordinary Core ``Policy``.

:class:`StackPolicy` is that adapter, and it gives up nothing:

- **The shield is still the only egress.** ``decide`` returns whatever
  :func:`~astro_mine.mind.exec._tick.decide_tick` emitted — the same shielded path the Executive
  uses, not a re-implementation (see that module).
- **The decision trace still accrues.** Every tick is recorded, so a Bench- or Sim-driven run still
  yields a :class:`~astro_mine.mind.trace.model.DecisionTrace` — canonical-JSON hashable and
  MCAP-serializable (RM-P1-MIND-07) — even though Mind never saw the environment.
- **The degrade path still fires.** ``decide_tick`` assembles the ``BeliefView`` from
  ``Observation.comms``, so a real Sim ContactPlan blackout drives act-while-stale and
  reconcile-on-recovery exactly as a toy env's synthetic mask does (RM-P1-MIND-06).

**Episode boundaries.** Execution strategies are stateful across a run — they cache each tier's
decision, and the degrade strategy tracks the comms edge. Bench reuses *one* policy object across
every seed it scores, and nothing in the Core ``Policy`` contract announces "a new episode starts
now". So this class detects the boundary itself: a tick index that does not advance past the last
one (a rewind, canonically back to 0) begins a fresh episode, rebuilding the strategy and clearing
the trace. Without that, seed *n+1* would inherit seed *n*'s cached plans and the score would
depend on evaluation order — silently, and only across seeds. :meth:`reset` forces the same thing
explicitly for a caller that would rather not rely on the detection.
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.objective.model import ObjectiveSpec
from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.compose.graph import HierarchyGraph
from astro_mine.mind.exec._tick import decide_tick
from astro_mine.mind.exec.executive import build_strategy
from astro_mine.mind.trace.model import DecisionTrace, TickRecord

__all__ = ["StackPolicy"]


class StackPolicy:
    """A composed :class:`HierarchyGraph` behind the Core ``Policy`` contract.

    ``seed`` and ``objective`` are the fallbacks used when the caller's ``DecisionContext`` does
    not carry them (Sim and Bench both pass a seed; a bare harness may not).
    """

    def __init__(
        self,
        graph: HierarchyGraph,
        *,
        seed: int | None = None,
        objective: ObjectiveSpec | None = None,
    ) -> None:
        self._graph = graph
        self._seed = seed
        self._objective = objective
        self._records: list[TickRecord] = []
        self._strategy = build_strategy(graph)
        self._last_tick: int | None = None

    @property
    def graph(self) -> HierarchyGraph:
        """The composed stack this policy runs."""
        return self._graph

    @property
    def trace(self) -> DecisionTrace:
        """The decision trace accrued for the current episode (RM-P1-MIND-07)."""
        return DecisionTrace(
            stack_id=self._graph.stack_id,
            provenance=self._graph.provenance,
            ticks=tuple(self._records),
        )

    def reset(self) -> None:
        """Begin a fresh episode: rebuild the strategy and clear the trace."""
        self._strategy = build_strategy(self._graph)
        self._records = []
        self._last_tick = None

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        """One shielded tick of the composed stack, for a runtime that owns the loop."""
        if not observations:
            # Every agent has terminated; there is nothing to decide and no tick to record.
            return ActionBatch()

        tick = self._tick_index(observations)
        if self._last_tick is not None and tick <= self._last_tick:
            self.reset()  # the tick rewound: a new episode began under the same policy object
        self._last_tick = tick

        emitted, record = decide_tick(
            self._graph,
            self._strategy,
            observations,
            tick=tick,
            sim_time_s=observations[min(observations)].sim_time_s,
            seed=context.seed if context.seed is not None else self._seed,
            objective=context.objective if context.objective is not None else self._objective,
        )
        self._records.append(record)
        return emitted

    def _tick_index(self, observations: Mapping[AgentId, Observation]) -> int:
        """The tick the environment says it is on.

        The Executive counts ticks because it drives the loop; here the **environment is the
        clock** — it stamps a ``tick`` and a ``sim_time_s`` on every observation, and this class
        reads both from there rather than from ``DecisionContext``. Not a stylistic choice:
        ``DecisionContext.sim_time_s`` is a plain ``float`` defaulting to ``0.0``, so a caller that
        never sets it is indistinguishable from one at *t=0*, and trusting it would silently pin
        every tick of such a run to time zero. Agents step in lockstep, so any observation answers;
        take the lowest id for a deterministic pick.
        """
        return observations[min(observations)].tick
