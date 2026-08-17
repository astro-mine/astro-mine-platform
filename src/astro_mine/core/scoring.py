# SPDX-License-Identifier: Apache-2.0
"""The episode-scoring vocabulary — what a runner returns and a metric reads.

Bench runs a simulator through the ``EpisodeRunner`` seam and deliberately never imports Sim
(bench.md §2.2, "Bench composes, never a second simulator"). Sim, meanwhile, imported Bench —
because the *types* the seam is written in lived in Bench. The arrow pointed up the layer table
(conventions.md §3.2 rule 3), and §3.2 names this exact pair as the live example of a defect whose
"fix is inversion rather than permission". This module is that inversion: the vocabulary moves to
the waist, and both sides depend on it instead of on each other.

What is here is the *contract* and nothing else:

- :class:`EpisodeTrace` — one episode's scored input: an ordered stream of Core
  :class:`~astro_mine.core.messages.Observation` records plus a :class:`ScoringContext`;
- :class:`ScoringContext` and :class:`BeliefSnapshot` — the scorer-only inputs a metric may read
  and an agent may not;
- :class:`RunOutcome` — one runner execution's deterministic output for a ``(scenario, seed)``;
- :class:`ScoringRefused` — a runner's deliberate refusal to score;
- :class:`EpisodeScorer` — the seam a runner uses to score without owning a metric registry.

What is **not** here is the metric registry, the metric implementations, the scenario vocabulary,
and the scorecard. Those are Bench's, they encode opinions about what is worth measuring, and Core
holds mechanism rather than policy (core.md §2.2). A component that wants to score still needs
Bench; it just no longer needs to *import* Bench to describe what it produced.

Ground-truth isolation (prospect.md §9): the belief prior/history, the sealed ground truth, and the
PSR mask live in :class:`ScoringContext`, never in the agent-facing observation stream — a scorer
sees them, a policy never does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from astro_mine.core.messages import Observation
from astro_mine.core.resource import FieldDistribution

__all__ = [
    "BeliefSnapshot",
    "EpisodeScorer",
    "EpisodeTrace",
    "RunOutcome",
    "ScoringContext",
    "ScoringRefused",
]


@dataclass(frozen=True, slots=True)
class BeliefSnapshot:
    """The swarm's belief field at one instant: per-cell posterior distributions.

    ``cells`` maps an opaque cell id (shared with :attr:`ScoringContext.prior_belief` and
    ``psr_cells``) to its Core :class:`~astro_mine.core.resource.FieldDistribution` — an
    uncertainty-first ``mean`` + ``variance`` (prospect.md §2.1). Belief-quality metrics read
    the variance to measure how much a campaign reduced resource uncertainty.
    """

    sim_time_s: float
    cells: Mapping[str, FieldDistribution]


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """Scorer-only scoring inputs and parameters a metric may read (prospect.md §9).

    Everything here is defaulted, so an observation-only trace still scores the
    observation-derived metrics (water mass, energy/kg, comms robustness, discovery latency);
    the belief-quality and survival metrics additionally consume the belief history, PSR mask,
    and night windows a scenario supplies. These are scoring *parameters*, not a wire schema —
    there is no ``scoring_context.schema.json`` and none is implied by the move to Core.
    """

    # Belief-quality inputs (information gain, PSR-area characterized).
    prior_belief: Mapping[str, FieldDistribution] = field(default_factory=dict)
    belief_history: tuple[BeliefSnapshot, ...] = ()
    psr_cells: frozenset[str] = frozenset()
    cell_area_m2: float = 1.0
    #: A PSR cell counts as characterized once its posterior variance is at or below this.
    characterized_variance_threshold: float = 0.0
    # Discovery inputs (discovery latency).
    discovery_species: str = "water"
    #: A sensor reading of ``discovery_species`` at or above this counts as a detection.
    discovery_threshold: float = 0.0
    # ISRU accounting (water mass, energy/kg): read from ``SensorReading`` telemetry.
    water_species: str = "water"
    # Survival inputs (nights survived): (start_s, end_s) sim-time windows of lunar night.
    night_intervals: tuple[tuple[float, float], ...] = ()
    #: Minimum survivable temperature (K); ``None`` disables the thermal survival check.
    survivable_temperature_k: float | None = None


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """One episode's scored input: an ordered Core observation stream + a scoring context."""

    observations: tuple[Observation, ...] = ()
    context: ScoringContext = field(default_factory=ScoringContext)


class RunOutcome(BaseModel):
    """One runner execution's deterministic output for a single ``(scenario, seed)``.

    ``determinism_key`` is the runner's byte-for-byte reproducibility digest (a Sim-backed runner
    reports its own ``Trace.content_hash``, so the reproducibility oracle and Sim's own gate check
    *the same artifact* rather than two lookalikes); ``metrics`` are the per-metric scalar scores
    for this seed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    determinism_key: str
    metrics: dict[str, float]


class ScoringRefused(RuntimeError):
    """A runner declined to score, deliberately — part of the runner contract, not a failure.

    An engine-backed runner may find that scoring this scenario would produce a claim it cannot
    support: the canonical case is a pin that resolved by digest but rebuilt no provider, so the
    run would report metrics for content it never modelled (``astro-mine-sim#67``). A scorecard is
    a published claim, and there is no honest use for one made against a world that was never
    loaded — so the runner **raises** rather than returning a degraded trace, because a refusal
    that can be ignored will be.

    It is a distinct type so a caller can present it as an error a *user* can act on — the message
    names what is missing and which package supplies it — while a genuine engine bug keeps its
    traceback. Matching on message text would work today and rot tomorrow
    (``astro-mine-bench#79``).

    Raised by the runner and caught by whoever drives it; Bench never raises it itself. It lives at
    the waist rather than in Bench because both sides of the seam name it: the runner to raise, the
    harness to catch.
    """


@runtime_checkable
class EpisodeScorer(Protocol):
    """Turns episode traces into per-metric aggregate values.

    The seam that lets a *runner* report scored metrics without owning a metric registry. A runner
    that must return a :class:`RunOutcome` — Bench's determinism gate takes no policy and asks
    "does the environment reproduce" — needs metric values, and resolving a scenario's metric
    references to implementations is Bench's job, not the runner's. Before this existed, the Sim
    runner reached across and called ``bench.metrics.resolve_metrics``/``score`` directly, which is
    the last of the five runtime imports that made ``sim -> bench`` a lateral edge.

    ``metrics`` is typed ``Sequence[object]`` on purpose: they are the metric references a Bench
    ``ScenarioSpec`` carries, and Core does not own that vocabulary. This mirrors the choice Bench
    already made in the opposite direction, where ``BenchRunnerProvider`` types its content store
    as ``object`` "so Bench never names a Sim type". A value of ``None`` means the metric did not
    apply to this episode, which is distinct from scoring zero.
    """

    def __call__(
        self,
        traces_by_seed: Mapping[int, EpisodeTrace],
        metrics: Sequence[object],
        *,
        scenario_id: str | None = None,
        runner: str,
    ) -> Mapping[str, float | None]: ...
