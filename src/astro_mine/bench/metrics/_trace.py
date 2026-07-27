"""The episode-trace input a metric scores (bench.md §3).

A :class:`Metric` maps an *episode trace* to a value. In Phase 0 the trace is the
Bench-side view of what Sim will record to MCAP (sim.md §5): an ordered sequence of Core
:class:`~astro_mine.core.messages.Observation` records — exactly what the Environment API's
``step()`` emits — plus a :class:`ScoringContext` carrying the scorer-only inputs a metric
may read that an agent may not. **The MCAP-file decoder and the ``Sim → EpisodeTrace``
adapter are deferred to Phase 1** (they land with the real Sim runner, mirroring the
harness's ``reference_runner`` deferral); this module is the stable contract the metrics
compute against, so the reference metrics are real and testable today over synthetic traces.

Ground-truth isolation (prospect.md §9): the belief prior/history, the sealed ground truth,
and the PSR mask live in :class:`ScoringContext`, never in the agent-facing observation
stream — a scorer sees them, a policy never does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from astro_mine.core.messages import Observation
from astro_mine.core.resource import FieldDistribution

__all__ = ["BeliefSnapshot", "EpisodeTrace", "ScoringContext"]


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
    and night windows a scenario supplies. Bench owns these scoring parameters — they are not a
    Core schema.
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
