"""The metric plugin contract (bench.md §3).

A :class:`Metric` is a plugin ``(EpisodeTrace) -> MetricValue`` carrying a declared name,
interface version, SI-consistent unit, direction (higher/lower-better), and cross-seed
aggregation rule. It reuses Core's objective→metric vocabulary
(:class:`~astro_mine.core.objective.MetricDirection` /
:class:`~astro_mine.core.objective.MetricAggregation`) rather than inventing its own, so a
Bench metric key means the same thing to Core's ``MetricBinding`` (objective/model.py).

Reference metrics ship as replaceable examples (bench.md §4); community/registry-discovered
metric plugins are Phase 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from astro_mine.core.objective import MetricAggregation, MetricDirection
from astro_mine.core.scoring import EpisodeTrace

__all__ = ["Metric", "MetricComputationError", "MetricError", "MetricValue"]


class MetricError(Exception):
    """Base class for metric errors."""


class MetricComputationError(MetricError):
    """Raised when a metric cannot be computed from a trace (e.g. a non-finite result)."""


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One episode's metric outcome — uncertainty-first (bench.md §3 "scalar | distribution").

    ``value`` is ``None`` when the metric is **not applicable** to this episode — energy/kg
    with zero water produced, discovery latency when nothing was discovered, a belief metric
    on a trace with no belief history. Such episodes are excluded from aggregation but counted,
    so censoring stays visible in the scorecard's ``n`` rather than silently biasing the score.
    ``uncertainty`` is the metric's intrinsic per-episode 1-sigma when it emits a distribution
    summary (e.g. the spread of per-cell information gain), else ``None``.
    """

    value: float | None
    unit: str
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        if self.value is not None and not math.isfinite(self.value):
            raise MetricComputationError(f"metric value must be finite, got {self.value!r}")
        if self.uncertainty is not None and (
            not math.isfinite(self.uncertainty) or self.uncertainty < 0.0
        ):
            raise MetricComputationError(
                f"metric uncertainty must be finite and non-negative, got {self.uncertainty!r}"
            )


class Metric(Protocol):
    """The metric plugin interface: metadata plus a deterministic ``compute``.

    A metric MUST be a pure, deterministic function of the trace — same trace ⇒ same value —
    so results are reproducible and a leaderboard can re-execute to verify (bench.md §9). The
    metadata are read-only properties so a frozen dataclass metric satisfies the contract.
    """

    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def unit(self) -> str: ...
    @property
    def direction(self) -> MetricDirection: ...
    @property
    def aggregation(self) -> MetricAggregation: ...

    def compute(self, trace: EpisodeTrace) -> MetricValue: ...
