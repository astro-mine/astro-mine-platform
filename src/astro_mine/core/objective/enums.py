# SPDX-License-Identifier: Apache-2.0
"""ObjectiveSpec v0.1 — closed vocabularies (Core-owned).

Small, closed enums for the objective->metric binding. Like the SADF vocabularies
(``astro_mine.core.sadf.enums``) these grow only by RFC: members are append-only and
never removed or repurposed (conventions.md §3).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "MetricAggregation",
    "MetricDirection",
    "WindowKind",
]


class MetricDirection(StrEnum):
    """Whether a higher or lower metric value is better (scenario §13 scoring)."""

    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"


class MetricAggregation(StrEnum):
    """How a metric's per-seed / per-episode values combine into one score
    (bench.md §3; scenario §13). Deterministic and content-addressed so a design-time
    score and an operational reading of the same objective are comparable (LUNAR-TR-006)."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    P05 = "p05"
    P95 = "p95"


class WindowKind(StrEnum):
    """How a metric binding is evaluated over time (the ``evaluation_window`` kind).

    - ``cumulative`` — over the whole episode/campaign (the default if no window is set);
    - ``rolling`` — over a rolling window of ``duration_s`` (rate / sustained objectives,
      e.g. "10 t per lunar day");
    - ``per_phase`` — once per mission Phase (meaningful only for multi-phase Missions;
      RFC-0001, reserved P1).

    A ``rolling`` window requires ``duration_s``; ``cumulative``/``per_phase`` forbid it
    (enforced in the loader)."""

    CUMULATIVE = "cumulative"
    ROLLING = "rolling"
    PER_PHASE = "per_phase"
