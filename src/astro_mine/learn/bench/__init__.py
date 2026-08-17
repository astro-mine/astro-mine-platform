# SPDX-License-Identifier: Apache-2.0
"""Reproducible reference-score / throughput harness + the honest-eval seam (learn.md §8, §10).

Learn *produces* reference scores that [Bench](../../../../docs/architecture/bench.md) scores
and publishes; it does not host a leaderboard. :func:`reference_score` benchmarks a baseline
(learning curve + training throughput + held-out eval); :func:`evaluate` is the RM-P1-LEARN-06
honest-eval seam (held-out envs, seed sweep, variance, comms-stress hook).
"""

from __future__ import annotations

from astro_mine.learn.bench.reference import (
    EvalReport,
    ReferenceReport,
    evaluate,
    reference_score,
)

__all__ = ["EvalReport", "ReferenceReport", "evaluate", "reference_score"]
