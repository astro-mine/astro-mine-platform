"""Bench reporting — scorecards, provenance bundles, MCAP replay export, and the View handoff.

The bench.md §3 ``report`` module: the surface that packages Bench's results for the components that
render them. Its first realization (RM-P1-BENCH-12) is the **View handoff** — the full-metric
leaderboard dataset and the MCAP episode-replay manifest View surfaces (bench.md §6). Bench provides
the data and the replays; View renders them.

The data shapes here are imported by the leaderboard's FastAPI edge to serve View-facing endpoints,
and can be consumed programmatically. See :mod:`astro_mine.bench.report._view`.

Its second realization is the **performance report** (:mod:`astro_mine.bench.report._performance`):
the measured DEM-vs-surrogate speedup, at the error bound the substitution held to. It is published
*beside* a Scorecard and is **never** folded into ``Scorecard.content_hash`` — wall-clock is
non-deterministic by nature, and Bench's whole reproducibility guarantee rests on the scorecard hash
being a pure function of the run. See that module's docstring for why neither a Metric nor a
Scorecard field can carry a speedup.
"""

from __future__ import annotations

from astro_mine.bench.report._performance import (
    PerformanceReport,
    SeedSpeedup,
    SurrogateIdentity,
    TierTiming,
    ToolchainStamp,
    performance_report,
    toolchain_stamp,
)
from astro_mine.bench.report._view import (
    ViewLeaderboard,
    ViewLeaderboardRow,
    ViewReplay,
    export_leaderboard,
    replay_manifest,
)

__all__ = [
    "PerformanceReport",
    "SeedSpeedup",
    "SurrogateIdentity",
    "TierTiming",
    "ToolchainStamp",
    "ViewLeaderboard",
    "ViewLeaderboardRow",
    "ViewReplay",
    "export_leaderboard",
    "performance_report",
    "replay_manifest",
    "toolchain_stamp",
]
