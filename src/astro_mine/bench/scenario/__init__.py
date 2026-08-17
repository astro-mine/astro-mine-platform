# SPDX-License-Identifier: Apache-2.0
"""ScenarioSpec schema + content-hash resolver, and (later) the anchor scenario.

The versioned :class:`ScenarioSpec` pins the Core interface version and references
Worlds/Fleet/Prospect/Link content by hash (plus seeds, episode/horizon, termination,
metric set, budgets); :func:`resolve_scenario` materializes it into a content-addressed
:class:`ResolvedScenario` identity. The anchor scenario ("Lunar Polar Water-Ice
Prospecting v1"), with public dev seeds and an embargoed held-out seed set, lands next.

Backlog: RM-P0-BENCH-01 (this), RM-P0-BENCH-02 (anchor scenario)
astro-mine-bench#1
"""

from __future__ import annotations

from astro_mine.bench.scenario._resolve import (
    IncompatibleCoreSchema,
    ResolvedScenario,
    resolve_scenario,
)
from astro_mine.bench.scenario._spec import (
    BudgetSpec,
    ContentPins,
    ContentRef,
    EpisodeSpec,
    LatLonRegion,
    MetricRef,
    PlacementSpec,
    ScenarioSpec,
    ScoringSpec,
    SeedSet,
    SitePlacement,
    TerminationSpec,
)

__all__ = [
    "BudgetSpec",
    "ContentPins",
    "ContentRef",
    "EpisodeSpec",
    "IncompatibleCoreSchema",
    "LatLonRegion",
    "MetricRef",
    "PlacementSpec",
    "ResolvedScenario",
    "ScenarioSpec",
    "ScoringSpec",
    "SeedSet",
    "SitePlacement",
    "TerminationSpec",
    "resolve_scenario",
]
