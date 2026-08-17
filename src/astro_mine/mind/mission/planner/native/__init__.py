# SPDX-License-Identifier: Apache-2.0
"""Native mission-planner backends (RM-P1-MIND-03) — the ``[pddl]`` extra.

The real symbolic-planning engine behind Core's
:class:`~astro_mine.core.policy.protocol.MissionPlanner` sub-interface:
:class:`~astro_mine.mind.mission.planner.native.unified.UnifiedPlanningMissionPlanner` solves the
generated PDDL problem with **unified-planning** — the backend-agnostic façade over Fast Downward
/ OPTIC / ENHSP that mind.md §4/§11 names as the default — and turns the engine's plan into the
per-agent prospect decomposition.

Heavy, native-toolchain dependencies: this subpackage is imported only when the ``[pddl]`` extra
is installed, so every import of ``unified_planning`` is **deferred into the call** rather than
taken at module import (the base wheel must stay importable, and the entry-point provider below
is loaded by every ``TierRegistry.from_entry_points()``). Its tests are marker-gated (``pddl``)
and deselected in CI; the pure-Python reference planner remains the CI-tested default.
"""

from __future__ import annotations

from astro_mine.mind.mission.planner.native.unified import (
    UnifiedPlanningMissionPlanner,
    up_mission_plugin,
)

__all__ = ["UnifiedPlanningMissionPlanner", "up_mission_plugin"]
