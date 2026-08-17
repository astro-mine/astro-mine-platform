# SPDX-License-Identifier: Apache-2.0
"""Mission-architecture schema — the reserved RFC-0001 hooks (RM-P1-CORE-04).

**Schema only, no mechanism.** A :class:`MissionSpec` is an ordered set of :class:`Phase`\\
s, each in a :class:`Regime`, with descriptive design-time :class:`TrajectoryRef` /
:class:`ManeuverBudget` artifacts on the inter-phase :class:`Leg`\\ s and a typed
:class:`PhaseTransition` handoff event on the Environment API. A single-``surface``-phase
Mission is exactly today's campaign, so the whole schema is **additive**: existing
consumers ignore ``regime`` and a one-phase MissionSpec validates with no author action
(mission-model.md §5). Implementations (Transit/Trajectory/Sizing/Ledger, phase
sequencing) land in Phase 3; Core only reserves the hooks now so the narrow waist is never
retrofitted (mission-model.md §3).

Dual-use (RFC-0001 R3): :class:`TrajectoryRef` / :class:`Maneuver` are descriptive
artifacts that omit actuator/thruster/closed-loop-guidance channels **by schema**;
operational maneuver targeting stays gated by the ``operational_targeting`` capability tag.

Public API:

- the document — :class:`MissionDocument`, :class:`MissionSpec`, :class:`Phase`,
  :class:`Leg`, :class:`PhaseTransition`, :class:`TrajectoryRef`, :class:`Maneuver`,
  :class:`ManeuverBudget`, and the closed vocabularies :class:`Regime` /
  :class:`ManeuverType`;
- load + validate — :func:`load_mission` / :func:`validate_mission` / :func:`load_schema`;
- wire — :func:`to_wire` / :func:`from_wire` / :func:`to_proto` / :func:`from_proto`.
"""

from __future__ import annotations

from astro_mine.core.mission import enums, loader, model, wire
from astro_mine.core.mission.enums import ManeuverType, Regime
from astro_mine.core.mission.loader import (
    MissionError,
    MissionValidationError,
    load_mission,
    load_schema,
    validate_mission,
)
from astro_mine.core.mission.model import (
    Leg,
    Maneuver,
    ManeuverBudget,
    MissionConstraints,
    MissionDocument,
    MissionSpec,
    Phase,
    PhaseBoundary,
    PhaseTransition,
    Provenance,
    ReferenceState,
    TrajectoryRef,
    TrajectorySegment,
    Vec3,
)
from astro_mine.core.mission.wire import from_proto, from_wire, to_proto, to_wire

__all__ = [
    "Leg",
    "Maneuver",
    "ManeuverBudget",
    "ManeuverType",
    "MissionConstraints",
    "MissionDocument",
    "MissionError",
    "MissionSpec",
    "MissionValidationError",
    "Phase",
    "PhaseBoundary",
    "PhaseTransition",
    "Provenance",
    "ReferenceState",
    "Regime",
    "TrajectoryRef",
    "TrajectorySegment",
    "Vec3",
    "enums",
    "from_proto",
    "from_wire",
    "load_mission",
    "load_schema",
    "loader",
    "model",
    "to_proto",
    "to_wire",
    "validate_mission",
    "wire",
]
