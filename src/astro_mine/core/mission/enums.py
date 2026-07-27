"""Mission-architecture schema — closed vocabularies (Core-owned, RM-P1-CORE-04 / RFC-0001).

Reserved schema hooks only — **no mechanism** (mission-model.md §3). :class:`Regime` is the
closed, RFC-governed regime enum whose canonical home is
:mod:`astro_mine.core.sadf.enums` (an asset declares which regimes it operates in via
``mobility.regimes``); it is re-exported here so the Mission/Phase schema references the
one definition, mirroring how :mod:`astro_mine.core.registry.enums` reuses it. Adding a
regime — or a maneuver type — is an append-only change made by RFC (mission-model.md §3
"a small, closed, RFC-governed enum … to protect the waist").
"""

from __future__ import annotations

from enum import StrEnum

from astro_mine.core.sadf.enums import Regime

__all__ = ["ManeuverType", "Regime"]


class ManeuverType(StrEnum):
    """The kind of a descriptive :class:`~astro_mine.core.mission.model.Maneuver`.

    A design-time classification of a reference maneuver (impulsive Δv, a finite chemical
    burn, or a continuous low-thrust arc) — descriptive only, for trade studies and
    sizing. It is **not** an executable command: a ``TrajectoryRef`` omits actuator/
    thruster/closed-loop-guidance channels by schema (mission-model.md §4; RFC-0001 R3)."""

    IMPULSIVE = "impulsive"
    FINITE_BURN = "finite_burn"
    LOW_THRUST_ARC = "low_thrust_arc"
