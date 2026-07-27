"""Units, frames, and time — the shared spatial/temporal primitives (RM-P0-CORE-06).

SI everywhere; SPICE-backed body-fixed/inertial frames and TDB/ET epochs; explicit
planetary CRS. Ingest fails loudly on a missing or defaulted frame — no implicit
Earth/WGS84 conventions (conventions.md §5).

Core owns the *types and fail-loud validation*, not the geometry: a
:class:`ReferenceFrame`/:class:`Epoch` names a SPICE frame and a TDB instant but Core
never calls SPICE (no heavy deps, core.md §2.3). The name→transform resolution and
kernel management live in Worlds (RM-P0-WORLDS-02, SpiceyPy); these primitives are the
single source of truth Worlds and Link build on so the platform shares one vocabulary.

Public API:

- types — :class:`ReferenceFrame`, :class:`PlanetaryCRS`, :class:`Epoch`,
  :class:`EpochWindow`;
- vocabularies — :class:`TimeScale`, :class:`FrameClass`, :class:`SiUnit`
  (and the :data:`SI_UNITS` / :data:`DIMENSIONLESS_UNITS` / :data:`KNOWN_UNITS` sets);
- fail-loud guards — :func:`require_crs`, :func:`require_frame`, :func:`require_si_unit`,
  :func:`is_si_unit`, and the :class:`UnitsValidationError` they raise;
- canonical constants — :data:`MOON_BODY_FIXED`, :data:`INERTIAL_J2000`,
  :data:`J2000_EPOCH`, :data:`MOON` / :data:`SUN` / :data:`EARTH`.

Backlog: RM-P0-CORE-06 — https://github.com/astro-mine/astro-mine-core/issues/6
"""

from __future__ import annotations

from astro_mine.core.units import enums, model, validate
from astro_mine.core.units.enums import (
    DIMENSIONLESS_UNITS,
    KNOWN_UNITS,
    SI_UNITS,
    FrameClass,
    SiUnit,
    TimeScale,
)
from astro_mine.core.units.model import (
    EARTH,
    INERTIAL_J2000,
    J2000_EPOCH,
    MOON,
    MOON_BODY_FIXED,
    SUN,
    Epoch,
    EpochWindow,
    PlanetaryCRS,
    ReferenceFrame,
)
from astro_mine.core.units.validate import (
    UnitsError,
    UnitsValidationError,
    is_si_unit,
    require_crs,
    require_epoch,
    require_epoch_window,
    require_frame,
    require_si_unit,
    scales_equivalent,
)

__all__ = [
    "DIMENSIONLESS_UNITS",
    "EARTH",
    "INERTIAL_J2000",
    "J2000_EPOCH",
    "KNOWN_UNITS",
    "MOON",
    "MOON_BODY_FIXED",
    "SI_UNITS",
    "SUN",
    "Epoch",
    "EpochWindow",
    "FrameClass",
    "PlanetaryCRS",
    "ReferenceFrame",
    "SiUnit",
    "TimeScale",
    "UnitsError",
    "UnitsValidationError",
    "enums",
    "is_si_unit",
    "model",
    "require_crs",
    "require_epoch",
    "require_epoch_window",
    "require_frame",
    "require_si_unit",
    "scales_equivalent",
    "validate",
]
