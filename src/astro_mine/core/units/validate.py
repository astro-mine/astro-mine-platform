"""Fail-loud ingest guards for units / frames / time (RM-P0-CORE-06, RM-P1-CORE-08).

The "reject at ingest with a clear error" surface. Components (Worlds, Link) call these
at their boundaries so that spatial data without an explicit CRS or frame, an epoch
without a scale, or a unit outside the SI-consistent vocabulary is refused loudly —
never silently defaulted to Earth/WGS84 (conventions.md §5). Each guard accepts an
already-typed value or a raw mapping and returns the validated model, raising
:class:`UnitsValidationError` otherwise.

This module is the **reference implementation** of the six normative guard rules of
RFC-0007 Design §3 (core.md §8: "the Rust validator is the recommended fast path; Python
is the reference"). Every binding implements the same rules in its own language and
discharges the obligation by running the shared conformance vectors
(``schema/conformance.json``) in its own CI. The rules, in brief:

1. A frame is present; ``name`` / ``center`` are non-empty, whitespace-free tokens.
2. ``frame_class`` ∈ :class:`FrameClass`; ``scale`` ∈ :class:`TimeScale`.
3. :attr:`TimeScale.ET` and :attr:`TimeScale.TDB` denote the **same** scale (SPICE
   ET ≡ TDB) — compare with :func:`scales_equivalent`, never ``==``.
4. A CRS is present; ``body`` / ``body_fixed_frame`` are tokens; ``reference_radius_m``
   is finite and ``> 0``.
5. :attr:`EpochWindow.start` and ``.end`` are present and ``end`` is strictly after
   ``start``.
6. An Earth datum/projection marker (WGS84, EPSG:4326, urn:ogc:def:crs:OGC) is rejected
   when ``body`` is not ``EARTH`` and accepted when it is — an Earth CRS is not
   forbidden, an *implicit* one is. Refusing Earth CRSs outright is a component-local
   policy (e.g. View), **not** this Core rule.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import ValidationError

from astro_mine.core.units.enums import KNOWN_UNITS, TimeScale
from astro_mine.core.units.model import EARTH, Epoch, EpochWindow, PlanetaryCRS, ReferenceFrame

__all__ = [
    "UnitsError",
    "UnitsValidationError",
    "is_si_unit",
    "require_crs",
    "require_epoch",
    "require_epoch_window",
    "require_frame",
    "require_si_unit",
    "scales_equivalent",
]

#: Case-insensitive substrings that mark an Earth datum/projection (rule 6). Not a
#: closed list of every Earth CRS — the common defaulting-bug markers a lunar/other-body
#: product must never carry.
_EARTH_CRS_MARKERS = ("wgs84", "epsg:4326", "urn:ogc:def:crs:ogc")


class UnitsError(Exception):
    """Base class for units/frames/time errors."""


class UnitsValidationError(UnitsError):
    """Raised when a unit, frame, CRS, or epoch fails validation at the waist."""


def is_si_unit(unit: str) -> bool:
    """Return whether ``unit`` is a recognized SI-consistent unit token.

    True for an SI symbol (:data:`~astro_mine.core.units.enums.SI_UNITS`) or a
    recognized dimensionless ratio marker
    (:data:`~astro_mine.core.units.enums.DIMENSIONLESS_UNITS`).
    """
    return unit in KNOWN_UNITS


def require_si_unit(unit: str) -> str:
    """Return ``unit`` if it is SI-consistent, else raise :class:`UnitsValidationError`.

    Core validates the simple unit tokens used in Core-schema fields; it is not a
    dimensional-analysis engine — composite units (e.g. ``kg/m3``, ``J/kg``) in
    component data products are SI-consistent strings validated by those components,
    not enumerated here.
    """
    if not is_si_unit(unit):
        raise UnitsValidationError(
            f"unit {unit!r} is not a recognized SI-consistent unit token "
            f"(known: {', '.join(sorted(KNOWN_UNITS))})"
        )
    return unit


def scales_equivalent(a: TimeScale, b: TimeScale) -> bool:
    """Whether two time scales denote the same physical scale (rule 3).

    :attr:`TimeScale.ET` and :attr:`TimeScale.TDB` are the **same** scale (SPICE
    ET ≡ TDB), so this returns True for any ``(et, tdb)`` pairing. Consumers MUST use
    this instead of ``a == b`` when gating on TDB, so an epoch spelled ``et`` is not
    spuriously rejected against a ``tdb`` one — a naive ``scale == TDB`` is a latent bug
    in every binding (RFC-0007 rule 3).
    """
    return a == b or {a, b} == {TimeScale.ET, TimeScale.TDB}


def require_frame(value: Any) -> ReferenceFrame:
    """Coerce/validate an explicit reference frame, failing loudly on a missing one.

    Accepts a :class:`~astro_mine.core.units.model.ReferenceFrame` or a mapping; rejects
    ``None``/missing/invalid — there is no implicit Earth/WGS84 frame (rules 1-2, enforced
    by the model's token/enum validators).
    """
    if value is None:
        raise UnitsValidationError(
            "a reference frame is required; none was given (no implicit Earth/WGS84 frame)"
        )
    if isinstance(value, ReferenceFrame):
        return value
    if isinstance(value, dict):
        try:
            return ReferenceFrame.model_validate(value)
        except ValidationError as exc:
            raise UnitsValidationError(f"invalid reference frame: {exc}") from exc
    raise UnitsValidationError(f"cannot interpret {type(value).__name__} as a reference frame")


def require_epoch(value: Any) -> Epoch:
    """Coerce/validate an explicit epoch, failing loudly on a missing one (rules 2-3).

    ``scale`` is required and must be a :class:`TimeScale` member. An ``et``-scaled epoch
    is accepted everywhere a ``tdb``-scaled one is (SPICE ET ≡ TDB); compare scales with
    :func:`scales_equivalent`, never ``==``.
    """
    if value is None:
        raise UnitsValidationError("an epoch is required; none was given (no implicit time scale)")
    if isinstance(value, Epoch):
        return value
    if isinstance(value, dict):
        try:
            return Epoch.model_validate(value)
        except ValidationError as exc:
            raise UnitsValidationError(f"invalid epoch: {exc}") from exc
    raise UnitsValidationError(f"cannot interpret {type(value).__name__} as an epoch")


def require_epoch_window(value: Any) -> EpochWindow:
    """Coerce/validate an epoch window, failing loudly on a missing/mis-ordered one.

    Both ``start`` and ``end`` must be present and ``end`` strictly after ``start``
    (rule 5, enforced by the model's ordering validator).
    """
    if value is None:
        raise UnitsValidationError("an epoch window is required; none was given")
    if isinstance(value, EpochWindow):
        return value
    if isinstance(value, dict):
        try:
            return EpochWindow.model_validate(value)
        except ValidationError as exc:
            raise UnitsValidationError(f"invalid epoch window: {exc}") from exc
    raise UnitsValidationError(f"cannot interpret {type(value).__name__} as an epoch window")


def _has_earth_marker(crs: PlanetaryCRS) -> bool:
    """Whether the CRS's projection/datum carries an Earth datum/projection marker."""
    for field in (crs.projection, crs.datum):
        if field is not None and any(m in field.lower() for m in _EARTH_CRS_MARKERS):
            return True
    return False


def require_crs(value: Any) -> PlanetaryCRS:
    """Coerce/validate an explicit planetary CRS, failing loudly on a missing/implicit one.

    Accepts a :class:`~astro_mine.core.units.model.PlanetaryCRS` or a mapping; rejects
    ``None``/missing/invalid so spatial data without an explicit CRS is refused at ingest
    (rule 4). Additionally applies the **body/datum consistency** rule (rule 6): an Earth
    datum/projection marker on a non-``EARTH`` body can only be a defaulting bug and is
    rejected; on ``EARTH`` it is accepted (Phase-2 Earth-analog deployments need Earth
    CRSs to be expressible). This is the Core rule; a component MAY additionally refuse
    Earth CRSs outright as its own policy (conventions.md §5; RM-P0-WORLDS-01).
    """
    if value is None:
        raise UnitsValidationError(
            "spatial data requires an explicit planetary CRS; none was given "
            "(no implicit Earth/WGS84)"
        )
    if isinstance(value, PlanetaryCRS):
        crs = value
    elif isinstance(value, dict):
        try:
            crs = PlanetaryCRS.model_validate(value)
        except ValidationError as exc:
            raise UnitsValidationError(f"invalid planetary CRS: {exc}") from exc
    else:
        raise UnitsValidationError(f"cannot interpret {type(value).__name__} as a planetary CRS")

    # Rule 4: the reference radius must be finite (the model's gt=0 admits +inf).
    if not math.isfinite(crs.reference_radius_m):
        raise UnitsValidationError(
            f"reference_radius_m must be finite and > 0, got {crs.reference_radius_m}"
        )
    # Rule 6: an Earth datum/projection marker is only valid on body EARTH.
    if crs.body.upper() != EARTH and _has_earth_marker(crs):
        raise UnitsValidationError(
            f"Earth CRS marker (WGS84 / EPSG:4326 / urn:ogc:def:crs:OGC) on non-Earth body "
            f"{crs.body!r}: an implicit Earth/WGS84 CRS is a defaulting bug (conventions.md §5). "
            "Set body=EARTH for a legitimate Earth-analog CRS."
        )
    return crs
