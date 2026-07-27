"""Units / frames / time — typed value primitives (RM-P0-CORE-06).

The shared spatial/temporal types every component resolves against: a
:class:`ReferenceFrame`, a :class:`PlanetaryCRS`, an :class:`Epoch`, and an
:class:`EpochWindow`. They are the single source of truth Worlds (RM-P0-WORLDS-01/02)
and Link (RM-P0-LINK-01) consume so the platform shares one frame/time vocabulary.

**SPICE-shaped, dependency-light.** These types *name* SPICE frames/bodies and carry
TDB seconds; they never call SPICE (Core forbids heavy deps, core.md §2.3). The
name→transform resolution and kernel management live in Worlds (RM-P0-WORLDS-02,
SpiceyPy). Core's job is to make a frame/CRS/epoch *explicit and well-formed* and to
fail loudly otherwise — there is no implicit Earth/WGS84 anywhere (conventions.md §5).

All quantities are SI; every value is frame- and scale-explicit.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astro_mine.core.units.enums import FrameClass, TimeScale

__all__ = [
    "EARTH",
    "INERTIAL_J2000",
    "J2000_EPOCH",
    "MOON",
    "MOON_BODY_FIXED",
    "SUN",
    "Epoch",
    "EpochWindow",
    "PlanetaryCRS",
    "ReferenceFrame",
]


class _Model(BaseModel):
    """Base for every units/frames/time model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


def _validate_token(value: str) -> str:
    """Require a non-empty, whitespace-free identifier (a SPICE frame/body name).

    A blank, padded, or whitespace-bearing name is almost always a defaulting bug;
    rejecting it at construction keeps frame/body identifiers clean and SPICE-ready.
    """
    if not value or value != value.strip() or any(c.isspace() for c in value):
        raise ValueError(f"must be a non-empty, whitespace-free token, got {value!r}")
    return value


class ReferenceFrame(_Model):
    """A named reference frame.

    ``name`` is a SPICE frame name (e.g. ``MOON_ME`` body-fixed, ``J2000`` inertial) —
    Core carries and validates the identifier; Worlds resolves it to a transform via
    SPICE. ``center`` is the SPICE body the frame is centered on (e.g. ``MOON``), or
    ``None`` for a centre-agnostic sky frame.
    """

    name: str
    frame_class: FrameClass
    center: str | None = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _validate_token(v)

    @field_validator("center")
    @classmethod
    def _check_center(cls, v: str | None) -> str | None:
        return None if v is None else _validate_token(v)


class PlanetaryCRS(_Model):
    """An explicit planetary coordinate reference system.

    The minimum needed to reproject spatial data without guessing: the ``body``, its
    body-fixed ``body_fixed_frame``, and the PROJ planetary reference radius
    ``reference_radius_m`` (the PROJ ``+R``; the Moon is modelled as a sphere,
    R ≈ 1_737_400 m — ellipsoid ``+a``/``+b`` is deferred). ``projection`` carries an
    explicit PROJ/WKT/EPSG string for a *projected* CRS (e.g. lunar polar stereographic
    for the Shackleton DEM); ``None`` means body-fixed geographic (lat/lon on the
    datum). No field defaults to an Earth/WGS84 value — a CRS is explicit or it is
    rejected (RM-P0-WORLDS-01; conventions.md §5).
    """

    body: str
    body_fixed_frame: str
    reference_radius_m: float = Field(gt=0.0)
    projection: str | None = None
    datum: str | None = None

    @field_validator("body", "body_fixed_frame")
    @classmethod
    def _check_token(cls, v: str) -> str:
        return _validate_token(v)


class Epoch(_Model):
    """An instant in TDB/ET (SI seconds past the J2000 TDB epoch).

    ``tdb_seconds`` is SPICE ephemeris time directly. ``scale`` is required (no
    default) so a scaleless epoch fails loudly; only :class:`TimeScale` values are
    representable, so a civil/atomic scale cannot be smuggled in.
    """

    tdb_seconds: float
    scale: TimeScale


class EpochWindow(_Model):
    """A half-open epoch interval ``[start, end)`` over which a product is defined.

    The canonical "defined epoch window" Worlds illumination/PSR (RM-P0-WORLDS-02/03)
    and Link contact windows (RM-P0-LINK-02) state their validity over. ``end`` must be
    strictly after ``start``.
    """

    start: Epoch
    end: Epoch

    @model_validator(mode="after")
    def _check_order(self) -> EpochWindow:
        if self.end.tdb_seconds <= self.start.tdb_seconds:
            raise ValueError(
                "EpochWindow end must be strictly after start "
                f"(start={self.start.tdb_seconds}, end={self.end.tdb_seconds})"
            )
        return self


# --- canonical constants (shared names so consumers don't reinvent them) ----------

#: NAIF body names the lunar anchor scenario resolves geometry against.
MOON = "MOON"
SUN = "SUN"
EARTH = "EARTH"

#: The Moon's body-fixed mean-Earth/polar-axis frame (Worlds loads the lunar frame
#: kernel; ``IAU_MOON`` is the lower-accuracy built-in alternative).
MOON_BODY_FIXED = ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center=MOON)

#: The Earth-mean-equator/equinox-of-J2000 inertial frame.
INERTIAL_J2000 = ReferenceFrame(name="J2000", frame_class=FrameClass.INERTIAL)

#: The J2000 TDB epoch (ephemeris-time origin).
J2000_EPOCH = Epoch(tdb_seconds=0.0, scale=TimeScale.TDB)
