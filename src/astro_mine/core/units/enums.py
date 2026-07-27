"""Units / frames / time — closed vocabularies (Core-owned, RM-P0-CORE-06).

The closed part of the units/frames/time waist: the admissible time scales, the
reference-frame classes, and the canonical SI unit catalog. Deliberately small;
members are append-only and grow only by RFC (conventions.md §5; mission-model.md §3
"grow by capability, not type"). Members are never removed or repurposed.

These vocabularies are *engine-neutral identifiers*, not behaviour: Core names SPICE
frames and TDB scales but never resolves them (no heavy deps, core.md §2.3) — the
name→geometry resolution is Worlds (RM-P0-WORLDS-02, SpiceyPy).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DIMENSIONLESS_UNITS",
    "KNOWN_UNITS",
    "SI_UNITS",
    "FrameClass",
    "SiUnit",
    "TimeScale",
]


class TimeScale(StrEnum):
    """Admissible epoch time scale at the waist.

    Only the SI-second, SPICE ephemeris scales are admitted: ``TDB`` (Barycentric
    Dynamical Time) and its SPICE alias ``ET`` (Ephemeris Time ≡ TDB in SPICE). Civil
    or atomic scales (UTC/TAI/…) are unrepresentable by construction, so a missing or
    non-TDB scale fails loudly — there is no implicit time scale (conventions.md §5).
    """

    TDB = "tdb"
    ET = "et"


class FrameClass(StrEnum):
    """Reference-frame class.

    ``BODY_FIXED`` rotates with a body (e.g. ``MOON_ME``); ``INERTIAL`` does not
    (e.g. ``J2000`` / ``ICRF``); ``TOPOCENTRIC`` is a local surface/site frame used by
    Link LOS and View. Closed; grows by RFC.
    """

    BODY_FIXED = "body_fixed"
    INERTIAL = "inertial"
    TOPOCENTRIC = "topocentric"


class SiUnit(StrEnum):
    """Canonical SI unit symbols the platform's Core-schema unit fields speak.

    The SI base and angle units plus the named derived units that appear in Core
    message/SADF ``unit`` fields. This is **not** a dimensional-analysis system:
    composite units in component data products (e.g. Worlds' ``kg·m⁻³`` regolith
    fields, a ``J/kg`` Bench metric) are SI-consistent strings validated by those
    components, never enumerated here. Append-only by RFC.
    """

    # SI base units
    METRE = "m"
    KILOGRAM = "kg"
    SECOND = "s"
    AMPERE = "A"
    KELVIN = "K"
    MOLE = "mol"
    CANDELA = "cd"
    # SI angle (supplementary)
    RADIAN = "rad"
    STERADIAN = "sr"
    # named derived units used in Core schemas
    HERTZ = "Hz"
    NEWTON = "N"
    PASCAL = "Pa"
    JOULE = "J"
    WATT = "W"
    VOLT = "V"
    COULOMB = "C"


#: The strict SI unit symbols (the values of :class:`SiUnit`).
SI_UNITS: frozenset[str] = frozenset(u.value for u in SiUnit)

#: Recognized dimensionless ratio markers used in Core ``si_unit`` / ``unit`` fields
#: (e.g. SADF ``ResourceTarget.si_unit``). Dimensionless, so not SI *symbols*, but
#: valid unit tokens at the waist. Append-only by RFC.
DIMENSIONLESS_UNITS: frozenset[str] = frozenset(
    {
        "dimensionless",
        "mass_fraction",
        "volume_fraction",
    }
)

#: Every unit token Core recognizes at the waist: SI symbols plus dimensionless
#: markers. The substrate for :func:`astro_mine.core.units.validate.require_si_unit`.
KNOWN_UNITS: frozenset[str] = SI_UNITS | DIMENSIONLESS_UNITS
