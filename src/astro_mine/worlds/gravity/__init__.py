"""Gravity — point-mass + low-order spherical-harmonic field evaluation (worlds.md §3, §11, §12).

worlds.md §3's module map reserves ``gravity/`` for "point-mass + spherical-harmonic gravity field
evaluation"; §11 recommends "**point-mass + spherical harmonics** for the Moon/Mars"; and §12's
Phase-0 MVP list names "point-mass + **low-order spherical-harmonic lunar gravity**". This module is
that: one zonal-harmonic kernel (:mod:`._zonal`), shared by every body pack, replacing the single
hard-coded J2 term that used to sit inlined in ``provider/_geometry.py`` — where the Moon carried
``j2 = 0.0`` ("point-mass only") and only Mars had a real coefficient.

A :class:`GravityModel` bundles **one published field**, coherently: its ``GM``, the reference
radius ``R`` its coefficients are normalized to, and its low-order zonal harmonics. Keeping them
together matters — the reference radius of a gravity field is *not* the body's CRS datum radius
(GRAIL's coefficients are referenced to R = 1738.0 km; the lunar CRS datum is the 1737.4 km
volumetric mean radius), and mixing coefficients across fields, or with the wrong ``R``, silently
corrupts the correction. The degree is **selectable**: a pack ships as many terms as its published
field justifies (the Moon carries J2/J3/J4), and the kernel evaluates whatever it is given.

**Provenance.** The lunar coefficients are the *archived, normalized* Stokes coefficients read
straight out of the GRAIL **GRGM1200A** spherical-harmonic ASCII data record on the PDS Geosciences
Node — not a secondary table — and unnormalized here by the documented
:func:`zonals_from_normalized` relation, so the committed numbers are exactly the published ones and
the conversion is itself testable. The regression test against them, with its error budget, is
``validation/grail_lunar_gravity.reference.json`` + ``tests/test_gravity.py``
(worlds.md §10; conventions.md §11).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from astro_mine.core.world import Vector
from astro_mine.worlds.gravity._zonal import (
    legendre_p,
    point_mass_magnitude,
    zonal_gravity,
    zonal_gravity_magnitude,
)

__all__ = [
    "GRGM1200A_CBAR_ZONALS",
    "GRGM1200A_GM_M3_S2",
    "GRGM1200A_REFERENCE_RADIUS_M",
    "GRGM1200A_SOURCE",
    "MARS_GRAVITY",
    "MARS_J2",
    "MOON_GRAVITY",
    "MOON_ZONALS",
    "GravityModel",
    "gravity_j2",
    "legendre_p",
    "point_mass_gravity",
    "point_mass_magnitude",
    "zonal_gravity",
    "zonal_gravity_magnitude",
    "zonals_from_normalized",
]


def zonals_from_normalized(cbar: Sequence[float], *, start_degree: int = 2) -> tuple[float, ...]:
    """Unnormalized zonal harmonics ``J_n`` from archived 4π-normalized ``C̄(n, 0)`` coefficients.

    Gravity fields are archived with **normalized** Stokes coefficients; the ``J_n`` a
    reduced-order model is written in terms of are the unnormalized ones::

        C(n, 0) = C̄(n, 0) · sqrt(2n + 1)      J_n = -C(n, 0)

    Doing the conversion here — rather than committing pre-converted numbers — means the constants
    in this file are byte-for-byte the values in the published archive, and the (easy to get wrong)
    normalization is a tested code path rather than an unverifiable hand calculation.
    """
    return tuple(
        -value * math.sqrt(2 * (start_degree + index) + 1) for index, value in enumerate(cbar)
    )


# --- The Moon: GRAIL GRGM1200A ------------------------------------------------------

#: The published field the lunar coefficients come from. GRGM1200A is the GSFC degree-1200 GRAIL
#: solution; the values below are read from its PDS spherical-harmonic ASCII data record
#: (``gggrx_1200a_sha.tab``) header + coefficient rows.
GRGM1200A_SOURCE = (
    "GRAIL GRGM1200A (GSFC), PDS Geosciences Node LRO/GRAIL archive "
    "grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200a_sha.tab; "
    "Goossens et al. (2016), LPSC XLVII #1484; the field's PDS label names "
    "Lemoine et al. (2014), GRL 41, 3382-3389, doi:10.1002/2014GL060027 as its best reference."
)

#: Lunar gravitational parameter GM (m^3 s^-2) — the GRGM1200A header value (4902.8001224453
#: km^3/s^2). This is the GM the field's coefficients are consistent with; the archive's *label*
#: quotes a value differing in the 9th significant figure, and the table header is the one to use.
GRGM1200A_GM_M3_S2 = 4.9028001224453e12

#: The reference radius the GRGM1200A coefficients are normalized to (m) — 1738.0 km. **Not** the
#: 1737.4 km volumetric-mean radius the lunar CRS uses as its datum: a J_n term scales as (R/r)^n,
#: so evaluating published coefficients against the wrong R is a real (if small) error.
GRGM1200A_REFERENCE_RADIUS_M = 1_738_000.0

#: The archived 4π-normalized zonal Stokes coefficients C̄(2,0), C̄(3,0), C̄(4,0) of GRGM1200A,
#: verbatim from the PDS data record. Tide-free convention (the archived degree-2 coefficients
#: exclude the permanent tide) — restoring it would raise J2 by ~0.045%, so the convention is
#: pinned here rather than left implicit.
GRGM1200A_CBAR_ZONALS = (
    -9.0884339347424299e-05,  # C̄(2, 0)
    -3.1973308084610398e-06,  # C̄(3, 0)
    +3.2347808442570100e-06,  # C̄(4, 0)
)

#: The lunar unnormalized zonal harmonics ``(J2, J3, J4)`` — J2 ≈ 2.0322e-4, the oblateness term
#: MOON_PACK previously set to **zero** ("point-mass only, per the Phase-0 model").
MOON_ZONALS = zonals_from_normalized(GRGM1200A_CBAR_ZONALS)


# --- Mars: GMM-3 --------------------------------------------------------------------

#: Mars J2 zonal harmonic (oblateness) — ~10x the Moon's; the value the Mars pack has always
#: carried (RM-P1-WORLDS-11), now sited with the lunar coefficients rather than in the provider.
MARS_J2 = 1.96045e-3

#: The reference radius Martian gravity fields (GMM-3, Genova et al. 2016, Icarus 272, 228-245)
#: normalize to — 3396.0 km, the equatorial radius; again not the 3389.5 km volumetric mean the
#: Mars CRS uses as its datum.
MARS_GRAVITY_REFERENCE_RADIUS_M = 3_396_000.0

#: Mars gravitational parameter GM (m^3 s^-2); surface g ~ 3.71 m/s^2.
MARS_GM_M3_S2 = 4.282837e13


@dataclass(frozen=True)
class GravityModel:
    """One published gravity field: ``GM``, its reference radius, and its zonal harmonics.

    The three travel together on purpose (see the module docstring): ``zonals`` is
    ``(J2, J3, J4, ...)`` from degree 2, unnormalized, and it is evaluated against
    ``reference_radius_m`` — the radius the *coefficients* are normalized to, which is not in
    general the body's CRS datum radius. An empty ``zonals`` is a pure point mass.
    """

    name: str
    gm_m3_s2: float
    reference_radius_m: float
    zonals: tuple[float, ...]
    source: str = ""

    @property
    def j2(self) -> float:
        """The degree-2 zonal harmonic, or 0.0 for a point-mass field."""
        return self.zonals[0] if self.zonals else 0.0

    @property
    def degree(self) -> int:
        """The highest zonal degree carried (2 for J2 only; 0 for a pure point mass)."""
        return len(self.zonals) + 1 if self.zonals else 0

    def acceleration(self, position: Vector) -> Vector:
        """Local gravity vector (m/s^2) at a body-fixed ``position``, in the local surface frame."""
        return zonal_gravity(
            position,
            gm_m3_s2=self.gm_m3_s2,
            reference_radius_m=self.reference_radius_m,
            zonals=self.zonals,
        )

    def magnitude(self, radius_m: float, latitude_deg: float) -> float:
        """Radial gravity magnitude (m/s^2) at a planetocentric radius and latitude."""
        return zonal_gravity_magnitude(
            radius_m,
            math.sin(math.radians(latitude_deg)),
            gm_m3_s2=self.gm_m3_s2,
            reference_radius_m=self.reference_radius_m,
            zonals=self.zonals,
        )


#: The lunar field: point-mass + the low-order GRAIL zonal harmonics (worlds.md §12's Phase-0 MVP).
MOON_GRAVITY = GravityModel(
    name="GRGM1200A",
    gm_m3_s2=GRGM1200A_GM_M3_S2,
    reference_radius_m=GRGM1200A_REFERENCE_RADIUS_M,
    zonals=MOON_ZONALS,
    source=GRGM1200A_SOURCE,
)

#: The Martian field — J2 only, the term the Mars pack already carried.
MARS_GRAVITY = GravityModel(
    name="GMM-3 (J2)",
    gm_m3_s2=MARS_GM_M3_S2,
    reference_radius_m=MARS_GRAVITY_REFERENCE_RADIUS_M,
    zonals=(MARS_J2,),
    source="Genova et al. (2016), Icarus 272, 228-245, doi:10.1016/j.icarus.2016.02.050",
)


def point_mass_gravity(position: Vector, *, gm_m3_s2: float = GRGM1200A_GM_M3_S2) -> Vector:
    """Local gravity vector (m/s^2) at ``position`` in the local topocentric surface frame.

    Point-mass gravity is radial-inward, so in the local surface frame whose +z is the local
    vertical it is exactly ``(0, 0, -g)`` with ``g = GM/r^2`` (``r`` the distance from the body
    centre). ``gm_m3_s2`` defaults to the lunar GM; a body pack passes its own. Zero at the body
    centre, where gravity is undefined.
    """
    return zonal_gravity(position, gm_m3_s2=gm_m3_s2, reference_radius_m=1.0, zonals=())


def gravity_j2(
    position: Vector, *, gm_m3_s2: float, reference_radius_m: float, j2: float
) -> Vector:
    """Point-mass + **J2 zonal** gravity in the local surface frame (RM-P1-WORLDS-11).

    The single-term special case of :func:`~astro_mine.worlds.gravity._zonal.zonal_gravity`, kept as
    the named entry point it has always been: the radial gravity carries the J2 oblateness
    correction ``g_r = GM/r^2 · [1 - (3/2) J2 (R/r)^2 (3 sin^2 φ - 1)]``. At ``j2 = 0`` this is
    exactly :func:`point_mass_gravity`.
    """
    return zonal_gravity(
        position,
        gm_m3_s2=gm_m3_s2,
        reference_radius_m=reference_radius_m,
        zonals=(j2,),
    )
