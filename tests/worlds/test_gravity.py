"""Gravity — the shared zonal-harmonic kernel + the GRAIL regression (worlds.md §3, §11, §12).

Three layers, matching how the rest of the suite is organised:

- **Kernel tests** drive the IO-free ``gravity/_zonal`` helpers (Legendre recursion, the point-mass
  reduction, the analytic J2 signature) — pure math, no fixtures.
- **The published-reference regression** (worlds.md §10; conventions.md §11) checks the *model
  constants* against ``validation/grail_lunar_gravity.reference.json`` — the GRAIL GRGM1200A
  coefficients read straight off the PDS archive — within that document's **stated error budget**.
  This is the test that fails if a coefficient is edited, mistyped, or re-sourced from another
  field.
- **Pack tests** check that the Moon and Mars evaluate the *same* kernel and that the Moon's
  oblateness term is actually switched on (it shipped as ``j2 = 0.0`` before issue #40).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from astro_mine.worlds.bodies import MARS_PACK, MARS_RADIUS_M, MOON_PACK
from astro_mine.worlds.crs import MOON_RADIUS_M
from astro_mine.worlds.gravity import (
    GRGM1200A_CBAR_ZONALS,
    MARS_GRAVITY,
    MOON_GRAVITY,
    GravityModel,
    gravity_j2,
    legendre_p,
    point_mass_gravity,
    zonal_gravity,
    zonal_gravity_magnitude,
    zonals_from_normalized,
)

REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "validation" / "grail_lunar_gravity.reference.json"
)


@pytest.fixture(scope="module")
def grail() -> dict[str, Any]:
    """The committed published GRAIL reference document."""
    doc: dict[str, Any] = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    assert doc["schema"] == "astro-mine-worlds/gravity-validation/v0.1"
    return doc


# --- the zonal kernel --------------------------------------------------------------


def test_legendre_recursion_matches_closed_forms() -> None:
    for x in (-1.0, -0.4, 0.0, 0.25, 1.0):
        assert legendre_p(0, x) == pytest.approx(1.0)
        assert legendre_p(1, x) == pytest.approx(x)
        assert legendre_p(2, x) == pytest.approx(0.5 * (3 * x**2 - 1))
        assert legendre_p(3, x) == pytest.approx(0.5 * (5 * x**3 - 3 * x))
        assert legendre_p(4, x) == pytest.approx((35 * x**4 - 30 * x**2 + 3) / 8)


def test_legendre_rejects_negative_degree() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        legendre_p(-1, 0.5)


def test_empty_zonals_reduce_to_point_mass() -> None:
    """No coefficients at all is exactly GM/r^2 — the point-mass floor."""
    r = MOON_RADIUS_M
    g = zonal_gravity_magnitude(
        r, 0.3, gm_m3_s2=MOON_GRAVITY.gm_m3_s2, reference_radius_m=1738e3, zonals=()
    )
    assert g == pytest.approx(MOON_GRAVITY.gm_m3_s2 / r**2)


def test_gravity_j2_reduces_to_point_mass_at_zero() -> None:
    """The pre-existing point-mass-reduction contract, now over the shared kernel (issue #40)."""
    pos = (0.0, 0.0, MOON_RADIUS_M + 5000.0)
    assert gravity_j2(
        pos, gm_m3_s2=MOON_GRAVITY.gm_m3_s2, reference_radius_m=MOON_RADIUS_M, j2=0.0
    ) == point_mass_gravity(pos, gm_m3_s2=MOON_GRAVITY.gm_m3_s2)


def test_zonal_gravity_is_zero_at_the_body_centre() -> None:
    assert zonal_gravity((0.0, 0.0, 0.0), gm_m3_s2=1.0, reference_radius_m=1.0, zonals=(1e-3,)) == (
        0.0,
        0.0,
        0.0,
    )
    assert point_mass_gravity((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_j2_only_matches_the_closed_form_correction() -> None:
    """The general kernel at n=2 must equal the analytic 1 - (3/2) J2 (R/r)^2 (3 sin^2 phi - 1)."""
    gm, radius, j2 = 4.9e12, 1.75e6, 2.0e-4
    for latitude in (-90.0, -45.0, 0.0, 30.0):
        sin_lat = math.sin(math.radians(latitude))
        expected = (
            gm / radius**2 * (1.0 - 1.5 * j2 * (MOON_RADIUS_M / radius) ** 2 * (3 * sin_lat**2 - 1))
        )
        got = zonal_gravity_magnitude(
            radius, sin_lat, gm_m3_s2=gm, reference_radius_m=MOON_RADIUS_M, zonals=(j2,)
        )
        assert got == pytest.approx(expected, rel=1e-12)


def test_oblateness_signature_at_constant_radius() -> None:
    """At the SAME radius a positive J2 makes polar gravity *smaller*, not larger.

    The familiar "gravity is stronger at the poles" is a statement about the *surface*, where the
    polar radius is smaller (:func:`test_oblateness_signature_at_the_true_surface_radii`). At fixed
    r the sign is the other way: the equatorial mass excess pulls harder there. Getting this
    backwards is the classic J2 sign error, so both directions are pinned.
    """
    pole = MOON_GRAVITY.magnitude(MOON_RADIUS_M, 90.0)
    equator = MOON_GRAVITY.magnitude(MOON_RADIUS_M, 0.0)
    assert pole < equator
    # And by the analytic first-order amount: g(pole)/g(eq) - 1 = -(9/2) J2 (R/r)^2. The residual is
    # the J3/J4 terms, which do not cancel between pole and equator — hence the 1e-2 tolerance.
    ratio = (MOON_GRAVITY.reference_radius_m / MOON_RADIUS_M) ** 2
    assert pole / equator - 1.0 == pytest.approx(-4.5 * MOON_GRAVITY.j2 * ratio, rel=1e-2)


def test_oblateness_signature_at_the_true_surface_radii() -> None:
    """On the real (flattened) surface the poles do pull harder — by ~0.15%.

    Evaluated at the published lunar polar (1736.0 km) and equatorial (1738.1 km) radii of the
    NASA/NSSDC Moon Fact Sheet, the field reproduces the ~0.15% pole-over-equator surface-gravity
    excess. This is the physical check the constant-radius one above cannot make.
    """
    pole = MOON_GRAVITY.magnitude(1_736_000.0, 90.0)
    equator = MOON_GRAVITY.magnitude(1_738_100.0, 0.0)
    assert pole > equator
    assert pole / equator - 1.0 == pytest.approx(0.0015, abs=0.0005)


# --- the published GRAIL regression (worlds.md §10) ---------------------------------


def test_lunar_constants_match_the_published_grail_archive(grail: dict[str, Any]) -> None:
    """The model's GM, reference radius, and normalized coefficients ARE the archived values."""
    reference = grail["reference"]
    assert MOON_GRAVITY.name == grail["field"]
    assert MOON_GRAVITY.gm_m3_s2 == reference["gm_m3_s2"]
    assert MOON_GRAVITY.reference_radius_m == reference["reference_radius_m"]
    normalized = reference["normalized_zonals"]
    assert (normalized["C20"], normalized["C30"], normalized["C40"]) == GRGM1200A_CBAR_ZONALS


def test_lunar_zonals_match_published_grail_coefficients(grail: dict[str, Any]) -> None:
    """J_n = -C̄_n0 · sqrt(2n+1) must reproduce the published unnormalized zonals in budget."""
    published = grail["reference"]["unnormalized_zonals"]
    budget = grail["error_budget"]["zonal_rel"]
    for index, name in enumerate(("J2", "J3", "J4")):
        assert MOON_GRAVITY.zonals[index] == pytest.approx(published[name], rel=budget)
    # The Moon's oblateness term is REAL now — the whole point of issue #40.
    assert MOON_GRAVITY.j2 > 2.0e-4
    assert MOON_GRAVITY.degree == 4


def test_normalization_relation_is_exact() -> None:
    """The unnormalization is the documented sqrt(2n+1) relation, degree by degree."""
    assert zonals_from_normalized((-1.0,), start_degree=2) == pytest.approx((math.sqrt(5.0),))
    assert zonals_from_normalized((-1.0,), start_degree=3) == pytest.approx((math.sqrt(7.0),))
    assert zonals_from_normalized(()) == ()


def test_lunar_surface_gravity_within_published_error_budget(grail: dict[str, Any]) -> None:
    """Evaluated at the lunar mean radius, the field reproduces the published 1.62 m/s^2."""
    reference = grail["reference"]["mean_surface_gravity_m_s2"]
    budget = grail["error_budget"]["mean_surface_gravity_rel"]
    for latitude in (-90.0, -45.0, 0.0, 45.0, 90.0):
        g = MOON_GRAVITY.magnitude(MOON_RADIUS_M, latitude)
        assert g == pytest.approx(reference, rel=budget)


def test_low_order_truncation_error_budget(grail: dict[str, Any]) -> None:
    """J3 + J4 move the surface field by less than the stated truncation budget vs J2 alone."""
    budget = grail["error_budget"]["truncation_rel"]
    j2_only = GravityModel(
        name="J2-only",
        gm_m3_s2=MOON_GRAVITY.gm_m3_s2,
        reference_radius_m=MOON_GRAVITY.reference_radius_m,
        zonals=MOON_GRAVITY.zonals[:1],
    )
    worst = max(
        abs(MOON_GRAVITY.magnitude(MOON_RADIUS_M, lat) / j2_only.magnitude(MOON_RADIUS_M, lat) - 1)
        for lat in range(-90, 91, 5)
    )
    assert 0.0 < worst < budget  # non-zero: the extra degrees genuinely do something


# --- the body packs share one kernel ------------------------------------------------


def test_moon_pack_carries_the_grail_field() -> None:
    """MOON_PACK is no longer point-mass-only (it shipped j2 = 0.0 before issue #40)."""
    assert MOON_PACK.gravity_model is MOON_GRAVITY
    assert MOON_PACK.j2 > 0.0
    assert MOON_PACK.gm_m3_s2 == MOON_GRAVITY.gm_m3_s2
    # Gravity is latitude-dependent now — the south pole (where the anchor operates) is no longer
    # the same number as the equator, which is exactly what j2 = 0.0 used to force.
    pole = MOON_PACK.gravity((0.0, 0.0, -MOON_RADIUS_M))
    equator = MOON_PACK.gravity((MOON_RADIUS_M, 0.0, 0.0))
    assert pole[2] != equator[2]
    # ...but only by the small J2 amount — the anchor's surface gravity is still ~1.62 m/s^2.
    assert abs(pole[2]) == pytest.approx(1.62, abs=0.02)
    assert 0.0 < abs(pole[2] / equator[2] - 1.0) < 1e-3


def test_both_packs_evaluate_the_same_kernel() -> None:
    """Lunar and Martian gravity go through one gravity/ module — no duplicated zonal kernel."""
    assert MARS_PACK.gravity_model is MARS_GRAVITY
    for pack, position in (
        (MOON_PACK, (0.0, 0.0, -MOON_RADIUS_M)),
        (MARS_PACK, (MARS_RADIUS_M, 0.0, 0.0)),
    ):
        assert pack.gravity(position) == pack.gravity_model.acceleration(position)
        assert pack.gravity(position) == zonal_gravity(
            position,
            gm_m3_s2=pack.gravity_model.gm_m3_s2,
            reference_radius_m=pack.gravity_model.reference_radius_m,
            zonals=pack.gravity_model.zonals,
        )


def test_mars_surface_gravity_is_physical() -> None:
    g = MARS_PACK.gravity((MARS_RADIUS_M, 0.0, 0.0))
    assert abs(g[2]) == pytest.approx(3.71, abs=0.05)
