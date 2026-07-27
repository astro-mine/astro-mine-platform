"""Zonal spherical-harmonic gravity kernels — pure, IO-free (worlds.md §3, §11).

The evaluation half of worlds.md §3's ``gravity/`` module: "point-mass + spherical-harmonic
gravity field evaluation". Pure math over plain 3-tuples (Core's
:data:`~astro_mine.core.world.Vector`) — no numpy, no SPICE, no IO — mirroring
``provider/_geometry.py``, which is where this kernel used to live inlined as a single
hard-coded J2 term.

**The model.** For an axially-symmetric body the exterior potential keeps only the *zonal*
harmonics :math:`J_n` (the :math:`m = 0` terms), and

.. math::

    U(r, \\phi) = \\frac{GM}{r}
        \\left[ 1 - \\sum_{n \\ge 2} J_n \\left(\\frac{R}{r}\\right)^n P_n(\\sin\\phi) \\right]

with :math:`\\phi` the planetocentric latitude, :math:`R` the coefficients' reference radius and
:math:`P_n` the Legendre polynomial. Differentiating radially gives the magnitude this module
returns:

.. math::

    g_r(r, \\phi) = \\frac{GM}{r^2}
        \\left[ 1 - \\sum_{n \\ge 2} (n+1) J_n
        \\left(\\frac{R}{r}\\right)^n P_n(\\sin\\phi) \\right]

At :math:`n = 2` alone this is exactly ``1 - (3/2) J2 (R/r)^2 (3 sin^2 phi - 1)`` — the term the
pre-existing ``gravity_j2`` carried — so a J2-only pack is byte-identical to the old kernel and an
all-zero coefficient set reduces exactly to point-mass.

**Stated approximation (the error budget).** Only the *radial* component is returned: the field is
reported as ``(0, 0, -g_r)`` in the local topocentric surface frame, dropping the latitudinal
:math:`g_\\phi = -(1/r) \\partial U / \\partial \\phi` term. That term is of order ``J2 * g`` — on
the Moon ~3e-4 * 1.62 ~= 5e-4 m/s^2, peaking at mid-latitudes and vanishing at the pole and the
equator — i.e. the deflection of the local vertical from the radial direction. Worlds
*parameterizes* gravity for surface terramechanics and does not propagate orbits (worlds.md §1), so
a surface-normal-aligned radial magnitude is the useful quantity and the deflection is below the
uncertainty of the regolith parameters it is used with. Orbit propagation that needs the full vector
field belongs to Sim/Transit, not here.

No ``pyshtools``-class dependency: at the low degrees worlds.md §12 asks for ("low-order
spherical-harmonic lunar gravity") the standard Bonnet recursion is a dozen lines and keeps the
package light (worlds.md §4 suggests such a library, but only as an option for the general case).
"""

from __future__ import annotations

from collections.abc import Sequence

from astro_mine.core.world import Vector

__all__ = [
    "legendre_p",
    "point_mass_magnitude",
    "zonal_gravity",
    "zonal_gravity_magnitude",
]


def legendre_p(degree: int, x: float) -> float:
    """The Legendre polynomial :math:`P_n(x)` by the Bonnet recursion.

    ``n P_n(x) = (2n - 1) x P_{n-1}(x) - (n - 1) P_{n-2}(x)``, seeded with ``P_0 = 1``,
    ``P_1 = x``. Stable and exact at the low degrees the zonal lunar/Martian fields use.
    """
    if degree < 0:
        raise ValueError(f"Legendre degree must be non-negative, got {degree}")
    if degree == 0:
        return 1.0
    previous, current = 1.0, x
    for n in range(2, degree + 1):
        previous, current = current, ((2 * n - 1) * x * current - (n - 1) * previous) / n
    return current


def point_mass_magnitude(radius_m: float, *, gm_m3_s2: float) -> float:
    """Point-mass gravity magnitude ``GM / r²`` (m·s⁻²) at a distance from the body centre."""
    return gm_m3_s2 / (radius_m * radius_m)


def zonal_gravity_magnitude(
    radius_m: float,
    sin_latitude: float,
    *,
    gm_m3_s2: float,
    reference_radius_m: float,
    zonals: Sequence[float],
) -> float:
    """Radial gravity magnitude (m·s⁻²) from a point mass + a low-order zonal harmonic series.

    ``zonals`` is ``(J2, J3, J4, ...)`` — the unnormalized zonal coefficients **starting at degree
    2**, as tabulated for a published gravity field. An empty (or all-zero) sequence gives exactly
    :func:`point_mass_magnitude`, so a body pack that declares no harmonics is unchanged. The degree
    is therefore *selectable*: a pack ships as many terms as its field justifies, and the kernel
    evaluates whatever it is given rather than a single hard-coded J2.
    """
    g = point_mass_magnitude(radius_m, gm_m3_s2=gm_m3_s2)
    if not zonals:
        return g
    ratio = reference_radius_m / radius_m
    correction = 0.0
    for index, j_n in enumerate(zonals):
        degree = index + 2
        correction += (degree + 1) * j_n * ratio**degree * legendre_p(degree, sin_latitude)
    return g * (1.0 - correction)


def zonal_gravity(
    position: Vector,
    *,
    gm_m3_s2: float,
    reference_radius_m: float,
    zonals: Sequence[float],
) -> Vector:
    """Local gravity vector (m·s⁻²) at a body-fixed ``position``, in the local surface frame.

    The local topocentric surface frame's ``+z`` is the local vertical, so a radial-inward field is
    ``(0, 0, -g_r)`` there. Zero at the body centre, where gravity is undefined — the same total,
    non-raising contract :func:`~astro_mine.worlds.gravity.point_mass_gravity` has always had.
    """
    x, y, z = position
    radius = (x * x + y * y + z * z) ** 0.5
    if radius == 0.0:
        return (0.0, 0.0, 0.0)
    magnitude = zonal_gravity_magnitude(
        radius,
        z / radius,
        gm_m3_s2=gm_m3_s2,
        reference_radius_m=reference_radius_m,
        zonals=zonals,
    )
    return (0.0, 0.0, -magnitude)
