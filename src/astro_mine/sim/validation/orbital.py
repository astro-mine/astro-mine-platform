"""Orbital regression against an analytic two-body oracle (RM-P0-SIM-10; sim.md §2.9, §10).

The Phase-0 orbital engine (:mod:`astro_mine.sim.engines.orbital`) is a reduced-order **two-body
RK4** propagator. Its natural, dependency-free oracle is the **closed-form Keplerian solution** of
the very same two-body system — so this module carries the analytic truth the engine is regressed
against in CI, with no external tool required for the shipped tier (the flight-grade GMAT regression
is the live cross-check in ``tests/test_validation_orbital`` and the higher tiers plug in later).

:func:`kepler_propagate` is the universal-variable formulation (Bate-Mueller-White / Curtis Alg.
3.4): singularity-free across eccentricity, so it covers the circular relay orbit and any ellipse.
:func:`validate_orbital_engine` regresses the engine's propagated state against it; the conserved
two-body invariants (specific orbital energy and angular momentum) give a second, model-level gate
via :func:`validate_orbital_conservation`. Both carry an explicit error budget.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from astro_mine.core.units import INERTIAL_J2000
from astro_mine.sim.engines._vecmath import Vec, add, cross, dot, norm, scale
from astro_mine.sim.engines.orbital import OrbitalEngine, orbital_engine_factory
from astro_mine.sim.runtime.rng import RngStreams
from astro_mine.sim.runtime.scenario import AgentSpec, OrbitalDynamics, Scenario
from astro_mine.sim.validation._report import OracleReport, validate_against_oracle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astro_mine.core.units import ReferenceFrame

__all__ = [
    "engine_positions_at_times",
    "kepler_propagate",
    "specific_angular_momentum",
    "specific_energy",
    "validate_orbital_conservation",
    "validate_orbital_engine",
]

#: Newton-iteration controls for the universal Kepler equation — a tight tolerance the analytic
#: oracle must beat the engine's truncation error by, and a generous iteration cap (it converges
#: quadratically in a handful of steps for bound orbits).
_KEPLER_TOL = 1e-11
_KEPLER_MAX_ITERS = 200
#: Below this |z| the Stumpff functions are evaluated by their series, avoiding 0/0 near z = 0.
_STUMPFF_SERIES_BAND = 1e-6


def _stumpff_c(z: float) -> float:
    """The Stumpff function ``C(z)`` (Curtis Alg. 3.1) — series-evaluated near zero."""
    if z > _STUMPFF_SERIES_BAND:
        s = math.sqrt(z)
        return (1.0 - math.cos(s)) / z
    if z < -_STUMPFF_SERIES_BAND:
        s = math.sqrt(-z)
        return (math.cosh(s) - 1.0) / (-z)
    return 0.5 - z / 24.0 + z * z / 720.0


def _stumpff_s(z: float) -> float:
    """The Stumpff function ``S(z)`` (Curtis Alg. 3.2) — series-evaluated near zero."""
    if z > _STUMPFF_SERIES_BAND:
        s = math.sqrt(z)
        return (s - math.sin(s)) / (s * s * s)
    if z < -_STUMPFF_SERIES_BAND:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / (s * s * s)
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def kepler_propagate(r0: Vec, v0: Vec, mu: float, dt_s: float) -> tuple[Vec, Vec]:
    """Propagate a two-body state ``(r0, v0)`` by ``dt_s`` under ``mu`` — the analytic oracle.

    The universal-variable formulation (Curtis Alg. 3.4): solves the universal Kepler equation for
    the anomaly by Newton iteration, then maps the state forward with the Lagrange ``f``/``g``
    coefficients. Singularity-free across eccentricity, so it is exact for the circular relay orbit
    and any ellipse/hyperbola. Returns the propagated ``(position, velocity)``.
    """
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    if dt_s == 0.0:
        return r0, v0
    sqrt_mu = math.sqrt(mu)
    r0_mag = norm(r0)
    if r0_mag == 0.0:
        raise ValueError("initial position must be non-zero")
    vr0 = dot(r0, v0) / r0_mag
    alpha = 2.0 / r0_mag - dot(v0, v0) / mu  # reciprocal semi-major axis (1/a)

    chi = sqrt_mu * abs(alpha) * dt_s  # initial guess (Curtis): good for bound orbits
    for _ in range(_KEPLER_MAX_ITERS):
        z = alpha * chi * chi
        c, s = _stumpff_c(z), _stumpff_s(z)
        f = (
            (r0_mag * vr0 / sqrt_mu) * chi * chi * c
            + (1.0 - alpha * r0_mag) * chi * chi * chi * s
            + r0_mag * chi
            - sqrt_mu * dt_s
        )
        df = (
            (r0_mag * vr0 / sqrt_mu) * chi * (1.0 - alpha * chi * chi * s)
            + (1.0 - alpha * r0_mag) * chi * chi * c
            + r0_mag
        )
        ratio = f / df
        chi -= ratio
        if abs(ratio) < _KEPLER_TOL:
            break

    z = alpha * chi * chi
    c, s = _stumpff_c(z), _stumpff_s(z)
    f_lagrange = 1.0 - (chi * chi / r0_mag) * c
    g_lagrange = dt_s - (1.0 / sqrt_mu) * chi * chi * chi * s
    position = add(scale(r0, f_lagrange), scale(v0, g_lagrange))
    r_mag = norm(position)
    fdot = (sqrt_mu / (r_mag * r0_mag)) * (alpha * chi * chi * chi * s - chi)
    gdot = 1.0 - (chi * chi / r_mag) * c
    velocity = add(scale(r0, fdot), scale(v0, gdot))
    return position, velocity


def specific_energy(r: Vec, v: Vec, mu: float) -> float:
    """The two-body specific orbital energy ``v²/2 - μ/r`` — a conserved invariant."""
    return dot(v, v) / 2.0 - mu / norm(r)


def specific_angular_momentum(r: Vec, v: Vec) -> Vec:
    """The two-body specific angular momentum ``r x v`` — a conserved invariant."""
    return cross(r, v)


def _orbital_engine(
    r0: Vec, v0: Vec, mu: float, substeps: int, frame: ReferenceFrame
) -> OrbitalEngine:
    """A one-orbiter :class:`OrbitalEngine` built through the public factory for regression."""
    scenario = Scenario(
        name="orbital-oracle",
        agents=(
            AgentSpec(
                agent_id="sat",
                initial_position_m=r0,
                velocity_mps=v0,
                battery_soc_j=1.0,
                frame=frame,
                dynamics=OrbitalDynamics(mu_m3_s2=mu, substeps=substeps),
            ),
        ),
        horizon_steps=1,
    )
    return orbital_engine_factory(scenario, RngStreams(0))


def _state(engine: OrbitalEngine) -> tuple[Vec, Vec]:
    sample = engine.export_coupling_state().by_agent["sat"]
    t = sample.pose.translation_m
    v = sample.linear_velocity_mps
    assert v is not None  # the orbital engine always exports velocity
    return (t.x, t.y, t.z), (v.x, v.y, v.z)


def engine_positions_at_times(
    r0: Vec,
    v0: Vec,
    mu: float,
    times_s: Sequence[float],
    *,
    substeps: int = 8,
    frame: ReferenceFrame = INERTIAL_J2000,
) -> list[Vec]:
    """The orbital engine's position at each elapsed time in ``times_s`` (non-decreasing).

    Advances the engine cumulatively to each requested epoch — the seam an external-oracle
    regression uses to sample the engine at an oracle's (possibly irregular) report times, e.g. the
    live GMAT cross-check. ``times_s`` must be non-decreasing; ``0.0`` returns the initial state.
    """
    engine = _orbital_engine(r0, v0, mu, substeps, frame)
    positions: list[Vec] = []
    t_prev = 0.0
    for t in times_s:
        if t > t_prev:
            engine.advance(t - t_prev)
            t_prev = t
        positions.append(_state(engine)[0])
    return positions


def validate_orbital_engine(
    r0: Vec,
    v0: Vec,
    mu: float,
    *,
    dt_s: float,
    steps: int,
    substeps: int = 8,
    budget: float = 1e-4,
    frame: ReferenceFrame = INERTIAL_J2000,
) -> OracleReport:
    """Regress the orbital engine's propagated position against :func:`kepler_propagate`.

    Steps the engine ``steps`` times by ``dt_s`` and compares its position at each step to the
    analytic two-body propagation from the same initial state — the worst relative position error
    must stay within ``budget``. This is the in-CI, dependency-free orbital regression (the live
    STK/GMAT cross-check is the external oracle in the test suite).
    """
    engine = _orbital_engine(r0, v0, mu, substeps, frame)
    actual: list[Vec] = []
    reference: list[Vec] = []
    for k in range(1, steps + 1):
        engine.advance(dt_s)
        position, _ = _state(engine)
        actual.append(position)
        reference.append(kepler_propagate(r0, v0, mu, dt_s * k)[0])
    return validate_against_oracle(
        actual, reference, budget=budget, name="orbital-vs-kepler", relative=True
    )


def validate_orbital_conservation(
    r0: Vec,
    v0: Vec,
    mu: float,
    *,
    dt_s: float,
    steps: int,
    substeps: int = 8,
    budget: float = 1e-6,
) -> OracleReport:
    """Gate the engine's two-body invariant drift — specific energy and angular momentum.

    Both are exactly conserved by the two-body dynamics; RK4 conserves them to high order. The worst
    relative drift of either across ``steps`` must stay within ``budget`` — a model-level oracle
    that needs no propagated reference at all.
    """
    engine = _orbital_engine(r0, v0, mu, substeps, INERTIAL_J2000)
    energy_0 = specific_energy(r0, v0, mu)
    momentum_0 = norm(specific_angular_momentum(r0, v0))
    worst = 0.0
    for _ in range(steps):
        engine.advance(dt_s)
        position, velocity = _state(engine)
        d_energy = abs((specific_energy(position, velocity, mu) - energy_0) / energy_0)
        d_momentum = abs(
            (norm(specific_angular_momentum(position, velocity)) - momentum_0) / momentum_0
        )
        worst = max(worst, d_energy, d_momentum)
    return OracleReport(
        name="orbital-conservation",
        max_error=worst,
        budget=budget,
        detail=f"{steps} steps; worst relative drift of energy/angular-momentum",
    )
