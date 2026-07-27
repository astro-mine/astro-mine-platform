"""Space-filling experiment design over the sampling box (RM-P1-SURR-03; surrogate.md §8).

Turns a :class:`~astro_mine.surrogate.datagen.policy.SamplingPolicy`'s ``parameter_bounds`` box into
a set of excavation-config rows ``(C, P)`` to label — a **Sobol** low-discrepancy sequence, a
**Latin-hypercube** stratified sample (both via ``scipy.stats.qmc``), or a full-factorial **grid**.
All three are deterministic in ``policy.seed``, so the same policy always yields the same design
points (the reproducibility contract). numpy + scipy only — no torch, no Sim.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
from scipy.stats import qmc

from astro_mine.surrogate.datagen.policy import DesignKind, SamplingPolicy

__all__ = ["design_points", "grid_design", "lhs_design", "sobol_design"]

FloatArray = npt.NDArray[np.float64]


def _box(policy: SamplingPolicy) -> tuple[FloatArray, FloatArray]:
    """The ``(lower, upper)`` bound arrays over the policy's ordered parameters."""
    names = policy.param_names
    lower = np.array([policy.parameter_bounds[n].low for n in names], dtype=np.float64)
    upper = np.array([policy.parameter_bounds[n].high for n in names], dtype=np.float64)
    return lower, upper


def _scale(unit: FloatArray, lower: FloatArray, upper: FloatArray) -> FloatArray:
    """Map unit-cube samples into the box (handles a degenerate zero-width dimension)."""
    return np.asarray(lower + unit * (upper - lower), dtype=np.float64)


def sobol_design(policy: SamplingPolicy) -> FloatArray:
    """``n_initial`` Sobol low-discrepancy points over the box ``(n_initial, P)``.

    Scrambled and seeded by ``policy.seed`` for a deterministic, low-discrepancy sweep. scipy warns
    when ``n_initial`` is not a power of two (a balance-property caveat, not an error); the sequence
    is still valid and deterministic, so the warning is suppressed.
    """
    lower, upper = _box(policy)
    sampler = qmc.Sobol(d=len(lower), scramble=True, seed=policy.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unit = sampler.random(policy.n_initial)
    return _scale(unit, lower, upper)


def lhs_design(policy: SamplingPolicy) -> FloatArray:
    """``n_initial`` Latin-hypercube stratified points over the box ``(n_initial, P)``."""
    lower, upper = _box(policy)
    sampler = qmc.LatinHypercube(d=len(lower), seed=policy.seed)
    unit = sampler.random(policy.n_initial)
    return _scale(unit, lower, upper)


def grid_design(policy: SamplingPolicy) -> FloatArray:
    """A full-factorial grid over the box ``(g**P, P)`` with ``g`` points per dimension.

    ``g`` is the per-dimension count that makes the lattice at least ``n_initial`` points
    (``ceil(n_initial ** (1/P))``, floored at 2), so ``grid`` yields ``g**P`` rows — an exhaustive
    small sweep rather than exactly ``n_initial`` points.
    """
    lower, upper = _box(policy)
    p = len(lower)
    g = max(2, int(np.ceil(policy.n_initial ** (1.0 / p))))
    axes = [np.linspace(lo, hi, g) for lo, hi in zip(lower, upper, strict=True)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1).astype(np.float64)


def design_points(policy: SamplingPolicy) -> FloatArray:
    """The design points ``(C, P)`` for ``policy.design`` — the dispatch the sweep calls."""
    if policy.design is DesignKind.SOBOL:
        return sobol_design(policy)
    if policy.design is DesignKind.LHS:
        return lhs_design(policy)
    return grid_design(policy)
