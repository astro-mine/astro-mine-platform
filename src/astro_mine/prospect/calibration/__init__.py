"""Calibration gate — credible-interval coverage checked in CI (prospect.md §10, §12; LUNAR-DR-005).

A belief field's stated uncertainty must be **honest**, not decorative: an over-confident field is a
credibility hazard that silently breaks every downstream active-perception result (prospect.md §9).
This module is the gate that guards against shipping one. It measures the **credible-interval
coverage** of a :class:`~astro_mine.core.resource.ResourceField` against **held-out ground truth** —
the fraction of held-out true values that fall inside the belief's central credible interval at each
nominal level — and fails when that empirical coverage strays from nominal beyond a documented
budget (``LUNAR-DR-005``: "belief fields carry full uncertainty (calibration checked in CI)").

Three public pieces:

- :mod:`~astro_mine.prospect.calibration.geostats` — the **geostatistical sanity** checks that
  prospect.md §10 names alongside coverage. :func:`empirical_variogram` / :func:`fit_variogram`
  recover a field's correlation length from a scattered sample (does the model think ice is as
  clumpy as it really is?), and :func:`loo_cross_validation` scores a backend's honest
  out-of-sample kriging error. They answer what coverage on a single fit cannot.
- :func:`check_calibration` — the pure scoring gate. It reads the belief only through the Core
  ``quantile`` contract (never assuming Gaussianity), so it stays valid for the P1 GP/GMRF backends.
  It returns a :class:`CalibrationReport` whose ``(levels, coverage)`` pairs *are* the reliability
  diagram (prospect.md §10) and whose ``passed`` flag is the gate verdict.
- :func:`build_calibration_case` — a seeded, reproducible held-out split. It draws a sealed
  :class:`~astro_mine.prospect.belief.ground_truth.GroundTruthField`, partitions the grid cells into
  a disjoint **training** and **held-out** set, conditions a belief on noisy observations of the
  training cells, and reveals the true values at the held-out cells. The same arguments always yield
  the same ``(belief, held_out)`` pair, so the gate is reproducible (the acceptance criterion that
  the held-out split be "documented and reproducible").

Reading the sealed truth is **capability-gated** (RM-P0-PROSPECT-05): :func:`build_calibration_case`
must be handed the Core ``GROUND_TRUTH_ACCESS`` capability — the calibration harness is exactly the
privileged, non-agent holder named in ``ground_truth.py`` (prospect.md §9). The belief it returns is
agent-safe (it carries no ground-truth handle); the held-out truth values it returns are a
privileged artifact and must never be exposed to a policy.

Backlog: RM-P0-PROSPECT-07 — https://github.com/astro-mine/astro-mine-prospect/issues/7
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.resource import Position, ResourceField
from astro_mine.core.sadf.enums import CapabilityTag
from astro_mine.prospect.belief.field import BeliefField
from astro_mine.prospect.belief.ground_truth import sample_ground_truth
from astro_mine.prospect.calibration.geostats import (
    LooReport,
    VariogramFit,
    empirical_variogram,
    fit_variogram,
    loo_cross_validation,
)
from astro_mine.prospect.field.metadata import FieldGrid
from astro_mine.prospect.priors.recipe import Prior

__all__ = [
    "DEFAULT_COVERAGE_TOLERANCE",
    "DEFAULT_LEVELS",
    "CalibrationReport",
    "HeldOutTruth",
    "LooReport",
    "VariogramFit",
    "build_calibration_case",
    "check_calibration",
    "empirical_variogram",
    "fit_variogram",
    "loo_cross_validation",
]

#: The nominal central credible-interval probabilities the gate scores coverage at. A symmetric
#: interval at level ``p`` spans the ``[(1-p)/2, (1+p)/2]`` quantiles; a well-calibrated belief
#: contains the truth at a rate close to ``p`` for every ``p``. The spread (median to the tails) is
#: what makes the reliability diagram informative — over-confidence shows first at the high levels.
DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.99)

#: The coverage error budget: the gate fails if empirical coverage deviates from nominal by more
#: than this at any tested level (a two-sided check, catching both over- and under-confidence). The
#: value is set well above the binomial sampling noise of a few-hundred-point held-out set
#: (``sqrt(p(1-p)/n) ~ 0.02`` at ``n = 400``) yet far below the coverage collapse of a genuinely
#: over-confident field (whose high-level coverage falls toward zero) — so a calibrated reference
#: belief passes with margin while a deliberately over-confident one fails decisively.
DEFAULT_COVERAGE_TOLERANCE = 0.1


@dataclass(frozen=True, eq=False)
class HeldOutTruth:
    """Held-out true field values at positions disjoint from the belief's observations.

    The calibration oracle: ``values[i]`` is the sealed ground truth at ``positions[i]``, read
    through the ``GROUND_TRUTH_ACCESS`` gate by :func:`build_calibration_case`. This is a
    **privileged artifact** — it carries revealed truth and must never be handed to a policy. It is
    not itself a sealed field object, so :func:`~astro_mine.prospect.isolation.assert_isolated` does
    not flag it; keeping it out of agent reach is the harness's responsibility (prospect.md §9).
    """

    positions: tuple[Position, ...]
    values: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class CalibrationReport:
    """The coverage scorecard for one belief against one held-out set — the gate verdict.

    ``coverage[i]`` is the empirical coverage of the ``levels[i]`` central credible interval over
    the ``n`` held-out points; ``(levels, coverage)`` together are the reliability diagram
    (prospect.md §10). ``max_deviation`` is the worst ``|nominal - empirical|`` across levels, and
    ``passed`` is ``max_deviation <= tolerance`` — the CI gate.
    """

    levels: tuple[float, ...]
    coverage: tuple[float, ...]
    n: int
    tolerance: float
    max_deviation: float
    passed: bool

    @property
    def reliability(self) -> tuple[tuple[float, float], ...]:
        """The reliability diagram as ``(nominal, empirical)`` coverage pairs."""
        return tuple(zip(self.levels, self.coverage, strict=True))


def check_calibration(
    belief: ResourceField,
    held_out: HeldOutTruth,
    *,
    levels: Sequence[float] = DEFAULT_LEVELS,
    tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
) -> CalibrationReport:
    """Score the credible-interval coverage of ``belief`` against ``held_out`` — the gate.

    For each nominal level ``p`` in ``levels``, forms the central credible interval
    ``[belief.quantile(pos, (1-p)/2), belief.quantile(pos, (1+p)/2)]`` at every held-out position
    and counts how often the held-out true value falls inside. The result is a
    :class:`CalibrationReport` whose ``passed`` flag is ``True`` iff the empirical coverage is
    within ``tolerance`` of nominal at every level. Reads the belief only through the Core
    ``quantile`` contract — no Gaussian assumption — so it works for any ``ResourceField`` backend.

    Raises ``ValueError`` on a non-positive ``tolerance``, an empty ``levels``, a level outside the
    open interval ``(0, 1)``, an empty ``held_out``, or a values/positions length mismatch.
    """
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    levels_t = tuple(float(p) for p in levels)
    if not levels_t:
        raise ValueError("levels must name at least one credible-interval probability")
    if any(not (0.0 < p < 1.0) for p in levels_t):
        raise ValueError(f"every level must lie in the open interval (0, 1); got {levels_t}")
    n = len(held_out)
    if n == 0:
        raise ValueError("held_out is empty; coverage needs at least one held-out point")
    truths = np.asarray(held_out.values, dtype=np.float64)
    if truths.shape != (n,):
        raise ValueError(
            f"held_out.values must be 1-D with one value per position ({n}); got {truths.shape}"
        )

    coverage: list[float] = []
    for p in levels_t:
        lo_q = (1.0 - p) / 2.0
        hi_q = (1.0 + p) / 2.0
        lo = np.array([belief.quantile(pos, lo_q) for pos in held_out.positions])
        hi = np.array([belief.quantile(pos, hi_q) for pos in held_out.positions])
        inside = (truths >= lo) & (truths <= hi)
        coverage.append(float(np.count_nonzero(inside)) / n)
    coverage_t = tuple(coverage)
    max_deviation = max(abs(p - c) for p, c in zip(levels_t, coverage_t, strict=True))
    return CalibrationReport(
        levels=levels_t,
        coverage=coverage_t,
        n=n,
        tolerance=float(tolerance),
        max_deviation=max_deviation,
        passed=max_deviation <= tolerance,
    )


def build_calibration_case(
    prior: Prior,
    *,
    seed: int,
    n_train: int,
    n_holdout: int,
    noise_sigma: float,
    capabilities: Iterable[CapabilityTag],
    length_scale: float | None = None,
) -> tuple[BeliefField, HeldOutTruth]:
    """Build a seeded, reproducible ``(belief, held_out)`` calibration case from ``prior`` (gated).

    Draws a sealed ground truth from ``prior`` under ``seed``, deterministically partitions the grid
    cells into a disjoint training set (``n_train`` cells) and held-out set (``n_holdout`` cells),
    conditions a :class:`~astro_mine.prospect.belief.field.BeliefField` on noisy observations of the
    training cells, and reveals the true values at the held-out cells. The same arguments always
    reproduce the same pair (the held-out split is a pure function of ``seed`` and the grid), so the
    calibration gate is reproducible.

    Reading the sealed truth is **capability-gated**: ``capabilities`` must carry
    ``GROUND_TRUTH_ACCESS`` (the calibration harness is the privileged, non-agent holder named in
    prospect.md §9), else :class:`~astro_mine.prospect.isolation.IsolationError` is raised before
    any truth is read. The returned belief carries no ground-truth handle (it is agent-safe); the
    returned :class:`HeldOutTruth` is a privileged artifact.
    """
    grid = prior.metadata.grid
    assert grid is not None  # a Prior always carries its grid (Prior.__init__ enforces it)
    if n_train < 0 or n_holdout <= 0:
        raise ValueError(
            f"need n_holdout > 0 and n_train >= 0; got n_train={n_train}, n_holdout={n_holdout}"
        )
    n_cells = grid.n_rows * grid.n_cols
    if n_train + n_holdout > n_cells:
        raise ValueError(
            f"n_train + n_holdout ({n_train + n_holdout}) exceeds the grid's {n_cells} cells"
        )

    truth = sample_ground_truth(prior, seed=seed, capabilities=capabilities)
    # Decorrelate the split permutation from the observation noise with two child seeds, so both are
    # a deterministic function of `seed` (reproducible) yet independent of each other.
    split_seed, obs_seed = (int(s) for s in np.random.SeedSequence(seed).generate_state(2))
    permutation = np.random.default_rng(split_seed).permutation(n_cells)
    train_cells = permutation[:n_train]
    held_out_cells = permutation[n_train : n_train + n_holdout]

    centers = _cell_centers(grid)  # (n_cells, 3); row-major, matching realization.reshape(-1)
    train_positions: list[Position] = [_as_position(centers[int(i)]) for i in train_cells]
    observations = truth.observe(
        train_positions, noise_sigma=noise_sigma, seed=obs_seed, capabilities=capabilities
    )
    belief = BeliefField.from_prior(prior, length_scale=length_scale).update(observations)

    flat_truth = truth.reveal(capabilities=capabilities).reshape(-1)
    held_positions = tuple(_as_position(centers[int(i)]) for i in held_out_cells)
    held_values = np.ascontiguousarray(flat_truth[held_out_cells], dtype=np.float64)
    held_values.flags.writeable = False
    return belief, HeldOutTruth(positions=held_positions, values=held_values)


def _cell_centers(grid: FieldGrid) -> NDArray[np.float64]:
    """The ``(n_rows*n_cols, 3)`` row-major cell-center positions (flat index ``row*n_cols+col``).

    Matches the grid convention used by the belief/ground-truth backends, so a cell's center queries
    that exact cell (``GridField`` is cell-centered) and ``flat[row*n_cols + col]`` of a realization
    reshaped row-major is that cell's value — keeping held-out truth and belief query aligned.
    """
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    xs = grid.min_x_m + (np.arange(grid.n_cols) + 0.5) * dx
    ys = grid.min_y_m + (np.arange(grid.n_rows) + 0.5) * dy
    gx, gy = np.meshgrid(xs, ys)  # (n_rows, n_cols)
    centers = np.zeros((grid.n_rows * grid.n_cols, 3), dtype=np.float64)
    centers[:, 0] = gx.reshape(-1)
    centers[:, 1] = gy.reshape(-1)
    return centers


def _as_position(center: NDArray[np.float64]) -> Position:
    """A single ``(x, y, z)`` row of :func:`_cell_centers` as a Core :data:`Position`."""
    return (float(center[0]), float(center[1]), float(center[2]))
