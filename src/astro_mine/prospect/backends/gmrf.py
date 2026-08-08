"""GMRFField — a Gaussian Markov random field (SPDE / precision-matrix) backend (prospect.md §11).

A resource field whose spatial prior is a **Gaussian Markov random field** defined by a *sparse
precision matrix* on the lattice — the SPDE representation of a Matérn field (Lindgren, Rue &
Lindström 2011). Where the grid backend fuses observations cell-by-cell and the GP backend carries
a dense kernel, the GMRF encodes spatial dependence through a **sparse precision** ``Q``, so both
the prior and — crucially — the conditioned posterior stay sparse. That is what makes it the
recommended path for **large lattice domains** (prospect.md §11): conditioning is a sparse linear
solve, and marginal variances are extracted lazily per queried cell from a single factorization
rather than by inverting a dense covariance.

The precision is the **alpha = 2** SPDE operator — the *square* of the κ²-shifted graph Laplacian,
``Q = τ·(κ²I + L)²`` (:func:`_spde_precision`). The exponent is not cosmetic. The SPDE
``(κ² - Δ)^(alpha/2) x = W`` yields a Matérn field of smoothness ``nu = alpha - d/2``, so in 2-D
alpha = 1 gives nu = 0 — whose spectral density is not integrable, i.e. **not a valid Matérn
field**: it under-shoots the requested range badly (correlation at the requested range ~0.03 instead
of ~0.14). alpha = 2 gives nu = 1, a genuine Matérn field whose practical range is the
``correlation_length_m`` the caller asked for. Squaring costs a 13-point stencil instead of a
5-point one; the precision stays sparse.

It implements the Core ``ResourceField`` contract **identically** to the other backends: the
per-point posterior is Gaussian (mean + variance), so quantiles/samples flow through the shared
:mod:`~astro_mine.prospect.backends._gaussian` helper. Its uncertainty representation is therefore
**parametric** (a closed-form Gaussian marginal), which it declares for Zarr tagging
(:attr:`GMRFField.UNCERTAINTY_REPRESENTATION`; RM-P1-PROSPECT-10).

Like a belief field it supports a **replayable, content-addressed** :meth:`update`: appending
observations and re-conditioning is a pure function of ``(prior, ordered log, hyper-parameters)``,
so conditioning incrementally and in one batch produce identical posteriors (the belief-variant
semantics of RM-P0-PROSPECT-04). :meth:`realize` draws a sealed full-field realization from the
posterior (the ground-truth-variant use). Fitting is deterministic under a fixed seed
(conventions.md §1.5).

Backlog: RM-P1-PROSPECT-10 — astro-mine-prospect#20
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import ClassVar

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from astro_mine.core.resource import Position
from astro_mine.core.units import Epoch
from astro_mine.prospect.backends._gaussian import gaussian_quantile, gaussian_samples
from astro_mine.prospect.belief.observation import FieldObservation
from astro_mine.prospect.field.base import BaseResourceField
from astro_mine.prospect.field.metadata import FieldGrid, FieldMetadata

__all__ = ["GMRFField"]

#: The SPDE order this backend realizes, folded into :attr:`GMRFField.content_hash` so that changing
#: it is *visible in the content address* rather than a silent re-definition of every sealed field
#: ever drawn from a GMRF prior. alpha = 2 ⇒ Matérn nu = 1 in 2-D (see the module docstring).
_OPERATOR_TAG = "spde_alpha=2"


def _lattice_laplacian(n_rows: int, n_cols: int) -> sp.csc_matrix:
    """The graph Laplacian ``L = D - A`` of the 4-connected lattice (Neumann boundary).

    ``L`` is sparse, symmetric, positive-semidefinite — the lattice discretization of ``-Δ``. Note
    ``κ²I + L`` is the *factor*, not the precision: the SPDE ``(κ² - Δ)^(alpha/2) x = W`` has
    precision ``(κ²I + L)^alpha``, and this backend uses alpha = 2 (:func:`_spde_precision`). Row
    sums are zero (``L @ 1 = 0``), which keeps the constant-prior right-hand side closed-form.
    """
    n = n_rows * n_cols
    rows: list[int] = []
    cols: list[int] = []
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    rows.append(i)
                    cols.append(rr * n_cols + cc)
    adjacency = sp.coo_matrix(
        (-np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n)
    ).tocsr()
    degree = sp.diags(-np.asarray(adjacency.sum(axis=1)).ravel())
    return (degree + adjacency).tocsc()


def _spde_precision(n_rows: int, n_cols: int, kappa: float) -> sp.csc_matrix:
    """The alpha = 2 SPDE precision ``(κ²I + L)²`` on the lattice (unscaled — τ is applied later).

    The **square** is the whole point. ``(κ² - Δ)^(alpha/2) x = W`` gives a Matérn field of
    smoothness ``nu = alpha - d/2``; in 2-D only alpha = 2 (⇒ nu = 1) is a valid field, and only it
    puts the Matérn *practical range* ``rho = sqrt(8nu)/κ`` where :meth:`GMRFField.from_prior`
    promises it. The unsquared alpha = 1 operator delivers roughly a third of the requested range.

    ``B = κ²I + L`` is symmetric positive-definite (``L`` is PSD, ``κ² > 0``), so ``B²`` is too, and
    it stays sparse: a 13-point stencil (the 5-point Laplacian convolved with itself) rather than
    ``B``'s 5-point one. Since ``L @ 1 = 0`` we have ``B @ 1 = κ²·1`` and hence ``B² @ 1 = κ⁴·1`` —
    the identity :func:`_assemble` relies on for its closed-form constant-prior right-hand side.
    """
    n = n_rows * n_cols
    base = (
        (kappa**2) * sp.identity(n, format="csc", dtype=np.float64)
        + _lattice_laplacian(n_rows, n_cols)
    ).tocsc()
    return (base @ base).tocsc()


def _bilinear(
    grid: FieldGrid, position: Position
) -> tuple[tuple[int, int, int, int], float, float]:
    """The four edge-clamped corner cells and interpolation fractions for ``position``.

    Returns ``((r0, c0, r1, c1), tr, tc)`` — the cell-centered bilinear stencil the mean and the
    (lazily-computed) marginal variances are both interpolated over, so a query at an arbitrary
    position is consistent between mean and variance.
    """
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    fc = min(max((position[0] - grid.min_x_m) / dx - 0.5, 0.0), float(grid.n_cols - 1))
    fr = min(max((position[1] - grid.min_y_m) / dy - 0.5, 0.0), float(grid.n_rows - 1))
    c0 = int(np.floor(fc))
    r0 = int(np.floor(fr))
    return (r0, c0, min(r0 + 1, grid.n_rows - 1), min(c0 + 1, grid.n_cols - 1)), fr - r0, fc - c0


def _nearest_cells(grid: FieldGrid, xy: NDArray[np.float64]) -> NDArray[np.intp]:
    """The flat cell index nearest to each ``(x, y)`` observation (edge-clamped)."""
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    cols = np.clip(((xy[:, 0] - grid.min_x_m) / dx).astype(np.intp), 0, grid.n_cols - 1)
    rows = np.clip(((xy[:, 1] - grid.min_y_m) / dy).astype(np.intp), 0, grid.n_rows - 1)
    return rows * grid.n_cols + cols


def _hutchinson_mean_diag_inverse(factor: object, n: int, *, probes: int, seed: int) -> float:
    """Estimate ``mean(diag(A⁻¹))`` from a factorization via the Hutchinson trace estimator.

    ``trace(A⁻¹) = E[zᵀ A⁻¹ z]`` for ``z`` with unit-variance entries; averaging ``zᵀ (A⁻¹ z)``
    over ``probes`` Rademacher vectors and dividing by ``n`` estimates the mean marginal variance
    with only one sparse solve per probe — so the prior can be scaled to a target variance without
    a dense inverse. Seeded for reproducibility.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(probes):
        z = rng.integers(0, 2, size=n).astype(np.float64) * 2.0 - 1.0  # ±1 Rademacher
        total += float(z @ factor.solve(z))  # type: ignore[attr-defined]
    return total / (probes * n)


class GMRFField(BaseResourceField):
    """A GMRF/SPDE resource field — a sparse precision prior conditioned on scattered observations.

    Construct with :meth:`build` (single-shot) or :meth:`from_prior` + :meth:`update` (the
    replayable belief form). The posterior precision ``Q_post = Q_prior + diag(obs precisions)`` is
    sparse; its ``splu`` factorization is held once and reused for the mean solve and for **lazy,
    exact** per-cell marginal variances ``(Q_post⁻¹)_{ii}``. Queries bilinearly interpolate the
    mean and marginal variance; the per-point posterior is Gaussian, so quantiles/samples are
    parametric.
    """

    #: This backend reports a closed-form Gaussian marginal at each point — a **parametric**
    #: uncertainty representation, recorded in :meth:`zarr_attrs` so a persisted field is
    #: self-describing (RM-P1-PROSPECT-10).
    UNCERTAINTY_REPRESENTATION: ClassVar[str] = "parametric"

    def __init__(
        self,
        metadata: FieldMetadata,
        *,
        base: sp.csc_matrix,
        tau: float,
        kappa: float,
        prior_mean: float,
        prior_variance: float,
        default_noise: float,
        log: tuple[FieldObservation, ...],
    ) -> None:
        grid = metadata.grid
        if grid is None:
            raise ValueError("GMRFField requires metadata.grid (a FieldGrid spatial domain)")
        super().__init__(metadata)
        self._grid = grid
        self._base = base
        self._tau = tau
        self._kappa = kappa
        self._prior_mean = prior_mean
        self._prior_variance = prior_variance
        self._default_noise = default_noise
        self._log = log
        self._var_cache: dict[int, float] = {}
        self._mean, self._precision = _assemble(grid, base, tau, kappa, prior_mean, log)
        self._factor = spla.splu(self._precision)

    @classmethod
    def from_prior(
        cls,
        metadata: FieldMetadata,
        *,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        correlation_length_m: float | None = None,
        default_noise: float = 1e-2,
        marginal_probes: int = 16,
        seed: int = 0,
    ) -> GMRFField:
        """The un-observed GMRF prior: the alpha = 2 SPDE precision scaled to the target variance.

        The precision is ``τ·(κ²I + L)²`` (:func:`_spde_precision`), scaled so
        ``mean(diag(Q_prior⁻¹)) ≈ prior_variance`` via a seeded Hutchinson trace estimate over
        ``marginal_probes`` probes.

        ``correlation_length_m`` is the Matérn **practical range** ``rho`` — the separation at which
        the correlation has decayed to ~0.1 (0.14 for nu = 1, the standard INLA convention) — and
        fixes ``κ = sqrt(8nu)/rho = sqrt(8)/rho`` for the nu = 1 field the alpha = 2 operator
        realizes. It defaults to 15% of the larger grid extent. ``default_noise`` is the
        per-observation measurement variance used by :meth:`update` / :meth:`build` unless an
        observation carries its own ``noise_sigma``.
        """
        grid = metadata.grid
        if grid is None:
            raise ValueError("GMRFField requires metadata.grid (a FieldGrid spatial domain)")
        if prior_variance <= 0.0:
            raise ValueError(f"prior_variance must be positive, got {prior_variance}")
        if default_noise <= 0.0:
            raise ValueError(f"default_noise must be positive, got {default_noise}")
        if marginal_probes <= 0:
            raise ValueError(f"marginal_probes must be positive, got {marginal_probes}")

        extent = max(grid.max_x_m - grid.min_x_m, grid.max_y_m - grid.min_y_m)
        rng_range = 0.15 * extent if correlation_length_m is None else float(correlation_length_m)
        if rng_range <= 0.0:
            raise ValueError(f"correlation_length_m must be positive, got {rng_range}")
        cell = extent / max(grid.n_rows, grid.n_cols)
        kappa = float(np.sqrt(8.0) / (rng_range / cell))

        n = grid.n_rows * grid.n_cols
        base = _spde_precision(grid.n_rows, grid.n_cols, kappa)
        mean_marginal = _hutchinson_mean_diag_inverse(
            spla.splu(base), n, probes=marginal_probes, seed=seed
        )
        tau = mean_marginal / prior_variance
        return cls(
            metadata,
            base=base,
            tau=tau,
            kappa=kappa,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            default_noise=default_noise,
            log=(),
        )

    @classmethod
    def build(
        cls,
        metadata: FieldMetadata,
        *,
        train_points: Sequence[Position] | None = None,
        train_values: Sequence[float] | None = None,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        correlation_length_m: float | None = None,
        noise: float = 1e-2,
        marginal_probes: int = 16,
        seed: int = 0,
    ) -> GMRFField:
        """Single-shot: build the GMRF prior and condition it on ``train_points``/``train_values``.

        A convenience wrapper over :meth:`from_prior` + :meth:`update` for the ``make_backend``
        path, mirroring :meth:`GridField.build`. Every observation is a noisy point measurement of
        its nearest cell with variance ``noise``. With no observations the field is the prior.
        """
        field = cls.from_prior(
            metadata,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            correlation_length_m=correlation_length_m,
            default_noise=noise,
            marginal_probes=marginal_probes,
            seed=seed,
        )
        if train_points is None and train_values is None:
            return field
        if train_points is None or train_values is None:
            raise ValueError("train_points and train_values must be provided together, or neither")
        if len(train_points) != len(train_values):
            raise ValueError(
                f"train_points and train_values length mismatch: "
                f"{len(train_points)} vs {len(train_values)}"
            )
        observations = [
            FieldObservation(
                x_m=p[0], y_m=p[1], z_m=p[2], value=float(v), noise_sigma=float(np.sqrt(noise))
            )
            for p, v in zip(train_points, train_values, strict=True)
        ]
        return field.update(observations)

    def update(self, observations: Iterable[FieldObservation]) -> GMRFField:
        """Append ``observations`` and return the re-conditioned GMRF (a new field).

        The new field's log is this one's log followed by ``observations``, in order; the posterior
        re-conditions from the (unchanged) prior over the full log, so incremental and batch
        conditioning are byte-identical — the replay property (RM-P0-PROSPECT-04). Each observation
        contributes precision ``1 / noise_sigma²`` to its nearest cell.
        """
        return GMRFField(
            self._metadata,
            base=self._base,
            tau=self._tau,
            kappa=self._kappa,
            prior_mean=self._prior_mean,
            prior_variance=self._prior_variance,
            default_noise=self._default_noise,
            log=self._log + tuple(observations),
        )

    @property
    def log(self) -> tuple[FieldObservation, ...]:
        """The ordered observation log conditioned into this posterior."""
        return self._log

    def _marginal_variance(self, cell: int) -> float:
        """The exact posterior marginal variance ``(Q_post⁻¹)_{cell,cell}`` (lazily, cached)."""
        cached = self._var_cache.get(cell)
        if cached is not None:
            return cached
        e = np.zeros(self._precision.shape[0], dtype=np.float64)
        e[cell] = 1.0
        value = max(float(self._factor.solve(e)[cell]), 0.0)
        self._var_cache[cell] = value
        return value

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        (r0, c0, r1, c1), tr, tc = _bilinear(self._grid, position)
        top = self._mean[r0, c0] * (1.0 - tc) + self._mean[r0, c1] * tc
        bot = self._mean[r1, c0] * (1.0 - tc) + self._mean[r1, c1] * tc
        return float(top * (1.0 - tr) + bot * tr)

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        (r0, c0, r1, c1), tr, tc = _bilinear(self._grid, position)
        n_cols = self._grid.n_cols

        def var(r: int, c: int) -> float:
            return self._marginal_variance(r * n_cols + c)

        top = var(r0, c0) * (1.0 - tc) + var(r0, c1) * tc
        bot = var(r1, c0) * (1.0 - tc) + var(r1, c1) * tc
        return float(top * (1.0 - tr) + bot * tr)

    def marginal_variance_grid(self) -> NDArray[np.float64]:
        """The full per-cell posterior marginal-variance grid ``(n_rows, n_cols)``.

        Materializes every cell's exact marginal variance from the held factorization — O(N) sparse
        solves. Convenient for diagnostics / information-gain maps at reduced scale; for a very
        large lattice prefer per-position :meth:`variance` queries (which stay lazy).
        """
        n_cols = self._grid.n_cols
        out = np.empty((self._grid.n_rows, n_cols), dtype=np.float64)
        for r in range(self._grid.n_rows):
            for c in range(n_cols):
                out[r, c] = self._marginal_variance(r * n_cols + c)
        return out

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        return gaussian_quantile(self.mean(position), self.variance(position), q)

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        return gaussian_samples(self.mean(position), self.variance(position), n=n, seed=seed)

    def realize(self, *, seed: int) -> NDArray[np.float64]:
        """Draw one sealed full-field realization ``(n_rows, n_cols)`` from the posterior.

        A single deterministic ``N(mean, Q_post⁻¹)`` draw — the ground-truth-variant use of the
        GMRF: ``x = μ + Lᵀ⁻¹ z`` with ``z ~ N(0, I)`` and ``L`` the Cholesky factor of ``Q_post``,
        so ``cov(x) = Q_post⁻¹`` — the covariance of the alpha = 2 (Matérn nu = 1) prior,
        conditioned on the log. The draw therefore carries the *requested* correlation length: on
        the un-observed prior the lattice correlation at the requested range is ~0.14, the nu = 1
        practical-range convention (it was ~0.03 under the alpha = 1 operator this replaces — a
        field far whiter than the caller asked for). That is what makes a sealed truth drawn through
        it genuinely spatially structured. Uses a dense Cholesky (reduced-order; a very large
        lattice would use a sparse Cholesky). Reproducible in ``seed``.
        """
        dense = self._precision.toarray()
        chol = np.linalg.cholesky(dense)  # lower L with L Lᵀ = Q_post
        z = np.random.default_rng(seed).standard_normal(dense.shape[0])
        white = np.linalg.solve(chol.T, z)  # Lᵀ x = z → cov(x) = Q⁻¹
        return (self._mean.ravel() + white).reshape((self._grid.n_rows, self._grid.n_cols))

    @property
    def content_hash(self) -> str:
        """A stable content address over the prior spectrum, hyper-parameters, and the log.

        Two GMRFs with the same prior and the same ordered log share a hash — so replaying a log
        (in any chunking) reproduces the same posterior identity (prospect.md §5). The SPDE order
        (:data:`_OPERATOR_TAG`) is folded in alongside ``(τ, κ)``: the operator *is* part of the
        prior, so a future change of alpha must move the address rather than silently redefine what
        a pinned hash meant.
        """
        digest = hashlib.sha256()
        digest.update(self._metadata.model_dump_json().encode("utf-8"))
        digest.update(
            repr((_OPERATOR_TAG, self._tau, self._kappa, self._prior_mean)).encode("utf-8")
        )
        for observation in self._log:
            digest.update(observation.model_dump_json().encode("utf-8"))
        return digest.hexdigest()

    def zarr_attrs(self) -> dict[str, str]:
        """The self-describing attributes stamped onto a persisted (Zarr) field.

        Records the ``uncertainty_representation`` (here ``"parametric"``) alongside the
        species/unit so a stored or Hub-published field declares how its uncertainty is encoded
        (RM-P1-PROSPECT-10) — never a bare grid of numbers.
        """
        return {
            "uncertainty_representation": self.UNCERTAINTY_REPRESENTATION,
            "backend": "gmrf",
            "species": self.species,
            "unit": self.unit,
        }


def _assemble(
    grid: FieldGrid,
    base: sp.csc_matrix,
    tau: float,
    kappa: float,
    prior_mean: float,
    log: tuple[FieldObservation, ...],
) -> tuple[NDArray[np.float64], sp.csc_matrix]:
    """Condition the GMRF prior on the ordered ``log`` → ``(mean_grid, posterior precision)``.

    ``Q_prior = τ·base`` with ``base = (κ²I + L)²``; each observation adds precision
    ``1/noise_sigma²`` to its nearest cell.

    The prior right-hand side is closed-form because the Laplacian has zero row sums: ``L @ 1 = 0``
    ⇒ ``(κ²I + L) @ 1 = κ²·1`` ⇒ ``(κ²I + L)² @ 1 = κ⁴·1``, so
    ``Q_prior @ (prior_mean·1) = prior_mean·τ·κ⁴·1``. **The fourth power is load-bearing**: it is
    the alpha = 2 operator's exponent, and κ² here (the alpha = 1 form) would silently pull the
    un-observed posterior mean to ``prior_mean/κ²`` instead of ``prior_mean``
    (``test_build_with_no_observations_is_the_prior`` is the guard).

    Returns the posterior mean grid and the sparse posterior precision.
    """
    n = grid.n_rows * grid.n_cols
    q_prior = (tau * base).tocsc()
    diag_add = np.zeros(n, dtype=np.float64)
    rhs = prior_mean * tau * (kappa**4) * np.ones(n, dtype=np.float64)
    if log:
        xy = np.array([[o.x_m, o.y_m] for o in log], dtype=np.float64)
        cell_idx = _nearest_cells(grid, xy)
        precisions = np.array([1.0 / (o.noise_sigma**2) for o in log], dtype=np.float64)
        values = np.array([o.value for o in log], dtype=np.float64)
        np.add.at(diag_add, cell_idx, precisions)
        np.add.at(rhs, cell_idx, precisions * values)
    q_post = (q_prior + sp.diags(diag_add)).tocsc()
    mean_vec = spla.splu(q_post).solve(rhs)
    return mean_vec.reshape((grid.n_rows, grid.n_cols)), q_post
