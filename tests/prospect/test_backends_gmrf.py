"""RM-P1-PROSPECT-10 — the GMRF/SPDE sparse-precision backend behind the ResourceField contract.

Proves the deliverable: a Gaussian Markov random field for large lattice domains implements the
full uncertainty-first Core contract identically to the P0 backends, conditions on scattered
observations via a sparse precision solve with exact lazy marginal variances, supports the
replayable/content-addressed belief ``update`` and a ground-truth ``realize`` draw, passes the
calibration gate, and registers behind the ``make_backend`` field-backends extension point with no
Core change — declaring its (parametric) uncertainty representation for Zarr tagging.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import kv

from astro_mine.core.resource import FieldDistribution, check_resource_field
from astro_mine.core.units import MOON, MOON_BODY_FIXED, PlanetaryCRS
from astro_mine.prospect.backends import GMRFField, make_backend
from astro_mine.prospect.backends.gmrf import (
    _hutchinson_mean_diag_inverse,
    _lattice_laplacian,
    _spde_precision,
)
from astro_mine.prospect.belief.observation import FieldObservation
from astro_mine.prospect.calibration import DEFAULT_LEVELS, HeldOutTruth, check_calibration
from astro_mine.prospect.field import FieldGrid, FieldMetadata, Position

_CRS = PlanetaryCRS(
    body=MOON,
    body_fixed_frame="MOON_ME",
    reference_radius_m=1_737_400.0,
    projection="+proj=stere +lat_0=-90 +R=1737400",
)
_GRID = FieldGrid(
    min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=16, n_cols=16
)
_TRAIN_POINTS: list[Position] = [(0.0, 0.0, 0.0), (-600.0, -600.0, 0.0), (600.0, 600.0, 0.0)]
_TRAIN_VALUES = [0.6, 0.1, 0.1]
_CENTER: Position = (0.0, 0.0, 0.0)
_CORNER: Position = (-950.0, -950.0, 0.0)


def _metadata(grid: FieldGrid | None = _GRID) -> FieldMetadata:
    return FieldMetadata(
        species="water_equivalent_hydrogen",
        unit="mass_fraction",
        frame=MOON_BODY_FIXED,
        crs=_CRS,
        grid=grid,
    )


def _field(**kw: object) -> GMRFField:
    cfg: dict[str, object] = dict(
        prior_mean=0.1, prior_variance=0.25, correlation_length_m=300.0, noise=0.01, seed=0
    )
    cfg.update(kw)
    return GMRFField.build(
        _metadata(), train_points=_TRAIN_POINTS, train_values=_TRAIN_VALUES, **cfg
    )


# --- the Core contract ------------------------------------------------------------------------


def test_satisfies_resource_field_contract() -> None:
    check_resource_field(_field())


def test_posterior_is_uncertainty_first() -> None:
    post = _field().posterior(_CENTER)
    assert isinstance(post, FieldDistribution)
    assert post.variance > 0.0
    assert post.species == "water_equivalent_hydrogen"


def test_prior_marginal_variance_matches_target() -> None:
    # The Hutchinson-scaled precision reproduces the requested mean prior variance.
    prior = GMRFField.from_prior(_metadata(), prior_mean=0.1, prior_variance=0.25, seed=0)
    assert float(prior.marginal_variance_grid().mean()) == pytest.approx(0.25, rel=0.1)


def test_conditioning_tracks_data_and_shrinks_variance() -> None:
    field = _field()
    prior = GMRFField.from_prior(
        _metadata(), prior_mean=0.1, prior_variance=0.25, correlation_length_m=300.0, seed=0
    )
    # Near the hot observation the mean rises toward it and the variance drops below the prior.
    assert field.mean(_CENTER) > 0.2
    assert field.variance(_CENTER) < prior.variance(_CENTER)
    # Far from every observation the posterior reverts toward the prior mean.
    assert field.mean(_CORNER) == pytest.approx(0.1, abs=0.05)


def test_quantiles_ordered_and_samples_seeded() -> None:
    field = _field()
    assert (
        field.quantile(_CENTER, 0.05) < field.quantile(_CENTER, 0.5) < field.quantile(_CENTER, 0.95)
    )
    assert field.sample(_CENTER, n=3, seed=7) == field.sample(_CENTER, n=3, seed=7)
    assert field.sample(_CENTER, n=0) == ()


# --- replayable / content-addressed belief semantics -----------------------------------------


def _observations() -> list[FieldObservation]:
    return [
        FieldObservation(x_m=p[0], y_m=p[1], z_m=p[2], value=v, noise_sigma=0.1)
        for p, v in zip(_TRAIN_POINTS, _TRAIN_VALUES, strict=True)
    ]


def test_update_is_replayable_and_content_addressed() -> None:
    prior = GMRFField.from_prior(
        _metadata(), prior_mean=0.1, prior_variance=0.25, correlation_length_m=300.0, seed=0
    )
    obs = _observations()
    batch = prior.update(obs)
    incremental = prior.update(obs[:1]).update(obs[1:])
    assert batch.content_hash == incremental.content_hash
    assert batch.mean(_CENTER) == pytest.approx(incremental.mean(_CENTER))
    assert np.allclose(batch.marginal_variance_grid(), incremental.marginal_variance_grid())
    assert len(batch.log) == len(obs)
    # A different log ⇒ a different content address.
    assert prior.content_hash != batch.content_hash


def test_build_equals_from_prior_then_update() -> None:
    built = _field(noise=0.01)
    manual = GMRFField.from_prior(
        _metadata(),
        prior_mean=0.1,
        prior_variance=0.25,
        correlation_length_m=300.0,
        default_noise=0.01,
        seed=0,
    ).update(
        [
            FieldObservation(
                x_m=p[0], y_m=p[1], z_m=p[2], value=v, noise_sigma=float(np.sqrt(0.01))
            )
            for p, v in zip(_TRAIN_POINTS, _TRAIN_VALUES, strict=True)
        ]
    )
    assert built.mean(_CENTER) == pytest.approx(manual.mean(_CENTER))


def test_build_with_no_observations_is_the_prior() -> None:
    # This is the **kappa^4 right-hand-side guard**. `_assemble` solves the prior mean in closed
    # form from `Q_prior @ 1 = tau * kappa**(2*alpha) * 1`, which the alpha = 2 operator makes a
    # *fourth* power. Leave the alpha = 1 exponent (kappa**2) in place while squaring the operator
    # and the un-observed posterior mean silently relaxes to `prior_mean / kappa**2` rather than
    # `prior_mean` — a bug visible nowhere else in the suite. The 1e-9 tolerance is what bites.
    field = GMRFField.build(_metadata(), prior_mean=0.2, prior_variance=0.25, seed=0)
    assert field.log == ()
    assert field.mean(_CENTER) == pytest.approx(0.2, abs=1e-9)


def test_realize_is_a_seeded_full_field_draw() -> None:
    field = _field()
    r1 = field.realize(seed=5)
    assert r1.shape == (_GRID.n_rows, _GRID.n_cols)
    assert np.array_equal(r1, field.realize(seed=5))
    assert not np.array_equal(r1, field.realize(seed=6))


# --- the calibration gate ---------------------------------------------------------------------


def test_prior_field_is_calibrated() -> None:
    # Ground truth drawn from N(prior_mean, prior_variance); the GMRF prior's credible intervals
    # cover it at the nominal rate (the calibration gate reads only through `quantile`).
    grid = FieldGrid(
        min_x_m=-5_000.0, min_y_m=-5_000.0, max_x_m=5_000.0, max_y_m=5_000.0, n_rows=28, n_cols=28
    )
    md = _metadata(grid)
    m0, v0 = 0.3, 0.04
    truth = m0 + np.sqrt(v0) * np.random.default_rng(3).standard_normal((grid.n_rows, grid.n_cols))
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    positions = tuple(
        (grid.min_x_m + (c + 0.5) * dx, grid.min_y_m + (r + 0.5) * dy, 0.0)
        for r in range(grid.n_rows)
        for c in range(grid.n_cols)
    )
    held_out = HeldOutTruth(positions=positions, values=truth.ravel())
    field = GMRFField.from_prior(md, prior_mean=m0, prior_variance=v0, seed=0)
    report = check_calibration(field, held_out, levels=DEFAULT_LEVELS)
    assert report.passed, report.reliability


# --- self-describing / registration -----------------------------------------------------------


def test_zarr_attrs_declares_parametric_representation() -> None:
    attrs = _field().zarr_attrs()
    assert attrs["uncertainty_representation"] == "parametric"
    assert attrs["backend"] == "gmrf"
    assert attrs["species"] == "water_equivalent_hydrogen"


def test_make_backend_routes_to_gmrf() -> None:
    field = make_backend(
        "gmrf",
        _metadata(),
        train_points=_TRAIN_POINTS,
        train_values=_TRAIN_VALUES,
        prior_mean=0.1,
        prior_variance=0.25,
        correlation_length_m=300.0,
        seed=0,
    )
    assert isinstance(field, GMRFField)


# --- error paths ------------------------------------------------------------------------------


def test_missing_grid_rejected() -> None:
    with pytest.raises(ValueError, match=r"requires metadata\.grid"):
        GMRFField.from_prior(_metadata(grid=None))


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"prior_variance": 0.0}, "prior_variance must be positive"),
        ({"default_noise": 0.0}, "default_noise must be positive"),
        ({"marginal_probes": 0}, "marginal_probes must be positive"),
        ({"correlation_length_m": -1.0}, "correlation_length_m must be positive"),
    ],
)
def test_bad_hyperparameters_rejected(kw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GMRFField.from_prior(_metadata(), **kw)


def test_build_half_specified_observations_rejected() -> None:
    with pytest.raises(ValueError, match="provided together"):
        GMRFField.build(_metadata(), train_points=_TRAIN_POINTS)
    with pytest.raises(ValueError, match="length mismatch"):
        GMRFField.build(_metadata(), train_points=_TRAIN_POINTS, train_values=[0.1])


# --- internals: the SPDE precision + trace estimator ------------------------------------------


def test_lattice_laplacian_rows_sum_to_zero() -> None:
    lap = _lattice_laplacian(4, 5).toarray()
    assert np.allclose(lap.sum(axis=1), 0.0)  # graph Laplacian: zero row sums
    assert np.allclose(lap, lap.T)  # symmetric


# --- the SPDE operator is alpha = 2, and the field it defines is Matern nu = 1 -------------------
#
# These are *exact*: the correlation structure is read off a dense inverse of the precision, not
# estimated from draws, so there is no tolerance to game and no seed to get lucky on.

#: The Matern practical range, in cells, the tests below pin the operator against.
_RANGE_CELLS = 3
#: kappa = sqrt(8 nu)/rho with nu = 1 — the smoothness the alpha = 2 operator actually realizes.
_KAPPA = float(np.sqrt(8.0)) / _RANGE_CELLS


def _lattice_correlation(n: int, kappa: float) -> np.ndarray:
    """The exact correlation matrix of the SPDE prior on an ``n x n`` lattice."""
    covariance = np.linalg.inv(_spde_precision(n, n, kappa).toarray())
    sd = np.sqrt(np.diag(covariance))
    return np.asarray(covariance / np.outer(sd, sd))


def _mean_correlation_at(corr: np.ndarray, n: int, lag: int) -> float:
    """The mean correlation between cells ``lag`` apart (averaged over rows and columns)."""
    idx = np.arange(n * n).reshape(n, n)
    across = [corr[idx[r, c], idx[r, c + lag]] for r in range(n) for c in range(n - lag)]
    down = [corr[idx[r, c], idx[r + lag, c]] for r in range(n - lag) for c in range(n)]
    return 0.5 * (float(np.mean(across)) + float(np.mean(down)))


def test_the_spde_operator_is_the_alpha_2_square() -> None:
    """The precision is ``(kappa^2 I + L)^2`` — a structural guard nobody can silently un-square.

    ``(kappa^2 - Laplacian)^(alpha/2) x = W`` yields a Matern field of smoothness
    ``nu = alpha - d/2``, so in 2-D alpha = 1 gives nu = 0: a spectral density that is not
    integrable, i.e. not a valid field at all. Only the *square* is.
    """
    n_rows, n_cols, n = 6, 7, 42
    base = (_KAPPA**2) * np.eye(n) + _lattice_laplacian(n_rows, n_cols).toarray()
    q = _spde_precision(n_rows, n_cols, _KAPPA).toarray()

    np.testing.assert_allclose(q, base @ base, atol=1e-12)  # the square, not the base
    np.testing.assert_allclose(q, q.T, atol=1e-12)  # symmetric
    assert float(np.linalg.eigvalsh(q).min()) > 0.0  # and positive definite, so it is a precision

    # Squaring the 5-point Laplacian stencil convolves it with itself into a **13-point** one. That
    # count is the cheapest tripwire there is on the operator's order: an interior row of the
    # alpha=1 operator carries 5 non-zeros, of the alpha=2 operator 13. Un-square it and this fails.
    interior = (n_rows // 2) * n_cols + n_cols // 2
    assert np.count_nonzero(base[interior]) == 5
    assert np.count_nonzero(q[interior]) == 13


def test_the_lattice_correlation_matches_matern_nu1() -> None:
    """The realized lattice correlation is the Matern nu = 1 one, checked against the continuum.

    The continuum Matern nu = 1 correlation is ``C(r) = kappa*r * K_1(kappa*r)``. The 5-point
    stencil is a discretization of it, so the lattice sits a few percent below the continuum — but
    on the same curve, which is what "the backend is a Matern field" has to mean.
    """
    corr = _lattice_correlation(12, _KAPPA)
    lag1 = _mean_correlation_at(corr, 12, 1)

    continuum = float(_KAPPA * kv(1, _KAPPA))
    assert continuum == pytest.approx(0.6263, abs=1e-3)  # kappa*K_1(kappa) at one cell
    assert lag1 == pytest.approx(0.599, abs=0.03)  # the exact 12x12 lattice value
    assert abs(lag1 - continuum) < 0.10 * continuum  # the stencil's ~7% deficit, and no more


def test_the_gmrf_realizes_the_requested_range() -> None:
    """**Acceptance criterion 1** — the field is correlated over the range the caller asked for.

    ``correlation_length_m`` is the Matern *practical range* ``rho``, at which correlation has
    decayed to ~0.14 (nu = 1; the INLA convention ``rho = sqrt(8 nu)/kappa``). The alpha = 2
    operator hits it: 0.140 against the continuum's 0.1397. **The alpha = 1 operator this replaces
    gives 0.033 there** — it structures the field over roughly a *third* of the requested length,
    which is the bug in its cleanest form. This test is the one that would have caught it.
    """
    corr = _lattice_correlation(12, _KAPPA)
    at_range = _mean_correlation_at(corr, 12, _RANGE_CELLS)

    continuum = float(_RANGE_CELLS * _KAPPA * kv(1, _RANGE_CELLS * _KAPPA))
    assert continuum == pytest.approx(
        0.1397, abs=1e-3
    )  # sqrt(8)*K_1(sqrt(8)) — the ~0.14 convention
    assert at_range == pytest.approx(0.140, abs=0.03)


def test_hutchinson_estimator_recovers_mean_diag_inverse() -> None:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    mat = sp.diags([2.0, 4.0, 8.0, 16.0]).tocsc()  # diagonal ⇒ exact mean(diag inverse) known
    exact = float(np.mean([1 / 2, 1 / 4, 1 / 8, 1 / 16]))
    est = _hutchinson_mean_diag_inverse(spla.splu(mat), 4, probes=200, seed=0)
    assert est == pytest.approx(exact, rel=0.05)
