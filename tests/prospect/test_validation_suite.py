"""The validation suite prospect.md §10 calls for — and which did not exist (conventions.md §11).

Three families, all of which §10 names explicitly and none of which was implemented:

- **Cross-backend agreement** — "GP, GMRF, and grid backends agree (within stated tolerance) on a
  shared synthetic problem, so a backend swap is observable and bounded". Without this, the
  ``ResourceField`` contract guarantees the *shape* of a backend's answer but nothing about its
  *value*, and swapping a backend could silently move every downstream result.
- **Geostatistical sanity** — "recovered variograms/length-scales match the generating model on
  synthetic data; kriging cross-validation (leave-one-out) error within bounds". A field that
  recovers the wrong correlation length will send the swarm to the wrong place next, however
  well-calibrated its variance is.
- **Property-based invariants (Hypothesis)** — the four §10 names: conditioning never increases
  variance at a noiselessly-observed point; the posterior reduces to the prior under no
  observations; quantiles are monotone; units/CRS are preserved across updates.

The shared synthetic problem is one smooth bump sampled at scattered points — deliberately something
*every* backend should get right, so a disagreement means a bug, not a modeling difference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from astro_mine.core.resource import Position
from astro_mine.core.units import MOON, MOON_BODY_FIXED, PlanetaryCRS
from astro_mine.prospect.backends import GridField, make_backend
from astro_mine.prospect.backends._random_field import _PRACTICAL_RANGE_PER_SIGMA
from astro_mine.prospect.belief import BeliefField, FieldObservation
from astro_mine.prospect.calibration import (
    empirical_variogram,
    fit_variogram,
    loo_cross_validation,
)
from astro_mine.prospect.field import DEFAULT_QUANTILES, FieldGrid, FieldMetadata
from astro_mine.prospect.priors import SPECIES, UNIT, load_prior
from astro_mine.prospect.priors.recipe import Prior

_CRS = PlanetaryCRS(
    body=MOON,
    body_fixed_frame="MOON_ME",
    reference_radius_m=1_737_400.0,
    projection="+proj=stere +lat_0=-90 +R=1737400",
)
_GRID = FieldGrid(
    min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=24, n_cols=24
)
_METADATA = FieldMetadata(species=SPECIES, unit=UNIT, frame=MOON_BODY_FIXED, crs=_CRS, grid=_GRID)

#: The shared synthetic problem: a single **broad, smooth** Gaussian bump of known length scale,
#: densely sampled at scattered points. It is deliberately a field every backend can represent — a
#: dense GP, a sparse-precision GMRF, and an RBF measurement-fusion grid alike — so that a
#: disagreement here means a *bug*, not a modeling difference. (Each backend is given its natural
#: hyper-parameter for this problem; handing a smoother a correlation length far shorter than the
#: field's own would make it interpolate poorly between samples, and would be testing the
#: parameterization rather than the backend.)
_BUMP_LENGTH_M = 800.0
_BUMP_PEAK = 0.5
_BASELINE = 0.05
_SAMPLE_N = 200


def _truth_at(x: float, y: float) -> float:
    return _BASELINE + _BUMP_PEAK * math.exp(-(x * x + y * y) / (2.0 * _BUMP_LENGTH_M**2))


def _sample(n: int = _SAMPLE_N, seed: int = 0) -> tuple[list[Position], list[float]]:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-950.0, 950.0, size=(n, 2))
    points: list[Position] = [(float(x), float(y), 0.0) for x, y in xy]
    values = [_truth_at(p[0], p[1]) for p in points]
    return points, values


#: Held-out probes the backends are compared at — inside the domain, away from the sample points.
_PROBES: list[Position] = [
    (0.0, 0.0, 0.0),
    (300.0, -200.0, 0.0),
    (-450.0, 450.0, 0.0),
    (700.0, 100.0, 0.0),
    (-800.0, -300.0, 0.0),
]

#: Small, fast, deterministic GP fit knobs (matching the repo's existing backend tests).
_GP_CFG = {"n_iter": 100, "n_inducing": 16, "seed": 0}
_GRID_CFG = {"prior_mean": _BASELINE, "prior_variance": 0.05, "length_scale": 250.0, "noise": 1e-4}
_GMRF_CFG = {
    "prior_mean": _BASELINE,
    "prior_variance": 0.05,
    "correlation_length_m": _BUMP_LENGTH_M,
    "noise": 1e-4,
}


def _backends() -> dict[str, object]:
    points, values = _sample()
    return {
        "grid": make_backend(
            "grid", _METADATA, train_points=points, train_values=values, **_GRID_CFG
        ),
        "gmrf": make_backend(
            "gmrf", _METADATA, train_points=points, train_values=values, **_GMRF_CFG
        ),
        "gp": make_backend("gp", _METADATA, train_points=points, train_values=values, **_GP_CFG),
    }


# --- cross-backend agreement (prospect.md §10) ---------------------------------------------------


#: **The stated tolerance.** The three backends must agree on the posterior mean to within 20% of
#: the field's dynamic range (peak minus baseline) at every held-out probe. On the problem above the
#: realized worst-case spread is about 12% of the range, so the bound holds with margin while still
#: being tight enough that a backend which had drifted — or had silently stopped conditioning on its
#: data — would break it. This is what makes a backend swap "observable and bounded" (§10): the
#: contract guarantees the *shape* of a backend's answer, and this guarantees its *value*.
_AGREEMENT_TOLERANCE = 0.20 * _BUMP_PEAK


def test_gp_gmrf_and_grid_agree_on_a_shared_synthetic_problem() -> None:
    fields = _backends()

    for probe in _PROBES:
        estimates = {name: field.mean(probe) for name, field in fields.items()}  # type: ignore[attr-defined]
        spread = max(estimates.values()) - min(estimates.values())
        assert spread <= _AGREEMENT_TOLERANCE, f"backends disagree at {probe}: {estimates}"

        # And they agree with the *generating truth*, not merely with each other — three backends
        # sharing one bug, or all three quietly reverting to the prior mean, would otherwise pass.
        for name, estimate in estimates.items():
            error = abs(estimate - _truth_at(probe[0], probe[1]))
            assert error <= _AGREEMENT_TOLERANCE, f"{name} is off the truth at {probe}: {error}"


def test_every_backend_is_uncertainty_first_and_shrinks_variance_toward_the_data() -> None:
    points, _values = _sample()
    near = points[0]  # a sampled location
    far = (950.0, -950.0, 0.0)  # the far corner, away from every sample

    for name, field in _backends().items():
        assert field.variance(near) > 0.0, name  # type: ignore[attr-defined]
        # Data reduces uncertainty locally — the property every info-gain map depends on.
        assert field.variance(near) < field.variance(far), name  # type: ignore[attr-defined]


# --- geostatistical sanity: variogram / length-scale recovery (prospect.md §10) -------------------


#: The synthetic field is Gaussian-smoothed white noise with kernel std ``L``, whose
#: autocorrelation is ``exp(-h**2 / (4 L**2))``. The fitted model's correlation function is
#: ``exp(-h**2 / ell**2)``, so the length scale the estimator *should* recover is ``ell = 2 L``. The
#: kernel is chosen short relative to the domain so the empirical variogram fully saturates within
#: the sampled lags — a correlation length longer than the largest observed lag is not identifiable
#: from the data at all, and a test that pretended otherwise would be fitting noise.
#:
#: ``correlated_standard_normal`` is *asked* for a **practical range** (the separation at which
#: correlation reaches ~0.1 — the convention it shares with the GMRF backend), not for a kernel std,
#: so the range that produces a kernel of std ``L`` is ``3.035 * L``. The generated field — and so
#: every assertion below — is byte-for-byte the one this test always used; only the parameterization
#: of the request changed.
_TRUE_KERNEL_M = 100.0
_TRUE_RANGE_M = _PRACTICAL_RANGE_PER_SIGMA * _TRUE_KERNEL_M
_EXPECTED_RANGE_M = 2.0 * _TRUE_KERNEL_M


def _correlated_sample() -> tuple[list[Position], list[float]]:
    from astro_mine.prospect.backends._random_field import correlated_standard_normal

    rows = cols = 64
    cell = 2_000.0 / cols
    field = correlated_standard_normal(
        rows, cols, dx_m=cell, dy_m=cell, correlation_length_m=_TRUE_RANGE_M, seed=7
    )
    rng = np.random.default_rng(1)
    points: list[Position] = []
    values: list[float] = []
    for flat in rng.choice(rows * cols, size=400, replace=False):
        r, c = divmod(int(flat), cols)
        points.append((-1_000.0 + (c + 0.5) * cell, -1_000.0 + (r + 0.5) * cell, 0.0))
        values.append(float(field[r, c]))
    return points, values


def test_the_variogram_recovers_a_known_correlation_length() -> None:
    # Draw a field of *known* correlation length and check the estimator finds it. If a model cannot
    # recover how clumpy the ice actually is, it will site the swarm's next observation in the wrong
    # place — however well-calibrated its variance happens to be.
    points, values = _correlated_sample()
    lags, gamma, counts = empirical_variogram(points, values, n_lags=16, max_lag_m=800.0)
    fit = fit_variogram(lags, gamma, counts=counts)

    assert 0.6 * _EXPECTED_RANGE_M <= fit.correlation_length_m <= 1.6 * _EXPECTED_RANGE_M
    # The field is standard normal, so the semivariance saturates at ~1: the fitted sill recovers
    # the field's variance, and is dominated by *structure* rather than by nugget.
    assert 0.5 < fit.sill < 1.6
    assert fit.nugget < 0.3 * fit.sill
    assert gamma[0] < gamma[-1]  # and it increases with separation, as a variogram must


def test_the_correlated_latent_is_exactly_unit_variance_at_every_cell() -> None:
    """The invariant the whole correlated-realization path rests on (a regression guard).

    ``correlated_standard_normal`` must be ``N(0, 1)`` at *every* cell, or the sealed truth it seeds
    would silently stop being a realization of the prior it was drawn from — and the calibration
    gate, which compares the two, would be scoring against the wrong distribution. It is easy to get
    subtly wrong: smoothing white noise and dividing by the analytic ``sqrt(sum k**2)`` normalizer
    is correct **only** if each output cell sums *distinct* noise samples, which any padding scheme
    built from the field's own interior quietly violates (it over-disperses the draw).
    """
    from astro_mine.prospect.backends._random_field import correlated_standard_normal

    draws = np.stack(
        [
            correlated_standard_normal(
                16, 16, dx_m=50.0, dy_m=50.0, correlation_length_m=150.0, seed=s
            )
            for s in range(400)
        ]
    )
    per_cell_mean = draws.mean(axis=0)
    per_cell_std = draws.std(axis=0)

    assert float(np.abs(per_cell_mean).max()) < 0.2  # zero mean everywhere
    assert float(per_cell_std.min()) > 0.85  # unit variance everywhere — no edge collapse,
    assert float(per_cell_std.max()) < 1.15  # and no interior over-dispersion


def test_the_variogram_reports_no_structure_for_white_noise() -> None:
    # The estimator's honesty check: an uncorrelated field must not be fitted a long length scale.
    rng = np.random.default_rng(3)
    points: list[Position] = [
        (float(x), float(y), 0.0) for x, y in rng.uniform(-900.0, 900.0, size=(120, 2))
    ]
    values = [float(v) for v in rng.standard_normal(120)]
    lags, gamma, counts = empirical_variogram(points, values)
    fit = fit_variogram(lags, gamma, counts=counts)
    # A pure-nugget field: the semivariance is flat, so almost all of the variance is nugget, not
    # spatially-structured sill.
    assert fit.nugget > 2.0 * fit.sill


def test_the_fitted_model_evaluates_and_saturates() -> None:
    points, values = _correlated_sample()
    lags, gamma, counts = empirical_variogram(points, values, n_lags=16, max_lag_m=800.0)
    fit = fit_variogram(lags, gamma, counts=counts)

    # The fitted model is a usable curve: it starts at the nugget and saturates at nugget + sill.
    curve = fit.gamma(np.array([0.0, fit.correlation_length_m, 50.0 * fit.correlation_length_m]))
    assert curve[0] == pytest.approx(fit.nugget)
    assert curve[2] == pytest.approx(fit.nugget + fit.sill, rel=1e-6)
    assert curve[0] < curve[1] < curve[2]  # monotone in separation, as a variogram must be


def test_the_variogram_guards_its_inputs() -> None:
    with pytest.raises(ValueError, match="at least 3 samples"):
        empirical_variogram([(0.0, 0.0, 0.0)], [1.0])
    with pytest.raises(ValueError, match="length mismatch"):
        empirical_variogram([(0.0, 0.0, 0.0)] * 3, [1.0])
    with pytest.raises(ValueError, match="max_lag_m must be positive"):
        empirical_variogram([(0.0, 0.0, 0.0)] * 3, [1.0, 2.0, 3.0], max_lag_m=0.0)
    with pytest.raises(ValueError, match="at least 3 populated lag bins"):
        fit_variogram(np.array([1.0, 2.0]), np.array([0.1, 0.2]))


def test_a_variogram_with_no_spatial_structure_at_all_is_refused() -> None:
    # A *decreasing* variogram is not a spatial-correlation model — no positive sill fits it, and
    # the estimator says so rather than quietly returning a meaningless "fit".
    lags = np.array([100.0, 200.0, 300.0, 400.0])
    decreasing = np.array([1.0, 0.75, 0.5, 0.25])
    with pytest.raises(ValueError, match="no increasing Gaussian variogram fits"):
        fit_variogram(lags, decreasing)


# --- geostatistical sanity: leave-one-out kriging cross-validation (prospect.md §10) --------------


def test_leave_one_out_kriging_error_is_within_bounds() -> None:
    points, values = _sample(n=30, seed=2)

    def fit(train_points: list[Position], train_values: list[float]) -> GridField:
        return GridField.build(
            _METADATA,
            train_points=train_points,
            train_values=train_values,
            prior_mean=_BASELINE,
            prior_variance=0.05,
            length_scale=250.0,
            noise=1e-3,
        )

    report = loo_cross_validation(points, values, fit)  # type: ignore[arg-type]

    assert report.n == 30
    # The out-of-sample error is a small fraction of the field's dynamic range: a model that ignored
    # the data and predicted the prior mean everywhere would score roughly the field's own spread.
    assert report.rmse < 0.25 * _BUMP_PEAK
    assert report.mae <= report.rmse
    # And the *uncertainty* is honest too: errors standardized by the field's own predicted sigma
    # are O(1). Far above 1 would be over-confidence — a field claiming a precision it does not
    # have, which is exactly the credibility hazard prospect.md §9 names. (This bound is a genuine
    # gate, not a formality: it is what catches a backend whose variance collapses faster than its
    # accuracy improves.)
    assert report.standardized_rmse < 3.0


def test_leave_one_out_guards_its_inputs() -> None:
    with pytest.raises(ValueError, match="at least 3 samples"):
        loo_cross_validation([(0.0, 0.0, 0.0)], [1.0], lambda p, v: GridField.build(_METADATA))
    with pytest.raises(ValueError, match="length mismatch"):
        loo_cross_validation([(0.0, 0.0, 0.0)] * 3, [1.0], lambda p, v: GridField.build(_METADATA))


# --- Hypothesis property tests: the invariants prospect.md §10 names ------------------------------

_SMALL_GRID = FieldGrid(
    min_x_m=-500.0, min_y_m=-500.0, max_x_m=500.0, max_y_m=500.0, n_rows=6, n_cols=6
)
_SETTINGS = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)

_coords = st.floats(min_value=-499.0, max_value=499.0, allow_nan=False, allow_infinity=False)
_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_sigmas = st.floats(min_value=1e-3, max_value=0.5, allow_nan=False, allow_infinity=False)
_likelihoods = st.sampled_from(
    [None, "neutron_spectrometer", "nir_reflectance", "gpr", "drill_assay"]
)

_observations = st.builds(
    FieldObservation,
    x_m=_coords,
    y_m=_coords,
    value=_values,
    noise_sigma=_sigmas,
    likelihood=_likelihoods,
)


def _small_prior() -> Prior:
    return load_prior(grid=_SMALL_GRID)


@given(observations=st.lists(_observations, min_size=1, max_size=4))
@_SETTINGS
def test_property_conditioning_never_increases_variance(
    observations: list[FieldObservation],
) -> None:
    """Conditioning on data never makes the belief *less* certain — at any cell, ever.

    prospect.md §10's first named invariant. Every observation adds a non-negative precision to
    every cell, so the posterior variance is bounded above by the prior's everywhere. A violation
    would mean the belief had learned something and become *more* confused by it.
    """
    prior = _small_prior()
    belief = BeliefField.from_prior(prior)
    posterior = belief.update(observations)
    assert np.all(posterior.variance_grid() <= belief.variance_grid() + 1e-15)


@given(
    x=_coords,
    y=_coords,
    value=_values,
    sigma=st.floats(min_value=1e-4, max_value=1e-3, allow_nan=False),
)
@_SETTINGS
def test_property_a_near_noiseless_observation_collapses_variance_at_its_own_location(
    x: float, y: float, value: float, sigma: float
) -> None:
    """The sharper the reading, the tighter the belief at the point it was taken.

    The limiting form of §10's "conditioning never increases variance at a noiselessly-observed
    point": as ``noise_sigma`` goes to zero, the precision ``1/sigma**2`` diverges, so the posterior
    variance at that location must collapse well below the prior's.
    """
    prior = _small_prior()
    belief = BeliefField.from_prior(prior)
    observation = FieldObservation(x_m=x, y_m=y, value=value, noise_sigma=sigma)
    posterior = belief.update([observation])
    position = (x, y, 0.0)
    assert posterior.variance(position) < 0.01 * belief.variance(position)


@given(dummy=st.integers(min_value=0, max_value=3))
@_SETTINGS
def test_property_the_posterior_reduces_to_the_prior_with_no_observations(dummy: int) -> None:
    """With an empty log the posterior *is* the prior — bit for bit (prospect.md §10)."""
    prior = _small_prior()
    belief = BeliefField.from_prior(prior)
    np.testing.assert_array_equal(belief.mean_grid(), prior.mean)
    np.testing.assert_array_equal(belief.variance_grid(), prior.variance)
    np.testing.assert_array_equal(belief.update([]).mean_grid(), prior.mean)


@given(
    observations=st.lists(_observations, max_size=3),
    x=_coords,
    y=_coords,
    levels=st.lists(
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False), min_size=2, max_size=6
    ),
)
@_SETTINGS
def test_property_quantiles_are_monotone(
    observations: list[FieldObservation], x: float, y: float, levels: list[float]
) -> None:
    """Reported quantiles are non-decreasing in their level (prospect.md §10).

    A field whose P90 fell below its P10 would be reporting an incoherent distribution — and every
    downstream risk calculation reading it would be nonsense.
    """
    belief = BeliefField.from_prior(_small_prior()).update(observations)
    position = (x, y, 0.0)
    ordered = sorted(levels)
    quantiles = [belief.quantile(position, q) for q in ordered]
    assert quantiles == sorted(quantiles)

    # And the default posterior summary is coherent: its quantiles are ordered around the mean.
    posterior = belief.posterior(position)
    reported = [posterior.quantiles[q] for q in DEFAULT_QUANTILES]
    assert reported == sorted(reported)
    assert posterior.variance >= 0.0


@given(observations=st.lists(_observations, max_size=4))
@_SETTINGS
def test_property_units_and_crs_are_preserved_across_updates(
    observations: list[FieldObservation],
) -> None:
    """An update never silently changes *what* the field is, or *where* (prospect.md §10).

    Species, SI unit, reference frame, and the explicit planetary CRS are invariant along the whole
    update chain — the belief a subscriber reads is the same quantity, in the same place, as the
    prior it started from (``LUNAR-TR-001``: no implicit Earth/WGS84, ever).
    """
    prior = _small_prior()
    belief = BeliefField.from_prior(prior)
    updated = belief.update(observations)

    assert updated.species == prior.metadata.species == SPECIES
    assert updated.unit == prior.metadata.unit == UNIT
    assert updated.frame == prior.metadata.frame
    assert updated.metadata.crs == prior.metadata.crs
    assert updated.metadata.grid == prior.metadata.grid


@given(observations=st.lists(_observations, min_size=1, max_size=4))
@_SETTINGS
def test_property_the_posterior_is_replayable_in_any_chunking(
    observations: list[FieldObservation],
) -> None:
    """Conditioning incrementally and in one batch give the *same* posterior (prospect.md §5).

    The replay property, under Hypothesis rather than one hand-picked log: the posterior is a pure
    function of ``(prior, ordered log)``, so how a client chose to chunk its updates cannot change
    what it believes — which is exactly what the field service's fail-closed hash check relies on.
    """
    belief = BeliefField.from_prior(_small_prior())
    batch = belief.update(observations)
    incremental = belief
    for observation in observations:
        incremental = incremental.update([observation])

    assert batch.content_hash == incremental.content_hash
    np.testing.assert_array_equal(batch.mean_grid(), incremental.mean_grid())
    np.testing.assert_array_equal(batch.variance_grid(), incremental.variance_grid())
