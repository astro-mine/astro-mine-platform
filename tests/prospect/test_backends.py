"""RM-P0-PROSPECT-02 — the GP and grid inference backends behind the ResourceField contract.

Proves the deliverable: a GP posterior over a (PSR-scale) region computes
mean/variance/quantiles/samples, the grid backend is a drop-in alternative behind the same
contract, and the choice is an invisible config detail (``make_backend``). Also pins the
load-bearing properties the rest of the backlog leans on: uncertainty-first posteriors,
observation conditioning that shrinks variance toward data, ordered/calibrated quantiles, and
seeded/reproducible sampling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astro_mine.core.resource import FieldDistribution, ResourceField, check_resource_field
from astro_mine.core.units import MOON, MOON_BODY_FIXED, PlanetaryCRS
from astro_mine.prospect.backends import GPField, GridField, make_backend
from astro_mine.prospect.backends._gaussian import gaussian_quantile, gaussian_samples
from astro_mine.prospect.field import BaseResourceField, FieldGrid, FieldMetadata, Position

# A lunar south-pole CRS + grid (Shackleton vicinity), illustrative — consistent with Worlds.
_CRS = PlanetaryCRS(
    body=MOON,
    body_fixed_frame="MOON_ME",
    reference_radius_m=1_737_400.0,
    projection="+proj=stere +lat_0=-90 +R=1737400",
)
_GRID = FieldGrid(
    min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=40, n_cols=40
)

# A central "hot" sample amid low-grade surroundings — a bump the posterior should recover.
_TRAIN_POINTS: list[Position] = [
    (0.0, 0.0, 0.0),
    (-600.0, -600.0, 0.0),
    (600.0, 600.0, 0.0),
    (-600.0, 600.0, 0.0),
    (600.0, -600.0, 0.0),
]
_TRAIN_VALUES = [0.5, 0.1, 0.1, 0.1, 0.1]
_CENTER: Position = (0.0, 0.0, 0.0)
_CORNER: Position = (-950.0, -950.0, 0.0)

# Small, fast, deterministic GP fit knobs for the test suite.
_GP_CFG = {"n_iter": 60, "n_inducing": 5, "seed": 0}


def _metadata(grid: FieldGrid | None = _GRID) -> FieldMetadata:
    return FieldMetadata(
        species="water_equivalent_hydrogen",
        unit="mass_fraction",
        frame=MOON_BODY_FIXED,
        crs=_CRS,
        grid=grid,
    )


def _grid_field() -> GridField:
    return GridField.build(
        _metadata(),
        train_points=_TRAIN_POINTS,
        train_values=_TRAIN_VALUES,
        prior_mean=0.1,
        prior_variance=0.25,
    )


def _gp_field() -> GPField:
    return GPField(_metadata(), train_points=_TRAIN_POINTS, train_values=_TRAIN_VALUES, **_GP_CFG)


# --- the factory: interchangeable backends behind one contract -------------------------------


def test_factory_builds_both_kinds_satisfying_the_contract() -> None:
    for kind in ("gp", "grid"):
        field = make_backend(
            kind,
            _metadata(),
            train_points=_TRAIN_POINTS,
            train_values=_TRAIN_VALUES,
            **(_GP_CFG if kind == "gp" else {}),
        )
        assert isinstance(field, BaseResourceField)
        assert isinstance(field, ResourceField)
        assert check_resource_field(field) is None


def test_factory_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown backend kind"):
        make_backend("kriging", _metadata())


def test_backend_is_a_drop_in_for_a_consumer() -> None:
    # A consumer written against the contract is identical across backends.
    def read(field: ResourceField, position: Position) -> FieldDistribution:
        return field.posterior(position)

    for field in (_gp_field(), _grid_field()):
        dist = read(field, _CENTER)
        assert isinstance(dist, FieldDistribution)
        assert dist.species == "water_equivalent_hydrogen"
        assert dist.unit == "mass_fraction"


# --- contract behavior, parametrized across both backends ------------------------------------


@pytest.fixture(params=["gp", "grid"])
def field(request: pytest.FixtureRequest) -> BaseResourceField:
    return _gp_field() if request.param == "gp" else _grid_field()


def test_posterior_is_uncertainty_first(field: BaseResourceField) -> None:
    dist = field.posterior(_CENTER)
    assert isinstance(dist, FieldDistribution)
    assert isinstance(dist.mean, float)
    assert isinstance(dist.variance, float)
    assert dist.variance >= 0.0
    assert set(dist.quantiles) >= {0.05, 0.5, 0.95}


def test_quantiles_are_ordered_and_centered(field: BaseResourceField) -> None:
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    values = [field.quantile(_CENTER, q) for q in qs]
    assert values == sorted(values)
    # the median equals the mean for a Gaussian posterior
    assert field.quantile(_CENTER, 0.5) == pytest.approx(field.mean(_CENTER), abs=1e-9)


def test_sampling_is_seeded_and_reproducible(field: BaseResourceField) -> None:
    a = field.sample(_CENTER, n=5, seed=7)
    b = field.sample(_CENTER, n=5, seed=7)
    c = field.sample(_CENTER, n=5, seed=8)
    assert a == b
    assert len(a) == 5
    assert a != c  # different seed → different draws (variance is non-zero at the hot point)


def test_conditioning_recovers_the_bump(field: BaseResourceField) -> None:
    # Near the high-value observation the mean is pulled up and the variance shrinks below
    # the far-field (corner) posterior — the "plan to learn" property in miniature.
    assert field.mean(_CENTER) > field.mean(_CORNER)
    assert field.variance(_CENTER) < field.variance(_CORNER)


# --- the GP backend --------------------------------------------------------------------------


def test_gp_prior_with_no_observations_is_flat() -> None:
    field = GPField(_metadata(), prior_mean=0.2, prior_variance=0.05)
    for position in (_CENTER, _CORNER, (123.0, -456.0, 0.0)):
        assert field.mean(position) == pytest.approx(0.2)
        assert field.variance(position) == pytest.approx(0.05)


def test_gp_prior_rejects_nonpositive_variance() -> None:
    with pytest.raises(ValueError, match="prior_variance must be positive"):
        GPField(_metadata(), prior_variance=0.0)


def test_gp_fit_is_deterministic_under_a_seed() -> None:
    one = _gp_field().mean(_CENTER)
    two = _gp_field().mean(_CENTER)
    assert one == two


def test_gp_posterior_variance_is_nonnegative_everywhere() -> None:
    field = _gp_field()
    for position in (_CENTER, _CORNER, (300.0, -200.0, 0.0)):
        assert field.variance(position) >= 0.0


# --- the grid backend ------------------------------------------------------------------------


def test_grid_requires_a_grid_domain() -> None:
    with pytest.raises(ValueError, match="requires metadata"):
        GridField.build(_metadata(grid=None))


def test_grid_prior_with_no_observations_is_flat() -> None:
    field = GridField.build(_metadata(), prior_mean=0.2, prior_variance=0.05)
    assert field.mean(_CENTER) == pytest.approx(0.2)
    assert field.variance(_CENTER) == pytest.approx(0.05)


def test_grid_explicit_length_scale_path() -> None:
    field = GridField.build(
        _metadata(),
        train_points=_TRAIN_POINTS,
        train_values=_TRAIN_VALUES,
        length_scale=250.0,
    )
    assert field.mean(_CENTER) > field.mean(_CORNER)


def test_grid_query_outside_domain_clamps_to_edge() -> None:
    field = _grid_field()
    far = (10_000.0, 10_000.0, 0.0)
    corner_cell = (995.0, 995.0, 0.0)
    assert field.mean(far) == pytest.approx(field.mean(corner_cell), abs=1e-9)


def test_grid_rejects_nonpositive_prior_variance() -> None:
    with pytest.raises(ValueError, match="prior_variance must be positive"):
        GridField.build(_metadata(), prior_variance=-1.0)


def test_grid_rejects_nonpositive_length_scale() -> None:
    with pytest.raises(ValueError, match="length_scale must be positive"):
        GridField.build(
            _metadata(),
            train_points=_TRAIN_POINTS,
            train_values=_TRAIN_VALUES,
            length_scale=0.0,
        )


def test_grid_rejects_nonpositive_noise() -> None:
    with pytest.raises(ValueError, match="noise must be positive"):
        GridField.build(
            _metadata(),
            train_points=_TRAIN_POINTS,
            train_values=_TRAIN_VALUES,
            noise=0.0,
        )


def test_grid_low_level_rejects_bad_shapes() -> None:
    md = _metadata()
    bad = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="must have shape"):
        GridField(md, bad, bad)


def test_grid_low_level_rejects_negative_variance() -> None:
    md = _metadata()
    mean = np.zeros((40, 40), dtype=np.float64)
    variance = np.full((40, 40), -1.0, dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        GridField(md, mean, variance)


def test_grid_low_level_requires_grid_domain() -> None:
    arr = np.zeros((1, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="requires metadata"):
        GridField(_metadata(grid=None), arr, arr)


def test_grid_zero_variance_field_is_degenerate() -> None:
    md = _metadata()
    mean = np.full((40, 40), 0.3, dtype=np.float64)
    variance = np.zeros((40, 40), dtype=np.float64)
    field = GridField(md, mean, variance)
    assert field.quantile(_CENTER, 0.1) == pytest.approx(0.3)
    assert field.sample(_CENTER, n=3, seed=1) == (0.3, 0.3, 0.3)


# --- training validation + the Gaussian helper -----------------------------------------------


def test_training_pair_must_be_complete() -> None:
    with pytest.raises(ValueError, match="provided together"):
        make_backend("grid", _metadata(), train_points=_TRAIN_POINTS)


def test_training_pair_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        GridField.build(_metadata(), train_points=_TRAIN_POINTS, train_values=[0.1])


def test_gaussian_quantile_endpoints_and_validation() -> None:
    assert gaussian_quantile(0.3, 1.0, 0.5) == 0.3
    assert gaussian_quantile(0.3, 1.0, 0.0) == -math.inf
    assert gaussian_quantile(0.3, 1.0, 1.0) == math.inf
    assert gaussian_quantile(0.3, 0.0, 0.2) == 0.3  # degenerate
    with pytest.raises(ValueError, match="q must be in"):
        gaussian_quantile(0.0, 1.0, 1.5)
    with pytest.raises(ValueError, match="variance must be non-negative"):
        gaussian_quantile(0.0, -1.0, 0.5)


def test_gaussian_samples_edge_cases() -> None:
    assert gaussian_samples(0.0, 1.0, n=0, seed=0) == ()
    assert gaussian_samples(0.4, 0.0, n=2, seed=0) == (0.4, 0.4)
    with pytest.raises(ValueError, match="n must be non-negative"):
        gaussian_samples(0.0, 1.0, n=-1, seed=0)


def test_backends_getattr_rejects_unknown_name() -> None:
    # The lazy PEP-562 __getattr__ still rejects genuinely unknown names.
    import astro_mine.prospect.backends as backends

    missing = "does_not_exist"
    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(backends, missing)


def test_backends_dir_lists_the_lazy_exports() -> None:
    # __dir__ surfaces the lazily-imported backends for discovery / tab-completion.
    import astro_mine.prospect.backends as backends

    listed = dir(backends)
    assert {"GPField", "GMRFField", "GenerativeEnsembleField", "GridField"} <= set(listed)
