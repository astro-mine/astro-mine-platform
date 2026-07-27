"""RM-P1-PROSPECT-10 — the deep-generative / normalizing-flow backend behind the contract.

Proves the deliverable: a generative field with an **ensemble** uncertainty representation
implements the full uncertainty-first Core contract, learns a genuinely non-Gaussian (skewed)
marginal from skewed observations while staying calibrated, supports the replayable
belief ``update`` and a ground-truth ``realize`` draw, and registers behind the ``make_backend``
field-backends extension point with no Core change — declaring its ensemble representation for
Zarr tagging.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.core.resource import FieldDistribution, check_resource_field
from astro_mine.core.units import MOON, MOON_BODY_FIXED, PlanetaryCRS
from astro_mine.prospect.backends import GenerativeEnsembleField, make_backend
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
_CENTER: Position = (0.0, 0.0, 0.0)
# A fast, deterministic fit config for the suite.
_CFG: dict[str, object] = dict(
    prior_mean=0.2, prior_variance=0.1, length_scale=300.0, n_iter=120, seed=0
)


def _metadata(grid: FieldGrid | None = _GRID) -> FieldMetadata:
    return FieldMetadata(
        species="water_equivalent_hydrogen",
        unit="mass_fraction",
        frame=MOON_BODY_FIXED,
        crs=_CRS,
        grid=grid,
    )


def _skewed_observations(n: int = 40, seed: int = 0) -> tuple[list[Position], list[float]]:
    rng = np.random.default_rng(seed)
    pts = [(float(x), float(y), 0.0) for x, y in rng.uniform(-900.0, 900.0, size=(n, 2))]
    vals = list(0.2 + rng.exponential(0.15, size=n))  # right-skewed → non-Gaussian shape
    return pts, vals


def _field(**kw: object) -> GenerativeEnsembleField:
    pts, vals = _skewed_observations()
    cfg = dict(_CFG)
    cfg.update(kw)
    return GenerativeEnsembleField.build(_metadata(), train_points=pts, train_values=vals, **cfg)


# --- the Core contract ------------------------------------------------------------------------


def test_satisfies_resource_field_contract() -> None:
    check_resource_field(_field())


def test_posterior_is_uncertainty_first() -> None:
    post = _field().posterior(_CENTER)
    assert isinstance(post, FieldDistribution)
    assert post.variance > 0.0


def test_ensemble_is_non_gaussian() -> None:
    field = _field()
    ens = field.ensemble(_CENTER)
    assert ens.size == 256
    skew = float(((ens - ens.mean()) ** 3).mean() / ens.std() ** 3)
    assert (
        abs(skew) > 0.2
    )  # the flow captured the observations' skew — a genuinely non-Gaussian marginal
    assert not field.flow.is_identity


def test_prior_flow_is_identity_when_no_data() -> None:
    field = GenerativeEnsembleField.from_prior(_metadata(), **_CFG)
    assert field.flow.is_identity
    # With an identity flow the ensemble is (empirically) the Gaussian base.
    assert field.mean(_CENTER) == pytest.approx(0.2, abs=0.05)


def test_queries_are_deterministic() -> None:
    field = _field()
    assert np.array_equal(field.ensemble(_CENTER), field.ensemble(_CENTER))
    assert field.mean(_CENTER) == field.mean(_CENTER)


def test_quantiles_ordered_and_bounded() -> None:
    field = _field()
    assert (
        field.quantile(_CENTER, 0.05) < field.quantile(_CENTER, 0.5) < field.quantile(_CENTER, 0.95)
    )
    ens = field.ensemble(_CENTER)
    assert field.quantile(_CENTER, 0.0) == pytest.approx(float(ens.min()))
    assert field.quantile(_CENTER, 1.0) == pytest.approx(float(ens.max()))


def test_samples_are_seeded_and_empty_ok() -> None:
    field = _field()
    assert field.sample(_CENTER, n=4, seed=1) == field.sample(_CENTER, n=4, seed=1)
    assert field.sample(_CENTER, n=4, seed=1) != field.sample(_CENTER, n=4, seed=2)
    assert field.sample(_CENTER, n=0) == ()


# --- replayable belief semantics + ground-truth realize --------------------------------------


def test_update_is_replayable_and_content_addressed() -> None:
    pts, vals = _skewed_observations()
    obs = [
        FieldObservation(x_m=p[0], y_m=p[1], z_m=p[2], value=v, noise_sigma=0.1)
        for p, v in zip(pts, vals, strict=True)
    ]
    prior = GenerativeEnsembleField.from_prior(_metadata(), **_CFG)
    batch = prior.update(obs)
    incremental = prior.update(obs[:20]).update(obs[20:])
    assert batch.content_hash == incremental.content_hash
    assert batch.mean(_CENTER) == pytest.approx(incremental.mean(_CENTER))
    assert len(batch.log) == len(obs)
    assert prior.content_hash != batch.content_hash


def test_build_with_no_observations_is_the_prior() -> None:
    field = GenerativeEnsembleField.build(_metadata(), **_CFG)
    assert field.log == ()
    assert field.flow.is_identity


def test_realize_is_a_seeded_full_field_draw() -> None:
    field = _field()
    r1 = field.realize(seed=2)
    assert r1.shape == (_GRID.n_rows, _GRID.n_cols)
    assert np.array_equal(r1, field.realize(seed=2))
    assert not np.array_equal(r1, field.realize(seed=3))


# --- the calibration gate ---------------------------------------------------------------------


def test_prior_field_is_calibrated() -> None:
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
    field = GenerativeEnsembleField.from_prior(md, prior_mean=m0, prior_variance=v0, seed=0)
    report = check_calibration(field, held_out, levels=DEFAULT_LEVELS)
    assert report.passed, report.reliability


# --- self-describing / registration -----------------------------------------------------------


def test_zarr_attrs_declares_ensemble_representation() -> None:
    attrs = _field().zarr_attrs()
    assert attrs["uncertainty_representation"] == "ensemble"
    assert attrs["backend"] == "generative"
    assert attrs["ensemble_size"] == "256"


def test_make_backend_routes_to_generative() -> None:
    pts, vals = _skewed_observations()
    field = make_backend("generative", _metadata(), train_points=pts, train_values=vals, **_CFG)
    assert isinstance(field, GenerativeEnsembleField)


# --- error paths ------------------------------------------------------------------------------


def test_missing_grid_rejected() -> None:
    with pytest.raises(ValueError, match=r"requires metadata\.grid"):
        GenerativeEnsembleField.from_prior(_metadata(grid=None), **_CFG)


def test_ensemble_size_must_exceed_one() -> None:
    with pytest.raises(ValueError, match="ensemble_size must be > 1"):
        GenerativeEnsembleField.from_prior(_metadata(), ensemble_size=1, **_CFG)


def test_negative_sample_count_rejected() -> None:
    with pytest.raises(ValueError, match="n must be non-negative"):
        _field().sample(_CENTER, n=-1)


def test_quantile_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"q must be in \[0, 1\]"):
        _field().quantile(_CENTER, 1.5)


def test_build_half_specified_observations_rejected() -> None:
    pts, _ = _skewed_observations()
    with pytest.raises(ValueError, match="provided together"):
        GenerativeEnsembleField.build(_metadata(), train_points=pts, **_CFG)
    with pytest.raises(ValueError, match="length mismatch"):
        GenerativeEnsembleField.build(_metadata(), train_points=pts, train_values=[0.1], **_CFG)
