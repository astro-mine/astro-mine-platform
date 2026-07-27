"""RM-P0-PROSPECT-04 — Bayesian belief updating from an ordered, replayable observation log.

Proves the deliverable and its acceptance: feeding observations updates the belief (the mean tracks
the data, the variance shrinks where observed, weighted by each reading's likelihood); the log is
ordered and append-only; and **replaying the log reproduces the same posterior chain** — updating
incrementally equals updating in one batch, byte-for-byte (content-addressed). Also pins the
conditioning math against the shipped grid backend (a constant prior reduces to the grid backend).
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.core.resource import ResourceField, check_resource_field
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.prospect.backends import GridField
from astro_mine.prospect.belief import BeliefField, FieldObservation, load_observations
from astro_mine.prospect.field import FieldGrid, FieldMetadata, Position
from astro_mine.prospect.priors import SHACKLETON_CRS, SPECIES, UNIT, load_prior

_CENTER: Position = (0.0, 0.0, 0.0)
_CORNER: Position = (-900.0, -900.0, 0.0)


def _grid() -> FieldGrid:
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


def _metadata(grid: FieldGrid | None) -> FieldMetadata:
    return FieldMetadata(
        species=SPECIES, unit=UNIT, frame=MOON_BODY_FIXED, crs=SHACKLETON_CRS, grid=grid
    )


def _belief() -> BeliefField:
    return BeliefField.from_prior(load_prior(grid=_grid()))


def _hit(x: float, y: float, value: float, noise: float = 0.01) -> FieldObservation:
    return FieldObservation(x_m=x, y_m=y, value=value, noise_sigma=noise)


# --- the initial belief is the prior ---------------------------------------------------------


def test_initial_belief_is_the_prior() -> None:
    prior = load_prior(grid=_grid())
    belief = BeliefField.from_prior(prior)
    prior_field = prior.as_field()
    for position in (_CENTER, _CORNER):
        assert belief.mean(position) == pytest.approx(prior_field.mean(position))
        assert belief.variance(position) == pytest.approx(prior_field.variance(position))
    assert belief.log == ()
    assert belief.prior_hash == prior.content_hash


def test_belief_satisfies_the_resource_field_contract() -> None:
    belief = _belief().update([_hit(0.0, 0.0, 0.5)])
    assert isinstance(belief, ResourceField)
    assert check_resource_field(belief) is None


# --- updating: the mean tracks the data, the variance shrinks ---------------------------------


def test_update_pulls_the_mean_and_shrinks_the_variance() -> None:
    belief = _belief()
    post = belief.update([_hit(0.0, 0.0, 0.9)])  # a high reading at the centre
    assert post.mean(_CENTER) > belief.mean(_CENTER)  # pulled toward the observation
    assert post.variance(_CENTER) < belief.variance(_CENTER)  # learned → less uncertain
    # Far from the observation the posterior reverts to the prior.
    assert post.variance(_CORNER) == pytest.approx(belief.variance(_CORNER), rel=1e-6)


def test_update_is_immutable_and_appends_in_order() -> None:
    belief = _belief()
    a, b = _hit(0.0, 0.0, 0.5), _hit(100.0, 100.0, 0.3)
    post = belief.update([a, b])
    assert belief.log == ()  # the original is unchanged
    assert post.log == (a, b)  # appended in order
    assert post.update([_hit(200.0, 200.0, 0.2)]).log[:2] == (a, b)


def test_lower_noise_observations_carry_more_weight() -> None:
    prior = load_prior(grid=_grid())
    confident = BeliefField.from_prior(prior).update([_hit(0.0, 0.0, 0.5, noise=0.001)])
    uncertain = BeliefField.from_prior(prior).update([_hit(0.0, 0.0, 0.5, noise=0.1)])
    # The low-noise reading pulls the mean closer to 0.5 and shrinks the variance more.
    assert abs(confident.mean(_CENTER) - 0.5) < abs(uncertain.mean(_CENTER) - 0.5)
    assert confident.variance(_CENTER) < uncertain.variance(_CENTER)


# --- replay: incremental == batch, content-addressed -----------------------------------------


def test_replay_reproduces_the_same_posterior_chain() -> None:
    prior = load_prior(grid=_grid())
    log = [_hit(0.0, 0.0, 0.9), _hit(100.0, 100.0, 0.2), _hit(-500.0, 300.0, 0.4)]

    batch = BeliefField.from_prior(prior).update(log)

    incremental = BeliefField.from_prior(prior)
    for observation in log:
        incremental = incremental.update([observation])

    assert incremental.log == tuple(log)
    assert batch.content_hash == incremental.content_hash
    for position in (_CENTER, _CORNER, (100.0, 100.0, 0.0)):
        assert batch.mean(position) == pytest.approx(incremental.mean(position))
        assert batch.variance(position) == pytest.approx(incremental.variance(position))


def test_content_hash_is_stable_and_sensitive() -> None:
    prior = load_prior(grid=_grid())
    log = [_hit(0.0, 0.0, 0.9)]
    one = BeliefField.from_prior(prior).update(log)
    two = BeliefField.from_prior(prior).update(log)
    assert one.content_hash == two.content_hash  # same prior + same log → same identity
    assert one.content_hash != BeliefField.from_prior(prior).content_hash  # an obs changed it


def test_empty_update_is_an_identity() -> None:
    belief = _belief()
    same = belief.update([])
    assert same.content_hash == belief.content_hash
    assert same.mean(_CENTER) == pytest.approx(belief.mean(_CENTER))


# --- the CSV feed (the acceptance path) ------------------------------------------------------


def test_csv_feed_matches_a_direct_update() -> None:
    prior = load_prior(grid=_grid())
    rows = [
        "x_m,y_m,z_m,value,noise_sigma,time_s,sensor",
        "0,0,0,0.9,0.01,0,ns",
        "100,100,0,0.2,0.02,1,ns",
    ]
    from_csv = BeliefField.from_prior(prior).update(load_observations(rows))
    direct = BeliefField.from_prior(prior).update(
        [
            FieldObservation(x_m=0.0, y_m=0.0, value=0.9, noise_sigma=0.01, sensor="ns"),
            FieldObservation(
                x_m=100.0, y_m=100.0, value=0.2, noise_sigma=0.02, time_s=1.0, sensor="ns"
            ),
        ]
    )
    assert from_csv.content_hash == direct.content_hash


# --- the conditioning math reduces to the shipped grid backend -------------------------------


def test_constant_prior_conditioning_equals_the_grid_backend() -> None:
    md = _metadata(_grid())
    shape = (8, 8)
    prior_mean, prior_variance = 0.1, 0.25
    noise, length_scale = 0.01, 250.0
    points: list[Position] = [(0.0, 0.0, 0.0), (-600.0, -600.0, 0.0), (600.0, 600.0, 0.0)]
    values = [0.5, 0.1, 0.2]

    belief = BeliefField(
        md,
        np.full(shape, prior_mean),
        np.full(shape, prior_variance),
        tuple(
            FieldObservation(x_m=p[0], y_m=p[1], value=v, noise_sigma=noise**0.5)
            for p, v in zip(points, values, strict=True)
        ),
        length_scale=length_scale,
        prior_hash="constant",
    )
    grid_field = GridField.build(
        md,
        train_points=points,
        train_values=values,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        length_scale=length_scale,
        noise=noise,
    )
    for position in (*points, _CORNER, (300.0, -200.0, 0.0)):
        assert belief.mean(position) == pytest.approx(grid_field.mean(position))
        assert belief.variance(position) == pytest.approx(grid_field.variance(position))


# --- construction guards ---------------------------------------------------------------------


def test_constructor_requires_a_grid_domain() -> None:
    with pytest.raises(ValueError, match="requires metadata"):
        BeliefField(
            _metadata(None),
            np.zeros((8, 8)),
            np.ones((8, 8)),
            (),
            length_scale=None,
            prior_hash="x",
        )


def test_constructor_rejects_mismatched_prior_shapes() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        BeliefField(
            _metadata(_grid()),
            np.zeros((3, 3)),
            np.ones((3, 3)),
            (),
            length_scale=None,
            prior_hash="x",
        )


def test_constructor_rejects_nonpositive_prior_variance() -> None:
    with pytest.raises(ValueError, match="prior_variance must be positive"):
        BeliefField(
            _metadata(_grid()),
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            (),
            length_scale=None,
            prior_hash="x",
        )


def test_constructor_rejects_nonpositive_length_scale() -> None:
    with pytest.raises(ValueError, match="length_scale must be positive"):
        BeliefField(
            _metadata(_grid()),
            np.zeros((8, 8)),
            np.ones((8, 8)),
            (),
            length_scale=0.0,
            prior_hash="x",
        )
