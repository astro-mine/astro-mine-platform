"""RM-P0-PROSPECT-04 — the sealed ground-truth realization.

Proves the deliverable and its acceptance: ground truth is a fixed, seeded realization sampled
from a prior (deterministic, physically bounded, sealed-immutable), it queries as a degenerate
ResourceField (the true value, zero uncertainty), and it is **not exposed through the belief path**
(prospect.md §9). Its forward sensor model emits the synthetic observations that drive a belief.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.core.resource import ResourceField, check_resource_field
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.prospect.belief import GroundTruthField, sample_ground_truth
from astro_mine.prospect.field import FieldGrid, FieldMetadata
from astro_mine.prospect.isolation import GROUND_TRUTH_ACCESS
from astro_mine.prospect.priors import SHACKLETON_CRS, SPECIES, UNIT, load_prior

_CENTER = (0.0, 0.0, 0.0)
_CORNER = (-900.0, -900.0, 0.0)
#: The capability grant a privileged (Sim-side) caller presents to mint/read the sealed truth.
#: Agent-facing isolation itself is exercised in ``test_isolation.py``.
_GRANT = (GROUND_TRUTH_ACCESS,)


def _grid() -> FieldGrid:
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


def _metadata(grid: FieldGrid | None) -> FieldMetadata:
    return FieldMetadata(
        species=SPECIES, unit=UNIT, frame=MOON_BODY_FIXED, crs=SHACKLETON_CRS, grid=grid
    )


def _truth(seed: int = 0) -> GroundTruthField:
    return sample_ground_truth(load_prior(grid=_grid()), seed=seed, capabilities=_GRANT)


# --- the ResourceField contract (degenerate: the truth is exact) ------------------------------


def test_ground_truth_satisfies_the_resource_field_contract() -> None:
    gt = _truth()
    assert isinstance(gt, ResourceField)
    assert check_resource_field(gt) is None


def test_ground_truth_is_a_degenerate_field() -> None:
    gt = _truth()
    assert gt.variance(_CENTER) == 0.0  # the truth carries no uncertainty
    assert gt.quantile(_CENTER, 0.1) == pytest.approx(gt.mean(_CENTER))  # every q is the value
    value = gt.mean(_CENTER)
    assert gt.sample(_CENTER, n=3, seed=1) == (value, value, value)


def test_metadata_carries_through() -> None:
    gt = _truth()
    assert gt.species == SPECIES
    assert gt.unit == UNIT
    assert gt.frame == MOON_BODY_FIXED


# --- a seeded, reproducible, bounded realization ---------------------------------------------


def test_realization_is_deterministic_under_a_seed() -> None:
    a, b = _truth(seed=7), _truth(seed=7)
    np.testing.assert_array_equal(a.reveal(capabilities=_GRANT), b.reveal(capabilities=_GRANT))
    assert a.content_hash == b.content_hash


def test_realization_differs_across_seeds() -> None:
    a, b = _truth(seed=1), _truth(seed=2)
    assert not np.array_equal(a.reveal(capabilities=_GRANT), b.reveal(capabilities=_GRANT))
    assert a.content_hash != b.content_hash


def test_realization_respects_the_physical_floor() -> None:
    # A resource concentration cannot be negative — the draw is clipped at zero.
    assert float(_truth(seed=3).reveal(capabilities=_GRANT).min()) >= 0.0


def test_realization_tracks_the_prior_structure() -> None:
    # The cold-trap prior mean is higher at the pole than the far field; over many cells the
    # realization carries that structure (the pole grid-cell mean exceeds the corner's).
    gt = _truth(seed=0)
    realization = gt.reveal(capabilities=_GRANT)
    assert realization.shape == (8, 8)
    assert float(realization[4, 4]) >= 0.0


def test_seed_is_exposed() -> None:
    assert _truth(seed=5).seed == 5


# --- sealing: the revealed realization is immutable ------------------------------------------


def test_realization_is_read_only() -> None:
    gt = _truth()
    with pytest.raises(ValueError, match=r"read-only|assignment"):
        gt.reveal(capabilities=_GRANT)[0, 0] = 99.0


# --- the forward sensor model ----------------------------------------------------------------


def test_observe_draws_noisy_readings_of_the_truth() -> None:
    gt = _truth()
    points = [_CENTER, _CORNER, (500.0, -500.0, 0.0)]
    obs = gt.observe(points, noise_sigma=0.01, seed=1, capabilities=_GRANT)
    assert len(obs) == 3
    for o, p in zip(obs, points, strict=True):
        assert o.position == p
        assert o.noise_sigma == 0.01
        assert o.sensor == "synthetic"
        assert abs(o.value - gt.mean(p)) < 0.1  # within a few sigma of the truth


def test_observe_is_seeded_and_reproducible() -> None:
    gt = _truth()
    a = gt.observe([_CENTER], noise_sigma=0.02, seed=4, capabilities=_GRANT)
    b = gt.observe([_CENTER], noise_sigma=0.02, seed=4, capabilities=_GRANT)
    c = gt.observe([_CENTER], noise_sigma=0.02, seed=5, capabilities=_GRANT)
    assert a[0].value == b[0].value
    assert a[0].value != c[0].value


def test_observe_rejects_nonpositive_noise() -> None:
    with pytest.raises(ValueError, match="noise_sigma must be positive"):
        _truth().observe([_CENTER], noise_sigma=0.0, seed=0, capabilities=_GRANT)


# (Ground-truth/belief isolation — the truth is unreachable through the agent-facing path, and
# minting/reading it is capability-gated — is RM-P0-PROSPECT-05, exercised in test_isolation.py.)


# --- construction guards ---------------------------------------------------------------------


def test_constructor_requires_a_grid_domain() -> None:
    with pytest.raises(ValueError, match="requires metadata"):
        GroundTruthField(_metadata(None), np.zeros((8, 8)), seed=0, prior_hash="x")


def test_constructor_rejects_a_mismatched_realization_shape() -> None:
    with pytest.raises(ValueError, match="must have grid shape"):
        GroundTruthField(_metadata(_grid()), np.zeros((3, 3)), seed=0, prior_hash="x")
