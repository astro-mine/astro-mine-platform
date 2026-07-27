"""``SamplingPolicy`` validation + golden design determinism (RM-P1-SURR-03; surrogate.md §8).

The declarative datagen spec is validated (ordered box, positive counts) and content-addressed, and
its Sobol/LHS designs are bit-exact reproducible from the seed — the reproducibility contract a
surrogate's provenance leans on (surrogate.md §5). scipy's QMC is integer/`default_rng`-based and
bit-portable, so unlike the torch path these are true goldens, not tolerance gates.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

from astro_mine.surrogate.datagen import (
    AcquisitionKind,
    DesignKind,
    SamplingPolicy,
    design_points,
    grid_design,
    lhs_design,
    sobol_design,
)
from astro_mine.surrogate.report import Bound

# A fixed reference policy — the golden anchor (seed 7, 4-D excavation box).
_BOUNDS = {
    "density": Bound(low=1400.0, high=1600.0),
    "friction": Bound(low=0.4, high=0.7),
    "restitution": Bound(low=0.2, high=0.4),
    "tool_speed": Bound(low=0.05, high=0.08),
}
_POLICY_HASH = "sha256:ce91d40ea0e92cea77ec56e80a44c0db3419ffcd6fcc02c6b3210f2e019bb960"
_SOBOL_SHA = "sha256:10038b1d59c219fd3636aa13b58a3a9ac63e9ed203ec53b2c5df0c0d9eb45ce4"
_LHS_SHA = "sha256:6cfe5b38bdcd7fd4565a54db3594660e31e7dff88083acdfb98549b49a0137a7"


def _policy(**overrides: object) -> SamplingPolicy:
    kwargs: dict[str, object] = dict(
        parameter_bounds=_BOUNDS,
        design=DesignKind.SOBOL,
        n_initial=8,
        pool_size=16,
        n_per_round=3,
        seed=7,
    )
    kwargs.update(overrides)
    return SamplingPolicy(**kwargs)  # type: ignore[arg-type]


def _array_sha(array: np.ndarray) -> str:
    return (
        "sha256:" + hashlib.sha256(np.ascontiguousarray(array, dtype=">f8").tobytes()).hexdigest()
    )


def test_policy_is_frozen_and_rejects_unknown_fields() -> None:
    policy = _policy()
    with pytest.raises(ValidationError):
        policy.n_initial = 4  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SamplingPolicy(  # type: ignore[call-arg]
            parameter_bounds=_BOUNDS, n_initial=4, pool_size=8, surprise="nope"
        )


def test_ordered_bound_is_enforced_by_the_reused_bound_model() -> None:
    with pytest.raises(ValidationError):
        SamplingPolicy(
            parameter_bounds={"density": Bound(low=1600.0, high=1400.0)},
            n_initial=4,
            pool_size=8,
        )


def test_positive_count_validators() -> None:
    with pytest.raises(ValidationError):
        _policy(n_initial=0)  # ge=1
    with pytest.raises(ValidationError):
        _policy(n_per_round=0)  # ge=1
    with pytest.raises(ValidationError):
        _policy(pool_size=2, n_per_round=3)  # pool must cover a round
    with pytest.raises(ValidationError):
        SamplingPolicy(parameter_bounds={}, n_initial=1, pool_size=1)  # min_length=1


def test_param_names_are_the_ordered_box_axes() -> None:
    assert _policy().param_names == ("density", "friction", "restitution", "tool_speed")


def test_content_hash_is_stable_golden() -> None:
    assert _policy().content_hash() == _POLICY_HASH
    # Re-created identically → identical hash (content addressing).
    assert _policy().content_hash() == _policy().content_hash()


def test_sobol_and_lhs_designs_are_bit_exact_goldens() -> None:
    policy = _policy()
    sobol = sobol_design(policy)
    lhs = lhs_design(policy.model_copy(update={"design": DesignKind.LHS}))
    assert sobol.shape == (8, 4)
    assert _array_sha(sobol) == _SOBOL_SHA
    assert _array_sha(lhs) == _LHS_SHA
    # Same seed reproduces bit-for-bit.
    assert np.array_equal(sobol_design(policy), sobol)


def test_designs_lie_inside_the_box() -> None:
    lower = np.array([b.low for b in _BOUNDS.values()])
    upper = np.array([b.high for b in _BOUNDS.values()])
    for design in (
        sobol_design(_policy()),
        lhs_design(_policy(design=DesignKind.LHS)),
        grid_design(_policy(design=DesignKind.GRID)),
    ):
        assert np.all(design >= lower - 1e-9)
        assert np.all(design <= upper + 1e-9)


def test_grid_design_is_a_full_factorial_lattice() -> None:
    # ceil(8 ** (1/4)) = 2 points per dimension → 2**4 = 16 rows.
    grid = grid_design(_policy(design=DesignKind.GRID, n_initial=8))
    assert grid.shape == (16, 4)


def test_design_points_dispatches_on_design_kind() -> None:
    assert design_points(_policy(design=DesignKind.SOBOL)).shape == (8, 4)
    assert design_points(_policy(design=DesignKind.LHS)).shape == (8, 4)
    assert design_points(_policy(design=DesignKind.GRID)).shape[1] == 4


def test_acquisition_default_is_max_uncertainty() -> None:
    assert _policy().acquisition is AcquisitionKind.MAX_UNCERTAINTY
    assert _policy(acquisition=AcquisitionKind.RANDOM).acquisition is AcquisitionKind.RANDOM
