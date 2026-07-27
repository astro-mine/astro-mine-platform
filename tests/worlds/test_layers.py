"""Derived-layer NumPy kernels (RM-P0-WORLDS-01)."""

from __future__ import annotations

import numpy as np

from astro_mine.worlds.terrain._layers import (
    fill_voids,
    normal_from_slope_aspect,
    roughness,
    slope_aspect,
    terrain_hash,
    vertical_uncertainty,
)


def test_slope_of_a_constant_ramp_is_uniform() -> None:
    # A plane rising 1 m per metre eastward → 45° slope everywhere.
    res = 10.0
    xx = np.arange(20, dtype=np.float32)
    elev = np.tile(xx * res, (20, 1)).astype(np.float32)  # gradient 1.0 in x
    slope, aspect = slope_aspect(elev, res)
    assert np.allclose(slope, 45.0, atol=1e-3)
    # Aspect is uniform across the constant-gradient plane.
    assert np.allclose(aspect, aspect[0, 0], atol=1e-3)


def test_roughness_is_zero_on_a_flat_surface() -> None:
    flat = np.full((10, 10), 5.0, dtype=np.float32)
    assert np.allclose(roughness(flat), 0.0, atol=1e-6)


def test_roughness_is_positive_where_elevation_varies() -> None:
    rng = np.arange(100, dtype=np.float32).reshape(10, 10)
    assert float(roughness(rng).max()) > 0.0


def test_vertical_uncertainty_inflates_at_voids() -> None:
    void = np.zeros((4, 4), dtype=np.bool_)
    void[1, 1] = True
    field = vertical_uncertainty(void, baseline_m=1.0, void_factor=5.0)
    assert field[0, 0] == np.float32(1.0)
    assert field[1, 1] == np.float32(5.0)


def test_fill_voids_uses_valid_median_and_handles_all_void() -> None:
    elev = np.array([[1.0, 2.0], [3.0, 100.0]], dtype=np.float32)
    void = np.array([[False, False], [False, True]], dtype=np.bool_)
    filled = fill_voids(elev, void)
    assert filled[1, 1] == np.float32(np.median([1.0, 2.0, 3.0]))  # 2.0
    all_void = fill_voids(elev, np.ones_like(void))
    assert np.all(all_void == 0.0)


def test_normal_of_flat_ground_points_up() -> None:
    assert normal_from_slope_aspect(0.0, 0.0) == (0.0, 0.0, 1.0)


def test_terrain_hash_is_deterministic_and_data_sensitive() -> None:
    a = {"elevation": np.arange(9, dtype=np.float32).reshape(3, 3)}
    meta = {"k": "v"}
    assert terrain_hash(a, meta) == terrain_hash(a, meta)
    assert terrain_hash(a, meta).startswith("sha256:")
    b = {"elevation": np.ones((3, 3), dtype=np.float32)}
    assert terrain_hash(a, meta) != terrain_hash(b, meta)
    assert terrain_hash(a, meta) != terrain_hash(a, {"k": "w"})
