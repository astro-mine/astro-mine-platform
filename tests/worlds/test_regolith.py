"""Regolith terramechanics parameter field (RM-P0-WORLDS-05).

Pure-kernel tests pin the prior, the uniform means, the void-driven uncertainty inflation,
and the slope-modulation hook on hand-built grids; integration tests build a real field on an
ingested synthetic terrain product and check it shares the terrain CRS/grid, samples as Core
``RegolithParams``, and exposes *only* parameter data (no constitutive law — that is Sim's).
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import rasterio.transform

from astro_mine.core.world import RegolithParams
from astro_mine.worlds.regolith import (
    BUNDLE_SCHEMA,
    DEFAULT_LUNAR_PRIOR,
    PARAM_NAMES,
    ParamPrior,
    RegolithField,
    RegolithPrior,
    build_regolith_field,
)
from astro_mine.worlds.regolith._fields import regolith_hash, regolith_layers
from astro_mine.worlds.terrain import ingest_dem


def _prior_with(name: str, **kwargs: float) -> RegolithPrior:
    """A copy of the default prior with one parameter's fields overridden."""
    base = {n: p for n, p in DEFAULT_LUNAR_PRIOR.items()}
    cur = base[name]
    base[name] = ParamPrior(
        mean=kwargs.get("mean", cur.mean),
        uncertainty=kwargs.get("uncertainty", cur.uncertainty),
        slope_sensitivity=kwargs.get("slope_sensitivity", cur.slope_sensitivity),
    )
    return RegolithPrior(**base)  # type: ignore[arg-type]


# --- pure kernels ----------------------------------------------------------------


def test_default_prior_in_plausible_ranges() -> None:
    p = DEFAULT_LUNAR_PRIOR
    assert 1300.0 <= p.bulk_density.mean <= 1900.0
    assert 100.0 <= p.cohesion.mean <= 1000.0
    assert 30.0 <= p.friction_angle.mean <= 45.0
    assert 1.0e3 <= p.bearing_capacity.mean <= 1.0e5
    assert 30.0 <= p.thermal_inertia.mean <= 100.0
    for _, param in p.items():
        assert param.uncertainty > 0.0


def test_uses_slope() -> None:
    assert not DEFAULT_LUNAR_PRIOR.uses_slope()
    assert _prior_with("cohesion", slope_sensitivity=0.1).uses_slope()


def test_regolith_layers_uniform_means_and_uncertainty() -> None:
    void = np.zeros((4, 5), dtype=np.bool_)
    layers = regolith_layers(DEFAULT_LUNAR_PRIOR, void)
    expected = {n for name in PARAM_NAMES for n in (name, f"{name}_uncertainty")}
    assert set(layers) == expected
    for name, param in DEFAULT_LUNAR_PRIOR.items():
        assert layers[name].shape == (4, 5)
        assert layers[name].dtype == np.float32
        np.testing.assert_allclose(layers[name], param.mean)
        np.testing.assert_allclose(layers[f"{name}_uncertainty"], param.uncertainty)


def test_regolith_layers_inflates_uncertainty_at_voids() -> None:
    void = np.zeros((3, 3), dtype=np.bool_)
    void[1, 1] = True
    layers = regolith_layers(DEFAULT_LUNAR_PRIOR, void)
    unc = layers["bulk_density_uncertainty"]
    factor = DEFAULT_LUNAR_PRIOR.void_uncertainty_factor
    assert unc[1, 1] == pytest.approx(DEFAULT_LUNAR_PRIOR.bulk_density.uncertainty * factor)
    assert unc[0, 0] == pytest.approx(DEFAULT_LUNAR_PRIOR.bulk_density.uncertainty)
    np.testing.assert_allclose(layers["bulk_density"], DEFAULT_LUNAR_PRIOR.bulk_density.mean)


def test_regolith_layers_slope_modulation_and_clamp() -> None:
    void = np.zeros((2, 2), dtype=np.bool_)
    slope = np.array([[0.0, 10.0], [20.0, 30.0]])
    prior = _prior_with("bulk_density", mean=1000.0, slope_sensitivity=0.01)
    layers = regolith_layers(prior, void, slope)
    np.testing.assert_allclose(layers["bulk_density"], 1000.0 * (1.0 + 0.01 * slope), rtol=1e-5)
    np.testing.assert_allclose(layers["cohesion"], DEFAULT_LUNAR_PRIOR.cohesion.mean)  # unmodulated
    # A large negative sensitivity is clamped to a non-negative mean.
    clamped = regolith_layers(
        _prior_with("bulk_density", mean=100.0, slope_sensitivity=-1.0),
        np.zeros((1, 1), dtype=np.bool_),
        np.array([[200.0]]),
    )
    assert clamped["bulk_density"][0, 0] == 0.0


def test_regolith_layers_rejects_mismatched_slope_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        regolith_layers(DEFAULT_LUNAR_PRIOR, np.zeros((2, 2), dtype=np.bool_), np.zeros((3, 3)))


def test_regolith_hash_is_deterministic_and_content_sensitive() -> None:
    layers = {"a": np.zeros((2, 2), dtype=np.float32)}
    meta = {"schema": "x"}
    assert regolith_hash(layers, meta) == regolith_hash(layers, meta)
    assert regolith_hash(layers, meta).startswith("sha256:")
    assert regolith_hash({"a": np.ones((2, 2), dtype=np.float32)}, meta) != regolith_hash(
        layers, meta
    )
    assert regolith_hash(layers, {"schema": "y"}) != regolith_hash(layers, meta)


# --- integration: build over an ingested synthetic terrain product ---------------


def _layer_array(product, name: str) -> np.ndarray:
    with rasterio.open(product.layers[name]) as ds:
        return ds.read(1)


@pytest.fixture
def regolith(synthetic_dem, tmp_path):
    """A regolith field built on a coarsely-ingested synthetic terrain product."""
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    product = build_regolith_field(terrain, tmp_path / "regolith")
    return terrain, product, RegolithField.open(product)


def test_build_writes_layers_and_manifest(regolith) -> None:
    terrain, product, _ = regolith
    assert len(product.layers) == 2 * len(PARAM_NAMES)  # mean + uncertainty per param
    assert product.manifest["schema"] == BUNDLE_SCHEMA
    assert product.regolith_hash.startswith("sha256:")
    assert product.manifest["terrain_hash"] == terrain.terrain_hash
    # Shares the terrain grid exactly, so Sim consumes it without re-projection.
    assert product.transform == terrain.transform
    assert product.crs == terrain.crs


def test_params_returns_core_regolith_params_means(regolith) -> None:
    _, product, field = regolith
    x, y = rasterio.transform.xy(
        rasterio.transform.Affine(*product.transform), product.height // 2, product.width // 2
    )
    p = field.params(float(x), float(y))
    assert isinstance(p, RegolithParams)
    assert p.bulk_density_kg_m3 == pytest.approx(DEFAULT_LUNAR_PRIOR.bulk_density.mean)
    assert p.friction_angle_deg == pytest.approx(DEFAULT_LUNAR_PRIOR.friction_angle.mean)
    assert all(
        v is not None
        for v in (
            p.bulk_density_kg_m3,
            p.cohesion_pa,
            p.friction_angle_deg,
            p.bearing_capacity_pa,
            p.thermal_inertia_tiu,
        )
    )


def test_uncertainty_layers_are_prior_or_void_inflated(regolith) -> None:
    _, product, field = regolith
    means = _layer_array(product, "bulk_density")
    unc = _layer_array(product, "bulk_density_uncertainty")
    base = DEFAULT_LUNAR_PRIOR.bulk_density.uncertainty
    inflated = base * DEFAULT_LUNAR_PRIOR.void_uncertainty_factor
    np.testing.assert_allclose(means, DEFAULT_LUNAR_PRIOR.bulk_density.mean)  # uniform mean
    assert set(np.unique(unc)).issubset({np.float32(base), np.float32(inflated)})
    x, y = rasterio.transform.xy(rasterio.transform.Affine(*product.transform), 0, 0)
    sampled = field.uncertainty(float(x), float(y)).bulk_density_kg_m3
    assert sampled in (pytest.approx(base), pytest.approx(inflated))


def test_out_of_bounds_returns_none_params(regolith) -> None:
    _, _, field = regolith
    p = field.params(1.0e9, 1.0e9)
    assert p.bulk_density_kg_m3 is None
    assert field.uncertainty(1.0e9, 1.0e9).cohesion_pa is None


def test_hash_is_reproducible_across_builds(synthetic_dem, tmp_path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    a = build_regolith_field(terrain, tmp_path / "ra")
    b = build_regolith_field(terrain, tmp_path / "rb")
    assert a.regolith_hash == b.regolith_hash


def test_slope_modulation_varies_means(synthetic_dem, tmp_path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    prior = _prior_with("bulk_density", slope_sensitivity=0.01)
    product = build_regolith_field(terrain, tmp_path / "rmod", prior=prior)
    # The slope-modulated mean is no longer uniform; an unmodulated param stays uniform.
    assert np.unique(_layer_array(product, "bulk_density")).size > 1
    assert np.unique(_layer_array(product, "cohesion")).size == 1


def test_public_api_exposes_no_constitutive_law() -> None:
    # Separation of concerns: Worlds owns the parameter data, Sim owns the physics.
    import astro_mine.worlds.regolith as reg

    forbidden = ("step", "force", "contact", "constitutive", "simulate", "stress", "advance")
    names = list(reg.__all__) + dir(RegolithField)
    assert not [n for n in names if any(f in n.lower() for f in forbidden)]


def test_params_materializes_each_layer_once(regolith, monkeypatch) -> None:
    """``RegolithField.params`` opens its five layers once, not once per query (#48)."""
    import astro_mine.worlds.terrain._ingest as ingest

    _, product, field = regolith
    x, y = rasterio.transform.xy(
        rasterio.transform.Affine(*product.transform), product.height // 2, product.width // 2
    )

    real_open = ingest.rasterio.open
    opens: list[str] = []

    def counting_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        opens.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(ingest.rasterio, "open", counting_open)
    for _ in range(50):
        field.params(float(x), float(y))
    # One open per mean layer (the five PARAM_NAMES), across all 50 queries.
    assert len(opens) == len(PARAM_NAMES)
