"""End-to-end terrain ingest + TerrainModel (RM-P0-WORLDS-01)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from astro_mine.worlds import terrain
from astro_mine.worlds.crs import LUNAR_SOUTH_POLAR_STEREOGRAPHIC
from astro_mine.worlds.terrain._ingest import (
    RasterLayer,
    TerrainIngestError,
    read_layer,
    reproject_dem,
)

_LAYERS = {"elevation", "slope", "aspect", "roughness", "vertical_uncertainty", "void_mask"}


def test_ingest_produces_all_layers_and_manifest(synthetic_dem: Path, tmp_path: Path) -> None:
    product = terrain.ingest_dem(synthetic_dem, tmp_path / "out", resolution_m=2000.0)
    assert set(product.layers) == _LAYERS
    assert all(p.exists() for p in product.layers.values())
    assert product.terrain_hash.startswith("sha256:")

    manifest = json.loads((product.path / "manifest.json").read_text())
    assert manifest["terrain_hash"] == product.terrain_hash
    assert manifest["crs"]["body"] == "MOON"
    assert manifest["layers"]["slope"]["units"] == "degree"
    # Content-addressed → no wall-clock fields in the manifest.
    assert not any("time" in k or "date" in k for k in manifest)


def test_ingest_reprojects_to_the_explicit_lunar_crs(synthetic_dem: Path, tmp_path: Path) -> None:
    product = terrain.ingest_dem(synthetic_dem, tmp_path / "out", resolution_m=2000.0)
    assert product.crs == LUNAR_SOUTH_POLAR_STEREOGRAPHIC
    with rasterio.open(product.layers["elevation"]) as ds:
        assert ds.crs.is_projected
        assert "1737400" in ds.crs.to_proj4()  # the Moon sphere, not Earth/WGS84


def test_ingest_is_deterministic(synthetic_dem: Path, tmp_path: Path) -> None:
    a = terrain.ingest_dem(synthetic_dem, tmp_path / "a", resolution_m=2000.0)
    b = terrain.ingest_dem(synthetic_dem, tmp_path / "b", resolution_m=2000.0)
    assert a.terrain_hash == b.terrain_hash


def test_model_samples_valid_and_out_of_bounds(synthetic_dem: Path, tmp_path: Path) -> None:
    product = terrain.ingest_dem(synthetic_dem, tmp_path / "out", resolution_m=2000.0)
    model = terrain.TerrainModel.open(product)
    with rasterio.open(product.layers["elevation"]) as ds:
        cx = (ds.bounds.left + ds.bounds.right) / 2.0
        cy = (ds.bounds.bottom + ds.bounds.top) / 2.0
        far_left = ds.bounds.left - 1_000_000.0

    center = model.sample(cx, cy)
    assert center.in_bounds
    assert center.slope_deg >= 0.0

    outside = model.sample(far_left, cy)
    assert not outside.in_bounds
    assert outside.normal == (0.0, 0.0, 1.0)
    assert outside.is_void


def test_model_flags_void_cells(synthetic_dem: Path, tmp_path: Path) -> None:
    product = terrain.ingest_dem(synthetic_dem, tmp_path / "out", resolution_m=2000.0)
    with rasterio.open(product.layers["void_mask"]) as ds:
        mask = ds.read(1)
        voids = np.argwhere(mask >= 1)
        assert voids.size > 0, "the fixture + reprojection edges must yield void cells"
        row, col = (int(voids[0][0]), int(voids[0][1]))
        x, y = ds.xy(row, col)

    sample = terrain.TerrainModel.open(product).sample(x, y)
    assert sample.is_void


def test_crsless_source_is_rejected(crsless_dem: Path) -> None:
    with pytest.raises(TerrainIngestError):
        reproject_dem(crsless_dem, LUNAR_SOUTH_POLAR_STEREOGRAPHIC, resolution_m=2000.0)


def test_raster_layer_matches_read_layer(synthetic_dem: Path, tmp_path: Path) -> None:
    """The in-memory :class:`RasterLayer` sampler is byte-identical to :func:`read_layer` (#48).

    Probes a grid spanning interior cells, the exact bounds edges (the ``nodata``-off-array-edge
    path), voids, and out-of-bounds points — the substitution must reproduce every one, or the
    provider's determinism contract would shift.
    """
    import math

    product = terrain.ingest_dem(synthetic_dem, tmp_path / "out", resolution_m=2000.0)
    for layer_name in ("elevation", "slope", "void_mask"):
        path = product.layers[layer_name]
        cached = RasterLayer.open(path)
        with rasterio.open(path) as ds:
            left, bottom, right, top = ds.bounds
        # Interior/out-of-bounds spread, plus the exact bounds edges — where a point is
        # in-bounds but floors off the array, taking the ``nodata`` branch in both samplers.
        xs = [*np.linspace(left - 5_000.0, right + 5_000.0, 13), left, right]
        ys = [*np.linspace(bottom - 5_000.0, top + 5_000.0, 11), bottom, top]
        for x in xs:
            for y in ys:
                exp_value, exp_in = read_layer(path, float(x), float(y))
                got_value, got_in = cached.sample(float(x), float(y))
                assert got_in == exp_in, (layer_name, x, y)
                if math.isnan(exp_value):
                    assert math.isnan(got_value), (layer_name, x, y)
                else:
                    assert got_value == exp_value, (layer_name, x, y)


def test_terrain_model_materializes_each_layer_once(
    synthetic_dem: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TerrainModel.sample`` opens each of its four layers once, not once per query (#48)."""
    import astro_mine.worlds.terrain._ingest as ingest

    product = terrain.ingest_dem(synthetic_dem, tmp_path / "out", resolution_m=2000.0)
    model = terrain.TerrainModel.open(product)
    with rasterio.open(product.layers["elevation"]) as ds:
        cx = (ds.bounds.left + ds.bounds.right) / 2.0
        cy = (ds.bounds.bottom + ds.bounds.top) / 2.0

    real_open = ingest.rasterio.open
    opens: list[str] = []

    def counting_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        opens.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(ingest.rasterio, "open", counting_open)
    for _ in range(50):
        assert model.sample(cx, cy).in_bounds
    # Four layers (elevation/slope/aspect/void_mask), each opened exactly once across 50 samples.
    assert len(opens) == 4
    assert sorted(Path(p).name for p in opens) == [
        "aspect.tif",
        "elevation.tif",
        "slope.tif",
        "void_mask.tif",
    ]
