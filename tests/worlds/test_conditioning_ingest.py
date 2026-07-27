"""Diviner/LEND/M³ conditioning-layer ingest (RM-P1-WORLDS-14).

Mirrors the repo's "real pipeline against synthetic inputs" pattern: synthesized source
rasters (in a lunar geographic CRS) are ingested through the real reproject → convert → COG →
STAC → provenance pipeline and co-registered to an ingested synthetic DEM's grid, then read
back through :class:`ConditioningField` and the world-provider surface. Real PDS rasters are
fetched outside CI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.crs
import rasterio.transform
import rasterio.warp
from rasterio.transform import from_bounds

from astro_mine.core.world import check_world_provider
from astro_mine.worlds.crs import lunar_geographic_proj4
from astro_mine.worlds.ingest import (
    CONDITIONING_SPECS,
    ConditioningField,
    ingest_conditioning_layers,
    lend_epithermal_to_weh,
    m3_band_depth_to_water,
)
from astro_mine.worlds.provider import DemWorldProvider
from astro_mine.worlds.regolith import build_regolith_field
from astro_mine.worlds.terrain import ingest_dem

_WEST, _SOUTH, _EAST, _NORTH = -2.0, -87.0, 2.0, -86.0


def _write_source(path: Path, value: float) -> Path:
    """A small lunar-geographic source raster filled with ``value`` (a covered CRS)."""
    width, height = 40, 40
    data = np.full((height, width), value, dtype=np.float32)
    crs = rasterio.crs.CRS.from_proj4(lunar_geographic_proj4())
    transform = from_bounds(_WEST, _SOUTH, _EAST, _NORTH, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def sources(tmp_path):
    return {
        "diviner_temperature": _write_source(tmp_path / "diviner.tif", 95.0),  # K
        "lend_weh": _write_source(tmp_path / "lend.tif", 2.5),  # epithermal count rate
        "m3_water": _write_source(tmp_path / "m3.tif", 0.6),  # band depth
    }


@pytest.fixture
def terrain(synthetic_dem, tmp_path):
    return ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)


# --- unit: the reduced-order conversions -----------------------------------------


def test_lend_conversion_is_monotone_and_bounded() -> None:
    assert lend_epithermal_to_weh(np.array([5.0], np.float32))[0] == pytest.approx(0.0)  # dry
    assert lend_epithermal_to_weh(np.array([0.0], np.float32))[0] == pytest.approx(5.0)  # saturated
    assert lend_epithermal_to_weh(np.array([2.5], np.float32))[0] == pytest.approx(2.5)  # mid
    assert lend_epithermal_to_weh(np.array([9.0], np.float32))[0] == 0.0  # above dry -> clipped


def test_m3_conversion_is_linear_and_bounded() -> None:
    assert m3_band_depth_to_water(np.array([0.6], np.float32))[0] == pytest.approx(0.3)
    assert m3_band_depth_to_water(np.array([9.0], np.float32))[0] == 1.0  # clipped to max


# --- ingest: co-registration + provenance + STAC ---------------------------------


def test_layers_coregister_with_the_dem_grid(sources, terrain, tmp_path) -> None:
    layer_set = ingest_conditioning_layers(sources, terrain, tmp_path / "cond")
    assert set(layer_set.layers) == set(CONDITIONING_SPECS)
    assert layer_set.crs == terrain.crs
    for layer in layer_set.layers.values():
        with rasterio.open(layer.path) as ds:
            assert (ds.width, ds.height) == (terrain.width, terrain.height)  # co-registered
            assert ds.transform == rasterio.transform.Affine(*terrain.transform)
            assert ds.crs == rasterio.crs.CRS.from_proj4(terrain.crs.projection)


def test_manifest_carries_per_product_provenance(sources, terrain, tmp_path) -> None:
    layer_set = ingest_conditioning_layers(sources, terrain, tmp_path / "cond")
    man = layer_set.manifest["layers"]
    assert man["diviner_temperature"]["instrument"] == "Diviner"
    assert man["lend_weh"]["role"] == "weh"
    assert man["m3_water"]["units"] == "wt_percent"
    for entry in man.values():
        assert entry["source_hash"].startswith("sha256:")


def test_stac_catalog_is_written_and_valid_json(sources, terrain, tmp_path) -> None:
    layer_set = ingest_conditioning_layers(sources, terrain, tmp_path / "cond")
    catalog = json.loads(layer_set.stac_catalog.read_text())
    assert catalog["type"] == "Catalog"
    item_ids = {link["href"] for link in catalog["links"] if link["rel"] == "item"}
    assert item_ids == {f"./{name}.json" for name in CONDITIONING_SPECS}


def test_unknown_layer_name_rejected(terrain, tmp_path) -> None:
    with pytest.raises(KeyError):
        ingest_conditioning_layers(
            {"bogus": _write_source(tmp_path / "b.tif", 1.0)}, terrain, tmp_path / "c"
        )


# --- reader + provider surface ---------------------------------------------------


def _center_position(terrain_product):
    cx, cy = rasterio.transform.xy(
        rasterio.transform.Affine(*terrain_product.transform),
        terrain_product.height // 2,
        terrain_product.width // 2,
    )
    proj = rasterio.crs.CRS.from_proj4(terrain_product.crs.projection)
    geo = rasterio.crs.CRS.from_proj4(lunar_geographic_proj4())
    lons, lats = rasterio.warp.transform(proj, geo, [cx], [cy])
    lon, lat = math.radians(lons[0]), math.radians(lats[0])
    r = float(terrain_product.crs.reference_radius_m)
    return (float(cx), float(cy)), (
        r * math.cos(lat) * math.cos(lon),
        r * math.cos(lat) * math.sin(lon),
        r * math.sin(lat),
    )


def test_conditioning_field_samples_converted_values(sources, terrain, tmp_path) -> None:
    layer_set = ingest_conditioning_layers(sources, terrain, tmp_path / "cond")
    field = ConditioningField.open(layer_set)
    assert set(field.layers) == set(CONDITIONING_SPECS)
    (cx, cy), _ = _center_position(terrain)
    values = field.sample(cx, cy)
    assert values["diviner_temperature"] == pytest.approx(95.0, abs=0.5)  # passthrough K
    assert values["lend_weh"] == pytest.approx(2.5, abs=0.1)  # WEH from count 2.5
    assert values["m3_water"] == pytest.approx(0.3, abs=0.05)  # water from band depth 0.6
    assert field.sample_layer("lend_weh", cx, cy) == pytest.approx(2.5, abs=0.1)


def test_provider_serves_conditioning_through_the_surface(sources, terrain, tmp_path) -> None:
    build_regolith_field(terrain, tmp_path / "regolith")
    layer_set = ingest_conditioning_layers(sources, terrain, tmp_path / "cond")
    provider = DemWorldProvider.open(
        terrain,
        tmp_path / "regolith",
        conditioning=ConditioningField.open(layer_set),
        n_azimuth=8,
        max_radius_m=8000.0,
        abcorr="NONE",
    )
    check_world_provider(provider)  # still honours the Core WorldProvider contract
    (_, position) = _center_position(terrain)
    values = provider.conditioning_at(position)
    assert values["diviner_temperature"] == pytest.approx(95.0, abs=0.5)
    # Off-grid (a southern point far outside the small polar patch): every layer is NaN.
    r = float(terrain.crs.reference_radius_m)
    off_lat = math.radians(-70.0)
    off_pos = (r * math.cos(off_lat), 0.0, r * math.sin(off_lat))
    off = provider.conditioning_at(off_pos)
    assert all(math.isnan(v) for v in off.values())


def test_provider_without_conditioning_raises(terrain, tmp_path) -> None:
    build_regolith_field(terrain, tmp_path / "regolith")
    provider = DemWorldProvider.open(
        terrain, tmp_path / "regolith", n_azimuth=8, max_radius_m=8000.0, abcorr="NONE"
    )
    with pytest.raises(LookupError, match="no conditioning source"):
        provider.conditioning_at((1.0, 1.0, 1.0))
