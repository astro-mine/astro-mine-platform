"""The public synthetic DEM — so the shipped WorldSpec example has a copyable input (issue #60).

`synthetic_polar.world.yaml` could be copied, validated and hashed with nothing downloaded, and then
could not be **built**: the build starts at `ingest_dem`, which needs a raster, and the only raster
generator was a private `conftest` fixture an installed wheel does not expose. So the example's
promise held for authoring and broke at the build.

The load-bearing test here is
:func:`test_the_shipped_example_builds_a_bundle_from_the_wheel_alone` — the issue's acceptance
criterion, end to end, with no network and no SPICE kernels.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio

from astro_mine.core.units.validate import UnitsValidationError
from astro_mine.worlds import terrain
from astro_mine.worlds.crs import LUNAR_SOUTH_POLAR_STEREOGRAPHIC
from astro_mine.worlds.regolith import build_regolith_field
from astro_mine.worlds.spec import WorldSpec, build_world_bundle, example_world_spec_text
from astro_mine.worlds.terrain import NODATA, SYNTHETIC_SOURCE_ID, synthesize_dem
from astro_mine.worlds.thermal import diurnal_curve

# Small enough to keep the raster maths fast; the shape is resolution-independent by construction.
_COARSE_M = 200.0


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


# --- the raster itself ------------------------------------------------------------------------


def test_the_raster_covers_the_requested_region_in_the_requested_crs(tmp_path: Path) -> None:
    path = synthesize_dem(tmp_path / "dem.tif", resolution_m=_COARSE_M)
    with rasterio.open(path) as src:
        assert (
            src.crs.to_proj4()
        )  # an explicit CRS, never absent — `ingest_dem` rejects a CRS-less DEM
        assert src.width == src.height == round(10_000 / _COARSE_M)
        bounds = src.bounds
        assert (bounds.left, bounds.bottom, bounds.right, bounds.top) == (
            -5_000.0,
            -5_000.0,
            5_000.0,
            5_000.0,
        )
        assert src.nodata == NODATA


def test_the_defaults_are_the_shipped_examples_region(tmp_path: Path) -> None:
    """The zero-argument call must be exactly the input the shipped example was missing."""
    spec = WorldSpec.from_yaml_text(example_world_spec_text())
    path = synthesize_dem(tmp_path / "dem.tif")
    with rasterio.open(path) as src:
        assert (src.bounds.left, src.bounds.bottom) == (spec.region.min_x_m, spec.region.min_y_m)
        assert (src.bounds.right, src.bounds.top) == (spec.region.max_x_m, spec.region.max_y_m)
        assert src.width == round(
            (spec.region.max_x_m - spec.region.min_x_m) / spec.region.resolution_m
        )
    # ...and the example already declares the id that marks its terrain as a stand-in.
    assert spec.source_dem.id == SYNTHETIC_SOURCE_ID
    assert spec.source_dem.content_hash is None  # nothing sampled it, so nothing to pin


def test_it_is_byte_reproducible(tmp_path: Path) -> None:
    """A bundle needs a stable world_hash, so its input has to be stable (CX-REPRO)."""
    first = synthesize_dem(tmp_path / "a.tif", resolution_m=_COARSE_M).read_bytes()
    second = synthesize_dem(tmp_path / "b.tif", resolution_m=_COARSE_M).read_bytes()
    assert first == second


def test_the_seed_changes_the_surface_but_not_its_shape(tmp_path: Path) -> None:
    base = _read(synthesize_dem(tmp_path / "s0.tif", resolution_m=_COARSE_M, seed=0))
    other = _read(synthesize_dem(tmp_path / "s1.tif", resolution_m=_COARSE_M, seed=1))
    assert base.shape == other.shape
    assert not np.array_equal(base, other)


def test_the_surface_has_the_relief_the_derived_layers_need(tmp_path: Path) -> None:
    """A flat plane makes slope, aspect and roughness degenerate — the point is to exercise them."""
    elevation = _read(
        synthesize_dem(tmp_path / "dem.tif", resolution_m=_COARSE_M, void_fraction=0.0)
    )
    assert np.isfinite(elevation).all()
    # A basin: the rim stands above the floor by a good fraction of the declared relief.
    assert float(elevation.max() - elevation.min()) > 500.0
    # ...and it is not monotone in either axis (the craters and rim break the paraboloid's trend).
    row = elevation[elevation.shape[0] // 2]
    assert not (np.all(np.diff(row) >= 0) or np.all(np.diff(row) <= 0))


def test_voids_are_written_and_are_a_contiguous_patch(tmp_path: Path) -> None:
    """`fill_voids` and void uncertainty are real code paths; a gap-free DEM skips them."""
    elevation = _read(
        synthesize_dem(tmp_path / "dem.tif", resolution_m=_COARSE_M, void_fraction=0.04)
    )
    void = elevation == NODATA
    assert void.any()
    rows, cols = np.nonzero(void)
    # Contiguous: the patch is a solid rectangle, not scattered pixels.
    assert void.sum() == (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)


def test_no_voids_when_asked_for_none(tmp_path: Path) -> None:
    elevation = _read(
        synthesize_dem(tmp_path / "dem.tif", resolution_m=_COARSE_M, void_fraction=0.0)
    )
    assert not (elevation == NODATA).any()


def test_the_shape_is_resolution_independent(tmp_path: Path) -> None:
    """Same basin, sampled more finely — shrink the grid without changing the world."""
    coarse = _read(synthesize_dem(tmp_path / "c.tif", resolution_m=500.0, void_fraction=0.0))
    fine = _read(synthesize_dem(tmp_path / "f.tif", resolution_m=250.0, void_fraction=0.0))
    assert fine.shape == (coarse.shape[0] * 2, coarse.shape[1] * 2)
    # Compared on the mean, deliberately, and NOT on the extremes. The surface is a paraboloid whose
    # maximum sits at the domain corner, and the outermost cell *centre* converges on that corner
    # only as the grid refines — so a coarse grid reads a lower peak for the same function. That is
    # sampling, not a different world. The mean integrates the same surface at either resolution.
    assert float(fine.mean()) == pytest.approx(float(coarse.mean()), rel=0.02)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_x_m": -5_000.0}, "empty region"),
        ({"max_y_m": -6_000.0}, "empty region"),
        ({"resolution_m": 0.0}, "resolution_m must be positive"),
        ({"resolution_m": -20.0}, "resolution_m must be positive"),
        ({"void_fraction": 1.0}, r"void_fraction must be in \[0, 1\)"),
        ({"void_fraction": -0.1}, r"void_fraction must be in \[0, 1\)"),
    ],
)
def test_bad_arguments_are_refused(tmp_path: Path, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        synthesize_dem(tmp_path / "dem.tif", **kwargs)


def test_an_implicit_earth_crs_is_refused(tmp_path: Path) -> None:
    """`require_crs` at the boundary: an Earth datum on a lunar body is a defaulting bug."""
    earthish = LUNAR_SOUTH_POLAR_STEREOGRAPHIC.model_copy(
        update={"body": "MOON", "projection": "+proj=longlat +datum=WGS84 +no_defs"}
    )
    with pytest.raises(UnitsValidationError, match="Earth CRS marker"):
        synthesize_dem(tmp_path / "dem.tif", crs=earthish, resolution_m=_COARSE_M)


# --- the whole point: it feeds the real pipeline -----------------------------------------------


def test_it_ingests_to_the_grid_the_spec_declares(tmp_path: Path) -> None:
    spec = WorldSpec.from_yaml_text(example_world_spec_text())
    dem = synthesize_dem(tmp_path / "dem.tif", crs=spec.crs, resolution_m=_COARSE_M)
    product = terrain.ingest_dem(dem, tmp_path / "out", target_crs=spec.crs, resolution_m=_COARSE_M)
    assert product.width == product.height == round(10_000 / _COARSE_M)
    assert product.crs == spec.crs
    # Every derived layer is present and non-degenerate — the reason the surface is not a plane.
    assert set(product.layers) >= {"elevation", "slope", "aspect", "roughness", "void_mask"}
    assert float(np.nanmax(_read(product.layers["slope"]))) > 1.0


def test_the_shipped_example_builds_a_bundle_from_the_wheel_alone(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """The issue's acceptance criterion: spec -> bundle with `world.json`, nothing downloaded.

    No DEM fetch, and no SPICE kernels — the PSR layer the spec declares is the one thing this path
    cannot produce, and the build now **says so** rather than silently omitting it.
    """
    spec_path = tmp_path / "my.world.yaml"
    spec_path.write_text(example_world_spec_text(), encoding="utf-8")
    spec = WorldSpec.from_yaml(spec_path)

    dem = synthesize_dem(
        tmp_path / "dem.tif",
        crs=spec.crs,
        min_x_m=spec.region.min_x_m,
        min_y_m=spec.region.min_y_m,
        max_x_m=spec.region.max_x_m,
        max_y_m=spec.region.max_y_m,
        resolution_m=_COARSE_M,
    )
    product = terrain.ingest_dem(
        dem, tmp_path / "terrain", target_crs=spec.crs, resolution_m=_COARSE_M
    )
    regolith = build_regolith_field(product, tmp_path / "regolith")
    thermal = [diurnal_curve(name) for name in spec.layers.thermal_classes]

    bundle = build_world_bundle(
        spec,
        terrain=product,
        regolith=regolith,
        thermal=thermal,
        out_dir=tmp_path / "bundle",
    )

    assert (bundle.path / "world.json").is_file()
    assert bundle.world_hash.startswith("sha256:")
    assert {"terrain", "regolith", "thermal"} <= set(bundle.component_hashes)
    # The declared-but-absent PSR layer is disclosed, not silently dropped.
    assert "illumination" not in bundle.component_hashes
    messages = [str(w.message) for w in recwarn.list if w.category is UserWarning]
    assert any("declares PSR parameters" in message for message in messages), messages


def test_a_psr_free_spec_builds_without_a_warning(tmp_path: Path) -> None:
    """The warning must be about the *mismatch*, not about building without PSR at all."""
    spec = WorldSpec.from_yaml_text(example_world_spec_text())
    psr_free = spec.model_copy(
        update={
            "layers": spec.layers.model_copy(
                update={
                    "psr_semantics": None,
                    "psr_start": None,
                    "psr_days": None,
                    "psr_step_hours": None,
                }
            )
        }
    )
    dem = synthesize_dem(tmp_path / "dem.tif", crs=psr_free.crs, resolution_m=_COARSE_M)
    product = terrain.ingest_dem(
        dem, tmp_path / "terrain", target_crs=psr_free.crs, resolution_m=_COARSE_M
    )
    regolith = build_regolith_field(product, tmp_path / "regolith")
    thermal = [diurnal_curve(name) for name in psr_free.layers.thermal_classes]

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        build_world_bundle(
            psr_free,
            terrain=product,
            regolith=regolith,
            thermal=thermal,
            out_dir=tmp_path / "bundle",
        )
