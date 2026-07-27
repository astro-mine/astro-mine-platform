"""RM-P1-PROSPECT-12 — the real PDS raster-ingest prior-recipe.

Proves the deliverable and acceptance criteria (prospect.md §2.4, §3, §4, §6, §12) against
**synthetic** conditioning rasters (CI runs offline; the multi-MB real PDS fetch is a one-time,
documented step per ``LUNAR-TR-004``):

- a named recipe **ingests conditioning rasters, reprojects them onto the Shackleton prior grid**,
  and fits a prior with **per-product content-addressed provenance** (real ``source_hash`` per
  raster; LCROSS stays a magnitude anchor);
- it is **registered behind the existing registry, additive** — the parametric
  ``shackleton_water_ice_v1`` remains the offline default and is unchanged;
- the ingested prior is **reproducible** (deterministic ``content_hash``) and **materialized as a
  content-addressed bundle** the offline recipe re-fits from with **numpy alone** (fail-closed on a
  tampered bundle) — the local tier needs no network/GDAL;
- it is **calibration-valid** (RM-P0-PROSPECT-07) and **publishes** through the same
  ``resource_field_backend`` path (real source hashes flow into the manifest ``input_hashes``).

The reproject/materialize path needs GDAL, so the module is skipped when rasterio is absent (CI
installs the ``[ingest]`` extra via the dev group).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astro_mine.core.resource import check_resource_field
from astro_mine.core.units import UnitsValidationError
from astro_mine.prospect.calibration import (
    DEFAULT_COVERAGE_TOLERANCE,
    build_calibration_case,
    check_calibration,
)
from astro_mine.prospect.field import FieldGrid
from astro_mine.prospect.isolation import GROUND_TRUTH_ACCESS
from astro_mine.prospect.priors import (
    PDS_RECIPE_NAME,
    build_pds_prior,
    list_priors,
    load_conditioning_bundle,
    load_prior,
)
from astro_mine.prospect.priors.catalog import (
    LCROSS_WATER_WT_FRACTION,
    LEND_BACKGROUND_WEH,
    SHACKLETON_CRS,
)
from astro_mine.prospect.priors.ingest import (
    CONDITIONING_MEMBER,
    MANIFEST_MEMBER,
    RasterInput,
    bundle_content_hash,
    ingest_conditioning,
    materialize_conditioning_bundle,
    validate_manifest_crs,
)
from astro_mine.prospect.priors.pds import CONDITIONING_DIR_ENV
from astro_mine.prospect.publish import (
    BUNDLE_MEDIA_TYPE,
    build_field_manifest,
    bundle_digest,
    from_bundle,
    serialize_bundle,
)

pytest.importorskip("rasterio", reason="the [ingest] extra (GDAL/rasterio) is required")
import rasterio

# A small test grid in the Shackleton CRS, and slightly larger source coverage so the reprojection
# has full coverage to sample (except where a source deliberately withholds data).
_GRID = FieldGrid(
    min_x_m=-2_000.0, min_y_m=-2_000.0, max_x_m=2_000.0, max_y_m=2_000.0, n_rows=24, n_cols=24
)
_SRC_BOUNDS = (-3_000.0, -3_000.0, 3_000.0, 3_000.0)


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float | None, dtype: str) -> Path:
    """Write ``arr`` as a single-band GeoTIFF in the Shackleton CRS over ``_SRC_BOUNDS``."""
    h, w = arr.shape
    transform = rasterio.transform.from_bounds(*_SRC_BOUNDS, w, h)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=dtype,
        crs=rasterio.crs.CRS.from_proj4(SHACKLETON_CRS.projection),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def _synthetic_sources(tmp: Path) -> dict[str, RasterInput]:
    """Four synthetic conditioning rasters with clear spatial structure + partial-coverage cases."""
    tmp.mkdir(parents=True, exist_ok=True)
    n = 40
    yy, xx = np.mgrid[0:n, 0:n]
    center = (xx - n / 2) ** 2 + (yy - n / 2) ** 2 < (n / 4) ** 2  # a central cold-trap disc

    psr = np.where(center, 1, 0).astype(np.uint8)  # binary PSR mask
    # Diviner: cold (60 K) in the trap, warm (220 K) outside, stored as DN (K = DN*0.02); a nodata
    # corner exercises the coverage-weighted fill + sigma inflation.
    kelvin = np.where(center, 60.0, 220.0)
    dn = (kelvin / 0.02).astype(np.int16)
    dn[:5, :5] = -32768
    # LEND suppression index [0,1]: fully covered, smoothly higher toward the pole (grid centre).
    lend = (1.0 - (np.hypot(xx - n / 2, yy - n / 2) / (n / 2))).clip(0.0, 1.0).astype(np.float32)
    # M³ band depth: a hydrated patch (0.12) on an ignore-0 background → sparse coverage.
    m3 = np.zeros((n, n), dtype=np.float32)
    m3[8:18, 8:18] = 0.12

    return {
        "psr": RasterInput(
            path=_write_raster(tmp / "psr.tif", psr, nodata=None, dtype="uint8"),
            role="psr",
            units="fraction",
            citation="LOLA",
            resampling="average",
        ),
        "diviner_temperature": RasterInput(
            path=_write_raster(tmp / "diviner.tif", dn, nodata=-32768.0, dtype="int16"),
            role="measured_temperature",
            units="K",
            citation="Diviner",
            scale=0.02,
            nodata=-32768.0,
            resampling="average",
        ),
        "lend_suppression": RasterInput(
            path=_write_raster(tmp / "lend.tif", lend, nodata=None, dtype="float32"),
            role="neutron_suppression",
            units="suppression_index",
            citation="LEND",
            resampling="bilinear",
        ),
        "m3_band_depth": RasterInput(
            path=_write_raster(tmp / "m3.tif", m3, nodata=0.0, dtype="float32"),
            role="band_depth",
            units="band_depth",
            citation="M3",
            nodata=0.0,
            resampling="average",
        ),
    }


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """Ingest the synthetic sources onto ``_GRID`` and materialize a conditioning bundle."""
    sources = _synthetic_sources(tmp_path / "src")
    layer_set = ingest_conditioning(sources, grid=_GRID, crs=SHACKLETON_CRS)
    out = tmp_path / "conditioning"
    materialize_conditioning_bundle(layer_set, out)
    return out


# --- ingest + reproject onto the prior grid --------------------------------------------------


def test_ingest_reprojects_all_layers_onto_the_prior_grid(tmp_path: Path) -> None:
    sources = _synthetic_sources(tmp_path)
    layer_set = ingest_conditioning(sources, grid=_GRID, crs=SHACKLETON_CRS)
    assert set(layer_set.layers) == set(sources)
    for layer in layer_set.layers.values():
        assert layer.values.shape == (_GRID.n_rows, _GRID.n_cols)  # co-registered on the prior grid
        assert layer.source_hash.startswith("sha256:")
    # PSR/LEND fully cover the grid; M³ (ignore-0) and Diviner (nodata corner) are partial.
    assert np.isfinite(layer_set.array("psr")).all()
    assert np.isfinite(layer_set.array("lend_suppression")).all()
    assert not np.isfinite(layer_set.array("m3_band_depth")).all()
    # PSR averaged from a binary mask is a shadow *fraction* in [0, 1].
    psr = layer_set.array("psr")
    assert psr.min() >= 0.0 and psr.max() <= 1.0 and (psr > 0.0).any()


def test_ingest_requires_a_source_crs(tmp_path: Path) -> None:
    arr = np.ones((10, 10), dtype=np.float32)
    path = tmp_path / "no_crs.tif"
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        transform=rasterio.transform.from_bounds(*_SRC_BOUNDS, 10, 10),
    ) as dst:
        dst.write(arr, 1)
    src = RasterInput(path=path, role="psr", units="fraction", citation="LOLA")
    # A CRS-less source is the implicit-Earth defaulting bug; the fail-loud verdict now comes from
    # Core's require_crs guard (RM-P1-CORE-08), not a hand-rolled inline check (RM-P1-PROSPECT-14).
    with pytest.raises(UnitsValidationError, match="explicit planetary CRS"):
        ingest_conditioning({"psr": src}, grid=_GRID, crs=SHACKLETON_CRS)


# --- materialized bundle: content-addressed, numpy-only, fail-closed -------------------------


def test_materialized_bundle_round_trips_numpy_only(bundle_dir: Path) -> None:
    assert (bundle_dir / CONDITIONING_MEMBER).exists()
    manifest = json.loads((bundle_dir / MANIFEST_MEMBER).read_text())
    assert manifest["schema"].startswith("astro-mine-prospect/conditioning/")
    assert manifest["content_hash"].startswith("sha256:")
    bundle = load_conditioning_bundle(bundle_dir)
    assert set(bundle.roles) == {"psr", "measured_temperature", "neutron_suppression", "band_depth"}
    assert bundle.layer("psr") is not None
    assert bundle.layer("absent-role") is None
    assert bundle.source_hash("measured_temperature").startswith("sha256:")


def test_bundle_load_is_fail_closed_on_tamper(bundle_dir: Path) -> None:
    manifest_path = bundle_dir / MANIFEST_MEMBER
    manifest = json.loads(manifest_path.read_text())
    manifest["content_hash"] = "sha256:" + "0" * 64  # a lie about the arrays
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_conditioning_bundle(bundle_dir)


# --- manifest["crs"] is schema-validated + guarded before any rasterio machinery (criterion 1) ---


def _rewrite_manifest_crs(bundle_dir: Path, crs: dict[str, object] | None) -> None:
    """Overwrite the bundle manifest's ``crs`` block (drop it entirely when ``crs is None``)."""
    manifest_path = bundle_dir / MANIFEST_MEMBER
    manifest = json.loads(manifest_path.read_text())
    if crs is None:
        del manifest["crs"]
    else:
        manifest["crs"] = crs
    manifest_path.write_text(json.dumps(manifest))


def test_load_rejects_a_manifest_missing_a_crs_field_with_a_named_schema_error(
    bundle_dir: Path,
) -> None:
    # Drop a required CRS field (as a non-Python producer easily could): the load fails against
    # Core's units.schema.json, naming the offending field — not with an opaque rasterio error.
    _rewrite_manifest_crs(
        bundle_dir,
        {"body": "MOON", "body_fixed_frame": "MOON_ME"},  # reference_radius_m missing
    )
    with pytest.raises(ValueError, match=r"units\.schema\.json.*reference_radius_m"):
        load_conditioning_bundle(bundle_dir)


def test_load_rejects_an_earth_shaped_crs_on_a_lunar_body_with_a_guard_error(
    bundle_dir: Path,
) -> None:
    # An Earth datum on a lunar body is schema-valid (datum is just a string) but a defaulting bug;
    # require_crs rule 6 rejects it with a guard error, again before rasterio sees the CRS.
    _rewrite_manifest_crs(
        bundle_dir,
        {
            "body": "MOON",
            "body_fixed_frame": "MOON_ME",
            "reference_radius_m": 1_737_400.0,
            "datum": "WGS84",
        },
    )
    with pytest.raises(UnitsValidationError, match="Earth CRS marker"):
        load_conditioning_bundle(bundle_dir)


def test_validate_manifest_crs_accepts_a_valid_lunar_crs() -> None:
    crs = validate_manifest_crs(SHACKLETON_CRS.model_dump(mode="json"))
    assert crs == SHACKLETON_CRS


# --- the fit: real provenance, honest uncertainty, reproducible ------------------------------


def test_fit_produces_a_cited_calibrated_prior(bundle_dir: Path) -> None:
    bundle = load_conditioning_bundle(bundle_dir)
    prior = build_pds_prior(_GRID, bundle)

    # A valid Core ResourceField with the right species/unit/grid.
    check_resource_field(prior.as_field())
    assert prior.metadata.species == "water_equivalent_hydrogen"
    assert prior.metadata.unit == "mass_fraction"
    assert prior.metadata.grid == _GRID
    # Honest uncertainty: variance strictly positive everywhere; mean within [background, peak].
    assert (prior.variance > 0.0).all()
    assert prior.mean.min() >= LEND_BACKGROUND_WEH - 1e-9
    assert prior.mean.max() <= LCROSS_WATER_WT_FRACTION + 1e-9
    # Spatial structure: the cold-trap disc is wetter than the warm surround.
    assert prior.mean.max() > prior.mean.min() + 1e-3

    # Per-product content-addressed provenance: the four ingested rasters carry real hashes; LCROSS
    # stays a magnitude anchor with no raster.
    by_name = {c.short_name: c for c in prior.provenance.citations}
    for name in ("LOLA", "Diviner", "LEND", "M3"):
        assert by_name[name].source_hash is not None
        assert by_name[name].source_hash.startswith("sha256:")
    assert by_name["LCROSS"].source_hash is None
    assert prior.provenance.recipe == PDS_RECIPE_NAME


def test_fit_is_deterministic(bundle_dir: Path) -> None:
    bundle = load_conditioning_bundle(bundle_dir)
    assert (
        build_pds_prior(_GRID, bundle).content_hash == build_pds_prior(_GRID, bundle).content_hash
    )


def test_fit_rejects_a_bundle_with_no_usable_layers(tmp_path: Path) -> None:
    # A bundle whose only layer carries an unrecognized role → no ice-favorability signal.
    src = RasterInput(
        path=_write_raster(
            tmp_path / "x.tif", np.ones((8, 8), np.float32), nodata=None, dtype="float32"
        ),
        role="unrecognized_role",
        units="dimensionless",
        citation="LOLA",
    )
    layer_set = ingest_conditioning({"x": src}, grid=_GRID, crs=SHACKLETON_CRS)
    out = tmp_path / "empty"
    materialize_conditioning_bundle(layer_set, out)
    with pytest.raises(ValueError, match="no usable conditioning layers"):
        build_pds_prior(_GRID, load_conditioning_bundle(out))


def test_fit_rejects_a_mismatched_grid(bundle_dir: Path) -> None:
    other = FieldGrid(min_x_m=-1.0, min_y_m=-1.0, max_x_m=1.0, max_y_m=1.0, n_rows=4, n_cols=4)
    with pytest.raises(ValueError, match="does not match the conditioning bundle"):
        build_pds_prior(other, load_conditioning_bundle(bundle_dir))


def test_pds_prior_passes_the_calibration_gate(bundle_dir: Path) -> None:
    prior = build_pds_prior(_GRID, load_conditioning_bundle(bundle_dir))
    belief, held_out = build_calibration_case(
        prior,
        seed=7,
        n_train=0,
        n_holdout=300,
        noise_sigma=0.05,
        capabilities=(GROUND_TRUTH_ACCESS,),
    )
    report = check_calibration(belief, held_out)
    assert report.passed
    assert report.max_deviation < DEFAULT_COVERAGE_TOLERANCE


# --- registry integration: additive, offline default preserved ------------------------------


def test_recipe_is_registered_additively(bundle_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    priors = list_priors()
    assert PDS_RECIPE_NAME in priors
    assert "shackleton_water_ice_v1" in priors  # the parametric default is untouched

    # Resolved via the registry (env-pointed bundle) == a direct build.
    monkeypatch.setenv(CONDITIONING_DIR_ENV, str(bundle_dir))
    resolved = load_prior(PDS_RECIPE_NAME, grid=_GRID)
    direct = build_pds_prior(_GRID, load_conditioning_bundle(bundle_dir))
    assert resolved.content_hash == direct.content_hash


def test_recipe_fails_closed_without_a_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONDITIONING_DIR_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="no conditioning bundle"):
        load_prior(PDS_RECIPE_NAME, grid=_GRID)


# --- publish: real source hashes flow through the resource_field_backend manifest ------------


def test_publish_round_trip_carries_real_source_hashes(bundle_dir: Path) -> None:
    prior = build_pds_prior(_GRID, load_conditioning_bundle(bundle_dir))
    bundle_bytes = serialize_bundle(prior)
    manifest = build_field_manifest(prior, bundle_sha256=bundle_digest(bundle_bytes))

    # The four ingested-raster hashes flow into the manifest input_hashes automatically.
    assert manifest.provenance is not None
    assert len(manifest.provenance.input_hashes) == 4
    assert all(h.startswith("sha256:") for h in manifest.provenance.input_hashes)

    # Reopens into a live field without re-running the recipe (numpy only).
    field = from_bundle(manifest, {BUNDLE_MEDIA_TYPE: bundle_bytes})
    check_resource_field(field)
    assert field.species == prior.metadata.species


def test_content_hash_is_stable_across_reload(bundle_dir: Path) -> None:
    first = build_pds_prior(_GRID, load_conditioning_bundle(bundle_dir)).content_hash
    second = build_pds_prior(_GRID, load_conditioning_bundle(bundle_dir)).content_hash
    assert first == second
    # The bundle's own content hash matches its arrays (recomputed numpy-only).
    with np.load(bundle_dir / CONDITIONING_MEMBER) as npz:
        arrays = {k: np.asarray(npz[k], dtype=np.float32) for k in npz.files}
    manifest = json.loads((bundle_dir / MANIFEST_MEMBER).read_text())
    assert bundle_content_hash(arrays) == manifest["content_hash"]
