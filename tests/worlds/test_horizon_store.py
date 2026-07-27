"""Persisted horizon maps + Zarr field layers in the world bundle (issue #39; worlds.md §5, §7).

The acceptance criteria of the gap, end to end:

- the ``(H, W, n_azimuth)`` horizon map is **written to and read from a Zarr store** as part of the
  bundle publish/load path (it used to be recomputed in-process on every construction and never
  persisted at all);
- a **second N-D field layer** — the thermal diurnal curves — is Zarr-backed rather than a flat COG
  or a dropped array, matching worlds.md §5's format table;
- opening a bundle with **no** persisted horizon map recomputes it in-process and yields the **same
  ``illumination_hash``** as before (the load-path fallback);
- ``world_hash`` **changes** when a persisted horizon map's content changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astro_mine.worlds.crs import LUNAR_SOUTH_POLAR_STEREOGRAPHIC
from astro_mine.worlds.fields import read_field_zarr, zarr_store_hash
from astro_mine.worlds.illumination import (
    HORIZON_STORE_NAME,
    IlluminationError,
    IlluminationModel,
    PsrEpochSemantics,
)
from astro_mine.worlds.provider import DemWorldProvider
from astro_mine.worlds.regolith import build_regolith_field
from astro_mine.worlds.spec import Region, SourceRef, WorldSpec, build_world_bundle
from astro_mine.worlds.spec._bundle import _world_hash
from astro_mine.worlds.terrain import ingest_dem
from astro_mine.worlds.thermal import diurnal_curve

_N_AZIMUTH = 16
_MAX_RADIUS_M = 8000.0
_RESOLUTION_M = 2000.0


def _spec() -> WorldSpec:
    return WorldSpec(
        world_id="horizon-store-test",
        crs=LUNAR_SOUTH_POLAR_STEREOGRAPHIC,
        region=Region(
            min_x_m=-40_000.0,
            min_y_m=-40_000.0,
            max_x_m=40_000.0,
            max_y_m=40_000.0,
            resolution_m=_RESOLUTION_M,
        ),
        source_dem=SourceRef(id="synthetic"),
    )


def _model(terrain: object) -> IlluminationModel:
    return IlluminationModel(
        terrain,  # type: ignore[arg-type]
        n_azimuth=_N_AZIMUTH,
        max_radius_m=_MAX_RADIUS_M,
        abcorr="NONE",
    )


@pytest.fixture
def terrain(synthetic_dem: Path, tmp_path: Path):
    return ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=_RESOLUTION_M)


# --- the model persists and re-adopts its horizon map -------------------------------


def test_horizon_round_trips_through_zarr_with_an_identical_hash(terrain, tmp_path: Path) -> None:
    """A stored map must reconstruct the model exactly — same array, same illumination_hash."""
    built = _model(terrain)
    assert built.horizon_source == "recomputed"
    store = built.write_horizon_zarr(tmp_path / HORIZON_STORE_NAME)
    assert store.arrays == {"horizon": built.horizon.shape}

    loaded = IlluminationModel(
        terrain,
        n_azimuth=_N_AZIMUTH,
        max_radius_m=_MAX_RADIUS_M,
        abcorr="NONE",
        horizon_store=store.path,
    )
    assert loaded.horizon_source == "stored"
    assert np.array_equal(loaded.horizon, built.horizon)
    assert loaded.illumination_hash == built.illumination_hash


def test_missing_store_falls_back_to_an_in_process_recompute(terrain, tmp_path: Path) -> None:
    """The load-path fallback: no store is a cache miss, not an error (issue #39)."""
    built = _model(terrain)
    fallback = IlluminationModel(
        terrain,
        n_azimuth=_N_AZIMUTH,
        max_radius_m=_MAX_RADIUS_M,
        abcorr="NONE",
        horizon_store=tmp_path / "not-there.zarr",
    )
    assert fallback.horizon_source == "recomputed"
    assert np.array_equal(fallback.horizon, built.horizon)
    assert fallback.illumination_hash == built.illumination_hash


def test_a_mismatched_store_fails_loudly(terrain, tmp_path: Path) -> None:
    """A horizon built for other parameters must never be silently substituted (worlds.md §9)."""
    store = _model(terrain).write_horizon_zarr(tmp_path / HORIZON_STORE_NAME)
    with pytest.raises(IlluminationError, match="does not match this illumination model"):
        IlluminationModel(
            terrain,
            n_azimuth=_N_AZIMUTH * 2,  # a different skyline
            max_radius_m=_MAX_RADIUS_M,
            abcorr="NONE",
            horizon_store=store.path,
        )


def test_a_store_without_a_horizon_array_fails_loudly(terrain, tmp_path: Path) -> None:
    from astro_mine.worlds.fields import FieldArray, write_field_zarr

    store = write_field_zarr(
        tmp_path / "bogus.zarr",
        [FieldArray(name="not_horizon", values=np.zeros((2, 2)), units="m", dims=("y", "x"))],
        attrs={},
    )
    with pytest.raises(IlluminationError, match="not a horizon-map store"):
        _model_with_store(terrain, store.path)


def _model_with_store(terrain: object, store: Path) -> IlluminationModel:
    return IlluminationModel(
        terrain,  # type: ignore[arg-type]
        n_azimuth=_N_AZIMUTH,
        max_radius_m=_MAX_RADIUS_M,
        abcorr="NONE",
        horizon_store=store,
    )


# --- the bundle ships the Zarr field layers -----------------------------------------


def _bundle(terrain, synthetic_spice, out: Path, *, thermal: bool = True):
    regolith = build_regolith_field(terrain, out / "regolith")
    psr = _model(terrain).psr_mask(
        synthetic_spice.window, 6.0 * 3600.0, semantics=PsrEpochSemantics.MISSION
    )
    return build_world_bundle(
        _spec(),
        terrain=terrain,
        regolith=regolith,
        psr=psr,
        thermal=[diurnal_curve("polar_lit"), diurnal_curve("crater_floor")] if thermal else None,
        out_dir=out / "bundle",
    )


def test_bundle_writes_the_horizon_map_as_a_zarr_field_layer(
    terrain, synthetic_spice, tmp_path: Path
) -> None:
    bundle = _bundle(terrain, synthetic_spice, tmp_path)
    store = bundle.path / "illumination" / HORIZON_STORE_NAME
    assert store.is_dir()

    arrays, attrs = read_field_zarr(store)
    model = _model(terrain)
    assert np.array_equal(arrays["horizon"], model.horizon)
    assert attrs["layer"] == "illumination/horizon"
    assert attrs["illumination_hash"] == model.illumination_hash

    # It is declared in the manifest, catalogued in STAC, and folded into the component hashes.
    fields = bundle.manifest["fields"]
    assert fields["illumination/horizon"]["path"] == f"illumination/{HORIZON_STORE_NAME}"
    assert fields["illumination/horizon"]["media_type"] == "application/vnd+zarr"
    assert bundle.component_hashes["illumination_horizon"] == zarr_store_hash(store)
    item = json.loads((bundle.path / "stac" / "illumination-horizon.json").read_text())
    assert item["assets"]["data"]["type"] == "application/vnd+zarr"
    # ...and the Zarr version is recorded in the toolchain, because the writer actually ran. It is
    # provenance, not hash input: the store's bytes are already covered by its own store_hash above.
    assert "zarr" in bundle.manifest["toolchain"]


def test_bundle_writes_the_thermal_curves_as_a_zarr_field_layer(
    terrain, synthetic_spice, tmp_path: Path
) -> None:
    """worlds.md §5 puts thermal among the "Field models -> Zarr" N-D layers (issue #39)."""
    bundle = _bundle(terrain, synthetic_spice, tmp_path)
    arrays, attrs = read_field_zarr(bundle.path / "thermal" / "curves.zarr")

    assert attrs["terrain_classes"] == ["crater_floor", "polar_lit"]  # sorted, as the bundle writes
    curves = {name: diurnal_curve(name) for name in attrs["terrain_classes"]}
    assert arrays["temperature_k"].shape == (2, curves["polar_lit"].phases.size)
    # The actual solved curves ship now — previously only their min/max reached thermal.json.
    for index, name in enumerate(attrs["terrain_classes"]):
        assert np.allclose(arrays["temperature_k"][index], curves[name].temperatures_k)
    assert np.allclose(arrays["phase"], curves["polar_lit"].phases)
    assert bundle.component_hashes["thermal_curves"].startswith("sha256:")


def test_world_hash_changes_when_the_persisted_horizon_content_changes(
    terrain, synthetic_spice, tmp_path: Path
) -> None:
    """Tampering with a stored chunk must move world_hash — the provenance-honesty criterion."""
    bundle = _bundle(terrain, synthetic_spice, tmp_path)
    store = bundle.path / "illumination" / HORIZON_STORE_NAME

    chunk = next(p for p in store.rglob("c/*/*/*") if p.is_file())
    chunk.write_bytes(chunk.read_bytes() + b"tampered")

    tampered = dict(bundle.component_hashes)
    tampered["illumination_horizon"] = zarr_store_hash(store)
    assert tampered["illumination_horizon"] != bundle.component_hashes["illumination_horizon"]
    assert _world_hash(bundle.spec, tampered) != bundle.world_hash


def test_a_bundle_with_no_horizon_ships_no_field_layer(
    terrain, synthetic_spice, tmp_path: Path
) -> None:
    """A PsrResult without a horizon (a test double) simply ships none — and says so honestly."""
    from dataclasses import replace

    psr = _model(terrain).psr_mask(
        synthetic_spice.window, 6.0 * 3600.0, semantics=PsrEpochSemantics.MISSION
    )
    bundle = build_world_bundle(
        _spec(),
        terrain=terrain,
        psr=replace(psr, horizon=None),
        out_dir=tmp_path / "bundle",
    )
    assert not (bundle.path / "illumination" / HORIZON_STORE_NAME).exists()
    assert "illumination_horizon" not in bundle.component_hashes
    assert bundle.manifest["fields"] == {}
    # No Zarr layer was written, so the Zarr version does not perturb this bundle's hash.
    assert "zarr" not in bundle.manifest["toolchain"]


# --- the provider load path adopts the stored map ------------------------------------


def test_provider_open_adopts_a_stored_horizon(terrain, tmp_path: Path) -> None:
    regolith = build_regolith_field(terrain, tmp_path / "regolith")
    store = _model(terrain).write_horizon_zarr(tmp_path / HORIZON_STORE_NAME)
    provider = DemWorldProvider.open(
        terrain,
        regolith,
        n_azimuth=_N_AZIMUTH,
        max_radius_m=_MAX_RADIUS_M,
        abcorr="NONE",
        horizon_store=store.path,
    )
    assert provider.illumination.horizon_source == "stored"
    assert provider.illumination.illumination_hash == _model(terrain).illumination_hash
