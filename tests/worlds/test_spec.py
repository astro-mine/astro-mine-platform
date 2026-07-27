"""RM-P0-WORLDS-07 — WorldSpec, the content-addressed bundle, STAC catalog, and 3D-Tiles export.

Proves the deliverable and its acceptance: a ``WorldSpec`` authored as YAML round-trips and
content-addresses by canonical JSON; :func:`build_world_bundle` composes the WORLDS-01..06 layer
products into a bundle whose ``world_hash`` is **reproducible** (two builds are byte-identical, the
"two clean checkouts" property) and changes with the spec; the STAC catalog resolves to the layer
COGs; and the 3D-Tiles tileset + glTF parse and carry the terrain mesh.
"""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import rasterio.crs
from jsonschema import Draft202012Validator
from numpy.typing import NDArray

from astro_mine.core import schema_registry
from astro_mine.core.schemas import core_schema
from astro_mine.core.units import MOON, MOON_BODY_FIXED, PlanetaryCRS, UnitsValidationError
from astro_mine.worlds.crs import LUNAR_SOUTH_POLAR_STEREOGRAPHIC, MOON_RADIUS_M, to_lonlat
from astro_mine.worlds.illumination import IlluminationModel, PsrEpochSemantics, PsrResult
from astro_mine.worlds.regolith import build_regolith_field
from astro_mine.worlds.spec import (
    LayerSpec,
    Region,
    SourceRef,
    WorldBundle,
    WorldSpec,
    build_world_bundle,
)
from astro_mine.worlds.spec._schema import WorldSchemaError, validate_units_object
from astro_mine.worlds.spec._tiles import enu_to_body_fixed, heightfield_mesh
from astro_mine.worlds.terrain import ingest_dem
from astro_mine.worlds.thermal import diurnal_curve


def _region() -> Region:
    return Region(
        min_x_m=-30_000.0,
        min_y_m=-30_000.0,
        max_x_m=30_000.0,
        max_y_m=30_000.0,
        resolution_m=2000.0,
    )


def _spec(**overrides: object) -> WorldSpec:
    base: dict[str, object] = {
        "world_id": "shackleton-test",
        "crs": LUNAR_SOUTH_POLAR_STEREOGRAPHIC,
        "region": _region(),
        "source_dem": SourceRef(id="synthetic-lola", description="CI stand-in DEM"),
        "layers": LayerSpec(regolith_prior="default_lunar", thermal_classes=("polar_lit",)),
        "description": "test world",
    }
    base.update(overrides)
    return WorldSpec(**base)  # type: ignore[arg-type]


# --- WorldSpec model -------------------------------------------------------------------------


def test_yaml_round_trip() -> None:
    spec = _spec()
    again = WorldSpec.from_yaml_text(spec.to_yaml())
    assert again == spec
    assert again.spec_hash == spec.spec_hash


def test_from_yaml_file(tmp_path: Path) -> None:
    spec = _spec()
    path = tmp_path / "world.yaml"
    path.write_text(spec.to_yaml(), encoding="utf-8")
    assert WorldSpec.from_yaml(path) == spec


def test_spec_hash_is_canonical_and_order_independent() -> None:
    spec = _spec()
    # A dict with different key insertion order hashes identically (canonical JSON).
    payload = spec.model_dump(mode="json")
    reordered = dict(reversed(list(payload.items())))
    assert WorldSpec.model_validate(reordered).spec_hash == spec.spec_hash
    assert spec.spec_hash.startswith("sha256:")


def test_region_rejects_inverted_extent() -> None:
    with pytest.raises(ValueError, match="max_x/max_y must exceed"):
        Region(min_x_m=0.0, min_y_m=0.0, max_x_m=-1.0, max_y_m=1.0, resolution_m=10.0)


def test_extra_field_is_forbidden() -> None:
    with pytest.raises(ValueError, match=r"extra|forbidden|not permitted"):
        _spec(bogus=1)


def test_from_yaml_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        WorldSpec.from_yaml_text("- 1\n- 2\n")


def test_json_schema_is_exposed() -> None:
    schema = WorldSpec.json_schema()
    assert "properties" in schema
    assert "world_id" in schema["properties"]


# --- bundle build ----------------------------------------------------------------------------


def test_build_terrain_only_bundle(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bundle = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")

    assert isinstance(bundle, WorldBundle)
    assert bundle.world_hash.startswith("sha256:")
    assert set(bundle.component_hashes) == {"terrain"}
    assert (bundle.path / "world.json").exists()
    assert (bundle.path / "terrain" / "elevation.tif").exists()
    assert bundle.stac_catalog.exists()
    assert bundle.tileset.exists()
    assert (bundle.path / "tiles" / "terrain.glb").exists()


def _full_bundle(dem: Path, root: Path, window: object) -> WorldBundle:
    terrain = ingest_dem(dem, root / "terrain", resolution_m=2000.0)
    regolith = build_regolith_field(terrain, root / "regolith")
    model = IlluminationModel(terrain, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    psr = model.psr_mask(window, 6.0 * 3600.0, semantics=PsrEpochSemantics.MISSION)  # type: ignore[arg-type]
    thermal = [diurnal_curve("polar_lit"), diurnal_curve("crater_floor")]
    return build_world_bundle(
        _spec(),
        terrain=terrain,
        regolith=regolith,
        psr=psr,
        thermal=thermal,
        out_dir=root / "bundle",
    )


def test_build_full_bundle(synthetic_dem: Path, synthetic_spice, tmp_path: Path) -> None:
    bundle = _full_bundle(synthetic_dem, tmp_path, synthetic_spice.window)

    assert set(bundle.component_hashes) == {
        "terrain",
        "regolith",
        "illumination",
        "illumination_horizon",  # the Zarr horizon-map field layer (issue #39)
        "thermal",
        "thermal_curves",  # the Zarr diurnal-curve field layer (issue #39)
    }
    assert (bundle.path / "regolith" / "bulk_density.tif").exists()
    assert (bundle.path / "illumination" / "psr_mask.tif").exists()
    assert (bundle.path / "illumination" / "horizon.zarr").is_dir()
    assert (bundle.path / "thermal" / "thermal.json").exists()
    assert (bundle.path / "thermal" / "curves.zarr").is_dir()
    thermal_doc = json.loads((bundle.path / "thermal" / "thermal.json").read_text())
    assert {c["terrain_class"] for c in thermal_doc["classes"]} == {"polar_lit", "crater_floor"}


def test_world_hash_is_reproducible(synthetic_dem: Path, synthetic_spice, tmp_path: Path) -> None:
    a = _full_bundle(synthetic_dem, tmp_path / "a", synthetic_spice.window)
    b = _full_bundle(synthetic_dem, tmp_path / "b", synthetic_spice.window)
    assert a.world_hash == b.world_hash
    # The whole bundle is byte-identical across builds (the "two clean checkouts" property).
    for rel in ("world.json", "stac/catalog.json", "tiles/tileset.json", "tiles/terrain.glb"):
        assert (a.path / rel).read_bytes() == (b.path / rel).read_bytes()


def test_world_hash_covers_the_psr_mask(
    synthetic_dem: Path, synthetic_spice, tmp_path: Path
) -> None:
    # RM-P1-WORLDS-15: the SPICE-derived PSR mask is folded into world_hash, so a bundle whose
    # shadow mask differs (e.g. from a kernel/ephemeris change) hashes differently — even though the
    # terrain horizon (illumination_hash) is unchanged.
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    model = IlluminationModel(terrain, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    psr = model.psr_mask(synthetic_spice.window, 6.0 * 3600.0, semantics=PsrEpochSemantics.MISSION)
    flipped = replace(psr, mask=~psr.mask)

    assert psr.illumination_hash == flipped.illumination_hash  # same terrain horizon
    assert psr.psr_hash != flipped.psr_hash  # ...but the PSR mask now moves the component hash

    b1 = build_world_bundle(_spec(), terrain=terrain, psr=psr, out_dir=tmp_path / "b1")
    b2 = build_world_bundle(_spec(), terrain=terrain, psr=flipped, out_dir=tmp_path / "b2")
    assert b1.world_hash != b2.world_hash


def test_world_hash_changes_with_spec(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    one = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "b1")
    two = build_world_bundle(
        _spec(world_id="other-world"), terrain=terrain, out_dir=tmp_path / "b2"
    )
    assert one.world_hash != two.world_hash


def test_crs_mismatch_is_rejected(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    other_crs = PlanetaryCRS(
        body=MOON,
        body_fixed_frame=MOON_BODY_FIXED.name,
        reference_radius_m=MOON_RADIUS_M,
        projection=f"+proj=stere +lat_0=90 +lon_0=0 +R={MOON_RADIUS_M:.1f} +units=m +no_defs",
    )
    with pytest.raises(ValueError, match="terrain CRS does not match"):
        build_world_bundle(_spec(crs=other_crs), terrain=terrain, out_dir=tmp_path / "bundle")


def test_psr_shape_mismatch_is_rejected(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bad_psr = PsrResult(
        mask=np.zeros((2, 2), dtype=np.bool_),
        ever_lit_fraction=0.0,
        void_mask=np.zeros((2, 2), dtype=np.bool_),
        window=None,  # type: ignore[arg-type]
        step_s=1.0,
        n_epochs=1,
        semantics=PsrEpochSemantics.MISSION,
        illumination_hash="sha256:deadbeef",
    )
    with pytest.raises(ValueError, match="PSR mask shape"):
        build_world_bundle(_spec(), terrain=terrain, psr=bad_psr, out_dir=tmp_path / "bundle")


# --- issue #36: the WorldSpec determines the PSR mask ----------------------------------------


def _declared_layers(**overrides: object) -> LayerSpec:
    """A LayerSpec that fully records its illumination parameters — the reproducible-anchor shape.

    ``psr_days=1.0`` matches the 24 h ``synthetic_spice.window`` the mask is sampled over, so the
    spec⇄mask window check passes on the happy path.
    """
    base: dict[str, object] = {
        "regolith_prior": "default_lunar",
        "illumination_n_azimuth": 16,
        "illumination_horizon_frame": "grid",
        "illumination_max_radius_m": 8000.0,
        "illumination_abcorr": "NONE",
        "psr_semantics": "mission",
        "psr_start": "2025-01-01T00:00:00Z",
        "psr_days": 1.0,
        "psr_step_hours": 6.0,
        "thermal_classes": ("polar_lit",),
    }
    base.update(overrides)
    return LayerSpec(**base)  # type: ignore[arg-type]


def _psr_spec(**layer_overrides: object) -> WorldSpec:
    return _spec(layers=_declared_layers(**layer_overrides))


def _psr_from_spec(spec: WorldSpec, terrain: object, window: object) -> PsrResult:
    """Build the PSR mask driven entirely by the spec's recorded parameters."""
    model = IlluminationModel.from_spec(spec, terrain)  # type: ignore[arg-type]
    step_hours = spec.layers.psr_step_hours or 6.0
    return model.psr_mask(
        window,  # type: ignore[arg-type]
        step_hours * 3600.0,
        semantics=PsrEpochSemantics(spec.layers.psr_semantics or "mission"),
    )


def test_new_layer_fields_round_trip_and_move_spec_hash() -> None:
    base = _spec()
    declared = _psr_spec()
    # Recording the parameters changes the declaration, hence spec_hash — they are load-bearing.
    assert declared.spec_hash != base.spec_hash
    again = WorldSpec.from_yaml_text(declared.to_yaml())
    assert again == declared
    assert again.layers.illumination_horizon_frame == "grid"
    assert again.layers.illumination_max_radius_m == 8000.0
    assert again.layers.illumination_abcorr == "NONE"
    assert again.layers.psr_days == 1.0
    assert again.layers.psr_step_hours == 6.0


def test_illumination_from_spec_reads_layer_parameters(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    model = IlluminationModel.from_spec(_psr_spec(), terrain)
    assert model.n_azimuth == 16
    assert model.max_radius_m == 8000.0
    assert model.abcorr == "NONE"
    assert model.horizon_frame.value == "grid"


def test_world_hash_is_reproducible_from_spec(
    synthetic_dem: Path, synthetic_spice: object, tmp_path: Path
) -> None:
    # The determinism gate (issue #36; worlds.md §10, conventions.md §11): two builds of the *same
    # fully-declared spec*, driven from the spec, yield the same world_hash. Unlike
    # test_world_hash_is_reproducible, construction reads the spec — so a mask-affecting parameter
    # that used to live outside it (horizon_frame / max_radius / abcorr / window / step) can no
    # longer move the hash between builds of one declaration.
    spec = _psr_spec()

    def _build(root: Path) -> WorldBundle:
        terrain = ingest_dem(synthetic_dem, root / "terrain", resolution_m=2000.0)
        regolith = build_regolith_field(terrain, root / "regolith")
        psr = _psr_from_spec(spec, terrain, synthetic_spice.window)  # type: ignore[attr-defined]
        return build_world_bundle(
            spec, terrain=terrain, regolith=regolith, psr=psr, out_dir=root / "bundle"
        )

    a = _build(tmp_path / "a")
    b = _build(tmp_path / "b")
    assert a.world_hash == b.world_hash
    assert (a.path / "world.json").read_bytes() == (b.path / "world.json").read_bytes()


def test_bundle_records_illumination_manifest(
    synthetic_dem: Path, synthetic_spice: object, tmp_path: Path
) -> None:
    # A published bundle carries illumination/manifest.json recording frame, radius, abcorr, window,
    # and step — the self-description that was missing (issue #36).
    spec = _psr_spec()
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    psr = _psr_from_spec(spec, terrain, synthetic_spice.window)  # type: ignore[attr-defined]
    bundle = build_world_bundle(spec, terrain=terrain, psr=psr, out_dir=tmp_path / "bundle")

    manifest = json.loads((bundle.path / "illumination" / "manifest.json").read_text())
    assert manifest["params"]["horizon_frame"] == "grid"
    assert manifest["params"]["max_radius_m"] == 8000.0
    assert manifest["params"]["abcorr"] == "NONE"
    assert manifest["psr"]["semantics"] == "mission"
    assert manifest["psr"]["step_s"] == 6.0 * 3600.0
    assert manifest["psr"]["psr_hash"] == bundle.component_hashes["illumination"]
    assert "start_tdb_s" in manifest["psr"]["window"]


@pytest.mark.parametrize(
    "override,needle",
    [
        ({"illumination_max_radius_m": 4000.0}, "illumination_max_radius_m"),
        ({"illumination_horizon_frame": "topocentric"}, "illumination_horizon_frame"),
        ({"illumination_abcorr": "LT+S"}, "illumination_abcorr"),
        ({"psr_step_hours": 12.0}, "psr_step_hours"),
        ({"psr_days": 2.0}, "psr_days"),
    ],
)
def test_build_rejects_mask_that_contradicts_spec(
    override: dict[str, object],
    needle: str,
    synthetic_dem: Path,
    synthetic_spice: object,
    tmp_path: Path,
) -> None:
    # The bundle refuses a PSR mask whose parameters disagree with the spec's declaration: the spec
    # must determine the mask, or world_hash is not reproducible from it (issue #36).
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    model = IlluminationModel(terrain, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    psr = model.psr_mask(
        synthetic_spice.window,  # type: ignore[attr-defined]
        6.0 * 3600.0,
        semantics=PsrEpochSemantics.MISSION,
    )
    spec = _psr_spec(**override)
    with pytest.raises(ValueError, match=needle):
        build_world_bundle(spec, terrain=terrain, psr=psr, out_dir=tmp_path / "bundle")


def test_undeclared_illumination_params_are_unconstrained(
    synthetic_dem: Path, synthetic_spice: object, tmp_path: Path
) -> None:
    # A spec that declares no illumination parameters (the pre-issue-#36 shape) is not rejected —
    # the check only bites on declared fields, so existing worlds keep building.
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    model = IlluminationModel(terrain, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    psr = model.psr_mask(
        synthetic_spice.window,  # type: ignore[attr-defined]
        6.0 * 3600.0,
        semantics=PsrEpochSemantics.MISSION,
    )
    bundle = build_world_bundle(_spec(), terrain=terrain, psr=psr, out_dir=tmp_path / "bundle")
    assert (bundle.path / "illumination" / "manifest.json").exists()


# --- STAC catalog ----------------------------------------------------------------------------


def test_stac_catalog_resolves_to_layer_cogs(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bundle = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")
    catalog = json.loads(bundle.stac_catalog.read_text())
    assert catalog["type"] == "Catalog"
    assert catalog["stac_version"] == "1.0.0"
    assert catalog["id"] == "shackleton-test"

    item_links = [link for link in catalog["links"] if link["rel"] == "item"]
    assert item_links
    stac_dir = bundle.stac_catalog.parent
    for link in item_links:
        item = json.loads((stac_dir / link["href"]).read_text())
        assert item["type"] == "Feature"
        asset_href = item["assets"]["data"]["href"]
        assert (stac_dir / asset_href).resolve().exists()  # the COG the item points at


# --- 3D-Tiles / glTF -------------------------------------------------------------------------


def _read_glb_json(path: Path) -> dict:
    raw = path.read_bytes()
    magic, version, _length = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67  # "glTF"
    assert version == 2
    chunk_len, chunk_type = struct.unpack_from("<II", raw, 12)
    assert chunk_type == 0x4E4F534A  # "JSON"
    return json.loads(raw[20 : 20 + chunk_len])


def test_tileset_and_glb_are_valid(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bundle = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")

    tileset = json.loads(bundle.tileset.read_text())
    assert tileset["asset"]["version"] == "1.1"
    assert tileset["root"]["content"]["uri"] == "terrain.glb"
    assert len(tileset["root"]["boundingVolume"]["box"]) == 12

    gltf = _read_glb_json(bundle.path / "tiles" / "terrain.glb")
    assert gltf["asset"]["version"] == "2.0"
    position = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert position["count"] > 0
    assert "min" in position and "max" in position


def test_heightfield_mesh_strides_and_fills_voids() -> None:
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    elevation = np.full((20, 20), np.nan, dtype=np.float64)  # all void → filled with 0.0
    mesh = heightfield_mesh(elevation, transform, max_dim=4)
    # Strided to <= 4 per axis: ceil(20/4)=5 step → indices 0,5,10,15 → 4x4 grid.
    assert mesh.vertex_count == 16
    assert mesh.triangle_count == 2 * 3 * 3  # (4-1)*(4-1) quads * 2 tris
    assert np.isfinite(mesh.positions).all()


# --- RM-P1-WORLDS-16: the published tileset-to-body transform ------------------------------


def _basis(root_transform: list[float]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split a column-major 4x4 into its east, north, up, and translation columns."""
    m = np.asarray(root_transform, dtype=np.float64).reshape(4, 4)
    return m[0, :3], m[1, :3], m[2, :3], m[3, :3]


@pytest.mark.parametrize(
    "longitude_deg,latitude_deg",
    [(0.0, -89.0107), (0.0, -90.0), (45.0, 12.5), (-179.9, 89.9), (120.0, 0.0)],
)
def test_enu_basis_is_orthonormal_and_right_handed(
    longitude_deg: float, latitude_deg: float
) -> None:
    """Including at the pole itself, where a cross-product construction would collapse."""
    east, north, up, _ = _basis(enu_to_body_fixed(longitude_deg, latitude_deg, 0.0, MOON_RADIUS_M))

    for axis in (east, north, up):
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-12)
    assert np.dot(east, north) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(east, up) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(north, up) == pytest.approx(0.0, abs=1e-12)
    # Right-handed: east x north = up.
    assert np.cross(east, north) == pytest.approx(up, abs=1e-12)


def test_enu_origin_sits_on_the_reference_sphere_raised_by_height() -> None:
    _, _, up, origin = _basis(enu_to_body_fixed(30.0, -60.0, 0.0, MOON_RADIUS_M))
    assert np.linalg.norm(origin) == pytest.approx(MOON_RADIUS_M, abs=1e-6)
    # The translation is radial: it is the up axis scaled by the body radius.
    assert origin == pytest.approx(np.asarray(up) * MOON_RADIUS_M, abs=1e-6)

    _, _, _, raised = _basis(enu_to_body_fixed(30.0, -60.0, 2_500.0, MOON_RADIUS_M))
    assert np.linalg.norm(raised) == pytest.approx(MOON_RADIUS_M + 2_500.0, abs=1e-6)


def test_enu_axes_point_the_way_their_names_say() -> None:
    """On the prime meridian at the equator: east is +y, north is +z, up is +x."""
    east, north, up, _ = _basis(enu_to_body_fixed(0.0, 0.0, 0.0, MOON_RADIUS_M))
    assert east == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert north == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert up == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


def test_heightfield_mesh_reports_the_strided_centroid() -> None:
    """The centroid is over the strided subsample, not the full grid — that gap is the whole bug."""
    transform = (20.0, 0.0, -2550.0, 0.0, -20.0, 32550.0)  # the View fixture's geotransform
    elevation = np.zeros((256, 256), dtype=np.float64)
    mesh = heightfield_mesh(elevation, transform, max_dim=64)

    centre_x, centre_y, mean_elevation = mesh.centroid
    # ceil(256/64) = 4 → cols 0,4,…,252 → mean col 126, not the grid centre 127.5.
    assert centre_x == pytest.approx(20.0 * 126 - 2550.0)
    assert centre_y == pytest.approx(-20.0 * 126 + 32550.0)
    # 30 m from the grid centre: exactly the offset a consumer reconstructing from `grid` inherits.
    assert abs(centre_x - 0.0) == pytest.approx(30.0)
    assert mean_elevation == pytest.approx(0.0)


def test_heightfield_mesh_centroid_carries_the_subtracted_mean_elevation() -> None:
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    elevation = np.full((8, 8), -1234.5, dtype=np.float64)
    mesh = heightfield_mesh(elevation, transform, max_dim=8)

    assert mesh.centroid[2] == pytest.approx(-1234.5)
    # Every vertex height is relative to that mean, which is why it must be published.
    assert mesh.positions[:, 1] == pytest.approx(np.zeros(mesh.vertex_count), abs=1e-6)


def test_tileset_publishes_a_body_fixed_root_transform(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bundle = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")

    root = json.loads(bundle.tileset.read_text())["root"]
    assert "transform" in root, "the tileset-to-body transform must be published, not left identity"
    assert len(root["transform"]) == 16

    east, north, up, origin = _basis(root["transform"])
    assert np.cross(east, north) == pytest.approx(up, abs=1e-9)

    anchor = bundle.manifest["tiles_anchor"]
    # RM-P1-WORLDS-17: the anchor frame is a typed Core ReferenceFrame (not a bare string) — for the
    # lunar CRS it is exactly MOON_BODY_FIXED.
    assert anchor["frame"] == MOON_BODY_FIXED.model_dump(mode="json")
    assert anchor["frame"]["name"] == MOON_BODY_FIXED.name
    # The manifest anchor and the tileset transform describe the same point on the body.
    expected = np.asarray(
        _basis(
            enu_to_body_fixed(
                anchor["origin"]["longitude_deg"],
                anchor["origin"]["latitude_deg"],
                anchor["origin"]["height_m"],
                MOON_RADIUS_M,
            )
        )[3]
    )
    assert origin == pytest.approx(expected, abs=1e-6)


def test_tiles_anchor_matches_an_independent_inverse_projection(
    synthetic_dem: Path, tmp_path: Path
) -> None:
    """Cross-check the published anchor against rasterio, not against our own forward code."""
    import rasterio.warp

    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bundle = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")

    grid = bundle.manifest["grid"]
    a, b, c, d, e, f = grid["transform"]
    step_col = -(-grid["width"] // 64)
    step_row = -(-grid["height"] // 64)
    mean_col = float(np.arange(0, grid["width"], step_col).mean())
    mean_row = float(np.arange(0, grid["height"], step_row).mean())
    centre_x = a * mean_col + b * mean_row + c
    centre_y = d * mean_col + e * mean_row + f

    lons, lats = rasterio.warp.transform(
        rasterio.crs.CRS.from_proj4(LUNAR_SOUTH_POLAR_STEREOGRAPHIC.projection or ""),
        rasterio.crs.CRS.from_proj4(f"+proj=longlat +R={MOON_RADIUS_M:.1f} +no_defs"),
        [centre_x],
        [centre_y],
    )
    origin = bundle.manifest["tiles_anchor"]["origin"]
    assert origin["longitude_deg"] == pytest.approx(lons[0], abs=1e-9)
    assert origin["latitude_deg"] == pytest.approx(lats[0], abs=1e-9)


def test_tiles_anchor_height_is_the_subtracted_patch_mean(
    synthetic_dem: Path, tmp_path: Path
) -> None:
    """A consumer adding a vertex height to the anchor height recovers an absolute elevation."""
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bundle = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")

    with rasterio.open(bundle.path / "terrain" / "elevation.tif") as ds:
        elevation = ds.read(1).astype(np.float64)
    mesh = heightfield_mesh(elevation, tuple(bundle.manifest["grid"]["transform"]), max_dim=64)

    height_m = bundle.manifest["tiles_anchor"]["origin"]["height_m"]
    assert height_m == pytest.approx(mesh.centroid[2])
    assert height_m != 0.0, "a nonzero datum is the point: it cannot be reconstructed downstream"


def test_tiles_field_stays_a_relative_path_string(synthetic_dem: Path, tmp_path: Path) -> None:
    """`tiles` must not become an object: WorldBundle.load and View both read it as a path."""
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bundle = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")

    assert bundle.manifest["tiles"] == "tiles/tileset.json"
    assert WorldBundle.load(bundle.path).tileset == bundle.path / "tiles" / "tileset.json"


def test_root_transform_places_vertices_within_the_tangent_plane_bound() -> None:
    """The transform is exact at the anchor and degrades only by the tile's own curvature.

    A single east-north-up frame is a *tangent plane*: a vertex `d` metres from the anchor sits
    above the sphere by the sagitta `d^2 / 2R`. That residual is inherent to a flat tile (it is
    what Cesium's own `eastNorthUpToFixedFrame` does too) and is the honest error bound — roughly
    3.7 m at the corners of a 5 km patch, against the 30 m planimetric and ~680 m vertical errors a
    consumer inherited when the transform was left identity.
    """
    transform = (20.0, 0.0, -2550.0, 0.0, -20.0, 32550.0)  # the View fixture's geotransform
    rows, cols = np.meshgrid(np.arange(256.0), np.arange(256.0), indexing="ij")
    radius = np.hypot((cols - 127.5) / 127.5, (rows - 127.5) / 127.5)
    elevation = -1200.0 * np.exp(-((radius / 0.55) ** 2))

    mesh = heightfield_mesh(elevation, transform, max_dim=64)
    centre_x, centre_y, mean_elevation = mesh.centroid
    longitude_deg, latitude_deg = to_lonlat(LUNAR_SOUTH_POLAR_STEREOGRAPHIC, centre_x, centre_y)
    east, north, up, origin = _basis(
        enu_to_body_fixed(longitude_deg, latitude_deg, mean_elevation, MOON_RADIUS_M)
    )

    half_diagonal = float(np.hypot(mesh.box[3], mesh.box[11]))
    bound_m = half_diagonal**2 / (2.0 * MOON_RADIUS_M)

    for index in (0, 63, 64 * 64 - 1):
        g_x, g_y, g_z = mesh.positions[index].astype(np.float64)
        # glTF Y-up -> 3D-Tiles Z-up: (x, y, z) -> (x, -z, y), i.e. (east, north, up).
        point = origin + g_x * east + (-g_z) * north + g_y * up

        row, col = (index // 64) * 4, (index % 64) * 4
        expected_elevation = elevation[row, col]
        actual_elevation = float(np.linalg.norm(point)) - MOON_RADIUS_M

        assert actual_elevation == pytest.approx(expected_elevation, abs=1.05 * bound_m)

    # And the bound really is small: sub-4 m over this patch, not the hundreds of metres before.
    assert bound_m < 4.0


# --- RM-P1-WORLDS-17: pin world.json crs / tiles_anchor.frame to Core's units schema ----------


def _core_units_schema() -> dict[str, Any]:
    """Core's canonical units schema, via Core's public accessor (RFC-0009 §2)."""
    return core_schema("astro_mine.core.units", "units.schema.json")


def _units_validator(defname: str) -> Draft202012Validator:
    """A validator (independent of the Worlds emit path) for a units.schema.json ``$def``."""
    units = _core_units_schema()
    return Draft202012Validator(
        {"$ref": units["$id"] + f"#/$defs/{defname}"}, registry=schema_registry()
    )


def test_world_json_crs_and_anchor_frame_validate_against_units_schema(
    synthetic_dem: Path, tmp_path: Path
) -> None:
    """AC1: the emitted world.json crs and tiles_anchor.frame validate against units.schema.json."""
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bundle = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")

    manifest = json.loads((bundle.path / "world.json").read_text())

    # The manifest publishes the Core schema references its units objects conform to.
    assert manifest["units_schema"]["id"] == _core_units_schema()["$id"]
    assert manifest["units_schema"]["crs"].endswith("#/$defs/PlanetaryCRS")
    assert manifest["units_schema"]["tiles_anchor_frame"].endswith("#/$defs/ReferenceFrame")

    # ...and the objects actually validate against those $defs (checked independently of emit).
    assert list(_units_validator("PlanetaryCRS").iter_errors(manifest["crs"])) == []
    assert (
        list(_units_validator("ReferenceFrame").iter_errors(manifest["tiles_anchor"]["frame"]))
        == []
    )


def test_worldspec_rejects_an_earth_datum_on_the_moon_at_emit() -> None:
    """AC2: a body="MOON" CRS carrying a WGS84 datum is rejected at emit, not at View's ingest."""
    earth_datum_on_moon = PlanetaryCRS(
        body=MOON,
        body_fixed_frame=MOON_BODY_FIXED.name,
        reference_radius_m=1_737_400.0,
        datum="WGS84",
    )
    # WorldSpec construction is the emit-side declaration; require_crs rule 6 refuses it here.
    with pytest.raises(UnitsValidationError, match="Earth CRS marker"):
        _spec(crs=earth_datum_on_moon)


def test_validate_units_object_rejects_a_malformed_frame() -> None:
    """The emit-time schema guard raises (listing errors) on a non-conforming units object."""
    with pytest.raises(WorldSchemaError, match="ReferenceFrame"):
        validate_units_object({"name": "MOON_ME", "frame_class": "not_a_class"}, "ReferenceFrame")


def test_worldspec_accepts_an_earth_crs_when_the_body_is_earth() -> None:
    """Rule 6 is a body/datum *consistency* check: an Earth datum is valid on body EARTH."""
    earth_crs = PlanetaryCRS(
        body="EARTH",
        body_fixed_frame="ITRF93",
        reference_radius_m=6_378_137.0,
        datum="WGS84",
    )
    spec = _spec(crs=earth_crs)  # constructs without raising
    assert spec.crs.body == "EARTH"
