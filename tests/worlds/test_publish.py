"""RM-P1-WORLDS-15 — Hub-publish the world bundle + the Core ``world_provider`` manifest.

Proves the deliverable and its acceptance (worlds.md §3, §5; hub.md §3, §9):

- :func:`publish_world_bundle` stores the bundle to a **local OCI-layout** ``Registry`` as a signed
  ``world`` artifact whose config is a Core ``world_provider`` :class:`PluginManifest`
  (``provenance.digest == world_hash``); a ``HubClient`` verifies + pulls it, **failing closed** on
  a tampered blob;
- a consumer resolves the world **by content hash** and rebuilds a live ``WorldProvider``
  (``sample`` / ``line_of_sight`` / PSR mask) through the ``astro_mine.providers`` entry point —
  **without importing** ``astro_mine.worlds`` in the resolve path;
- the deterministic bundle tar is byte-stable, so the same bundle yields the same layer digest
  ("two clean checkouts resolve the identical world").

Fully offline — a local registry directory under ``tmp_path``; no hosted Hub / Cloud.
"""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import rasterio.transform

from astro_mine.core.registry import (
    PluginKind,
    PluginManifest,
    PluginRegistry,
    Provenance,
)
from astro_mine.core.world import SurfacePoint, check_world_provider
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Registry, artifact_media_type
from astro_mine.hub.supply_chain import SupplyChainError, generate_keypair
from astro_mine.worlds.crs import LUNAR_SOUTH_POLAR_STEREOGRAPHIC
from astro_mine.worlds.illumination import IlluminationModel, PsrEpochSemantics
from astro_mine.worlds.provider import DemWorldProvider
from astro_mine.worlds.regolith import build_regolith_field
from astro_mine.worlds.spec import (
    BUNDLE_LAYER_MEDIA_TYPE,
    WORLD_ARTIFACT_KIND,
    LayerSpec,
    Region,
    SourceRef,
    WorldBundle,
    WorldSpec,
    build_world_bundle,
    build_world_manifest,
    publish_world_bundle,
)
from astro_mine.worlds.spec._publish import deterministic_bundle_tar
from astro_mine.worlds.terrain import ingest_dem

_NAME = "shackleton-de-gerlache"
_VERSION = "0.1.0"


def _spec(**overrides: object) -> WorldSpec:
    base: dict[str, object] = {
        "world_id": "shackleton-test",
        "crs": LUNAR_SOUTH_POLAR_STEREOGRAPHIC,
        "region": Region(
            min_x_m=-30_000.0,
            min_y_m=-30_000.0,
            max_x_m=30_000.0,
            max_y_m=30_000.0,
            resolution_m=2000.0,
        ),
        "source_dem": SourceRef(id="synthetic-lola", description="CI stand-in DEM"),
        "layers": LayerSpec(regolith_prior="default_lunar", thermal_classes=("polar_lit",)),
        "description": "test world",
    }
    base.update(overrides)
    return WorldSpec(**base)  # type: ignore[arg-type]


def _full_bundle(dem: Path, root: Path, window: object) -> WorldBundle:
    terrain = ingest_dem(dem, root / "terrain", resolution_m=2000.0)
    regolith = build_regolith_field(terrain, root / "regolith")
    model = IlluminationModel(terrain, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    psr = model.psr_mask(window, 6.0 * 3600.0, semantics=PsrEpochSemantics.MISSION)  # type: ignore[arg-type]
    return build_world_bundle(
        _spec(), terrain=terrain, regolith=regolith, psr=psr, out_dir=root / "bundle"
    )


def _layers_by_media_type(registry: Registry, digest: str) -> dict[str, bytes]:
    """Reconstruct the ``mediaType -> bytes`` layer map a consumer feeds ``from_bundle``."""
    image = registry.read_manifest(digest)
    return {layer["mediaType"]: registry.pull_blob(layer["digest"]) for layer in image["layers"]}


def _surface_position(provider: DemWorldProvider) -> tuple[float, float, float]:
    """A body-fixed position on the terrain surface at the grid centre."""
    row, col = provider._height // 2, provider._width // 2
    map_x, map_y = rasterio.transform.xy(provider._transform, row, col)
    map_x, map_y = float(map_x), float(map_y)
    sample = provider.terrain.sample(map_x, map_y)
    lon, lat = provider._map_to_lonlat(map_x, map_y)
    radius = provider._radius_m + float(sample.elevation_m)
    lon_r, lat_r = math.radians(lon), math.radians(lat)
    return (
        radius * math.cos(lat_r) * math.cos(lon_r),
        radius * math.cos(lat_r) * math.sin(lon_r),
        radius * math.sin(lat_r),
    )


def _provider_factory() -> Callable[[PluginManifest, Mapping[str, bytes]], Any]:
    """Load the world_provider factory the way a consumer does — via the entry point only."""
    return importlib.metadata.entry_points(group="astro_mine.providers")["world_provider"].load()


# --- self-describing bundle (the re-openable gap this issue closes) --------------------------


def test_bundle_products_carry_their_manifests(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    build_regolith_field(terrain, tmp_path / "regolith")
    regolith = tmp_path / "regolith"
    bundle = build_world_bundle(
        _spec(), terrain=terrain, regolith=regolith, out_dir=tmp_path / "bundle"
    )
    # The gap RM-P1-WORLDS-15 closes: each product dir is self-describing and re-openable.
    assert (bundle.path / "terrain" / "manifest.json").exists()
    assert (bundle.path / "regolith" / "manifest.json").exists()
    reopened = DemWorldProvider.open(bundle.path / "terrain", bundle.path / "regolith")
    check_world_provider(reopened)


def test_world_bundle_load_round_trips(synthetic_dem: Path, tmp_path: Path) -> None:
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    built = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")
    loaded = WorldBundle.load(built.path)
    assert loaded.world_hash == built.world_hash
    assert loaded.world_id == built.world_id
    assert loaded.component_hashes == built.component_hashes
    assert loaded.spec == built.spec


# --- the Core world_provider manifest --------------------------------------------------------


def test_build_world_manifest_declares_world_provider(
    synthetic_dem: Path, synthetic_spice: Any, tmp_path: Path
) -> None:
    bundle = _full_bundle(synthetic_dem, tmp_path, synthetic_spice.window)
    manifest = build_world_manifest(bundle, name=_NAME, version=_VERSION)

    assert manifest.kind == PluginKind.WORLD_PROVIDER
    assert manifest.core_interfaces == {"world_provider": "0.1.0"}
    assert manifest.license == "Apache-2.0"
    assert manifest.provenance is not None
    assert manifest.provenance.digest == bundle.world_hash
    assert manifest.attributes["bundle_media_type"] == BUNDLE_LAYER_MEDIA_TYPE
    # The manifest passes the full Core load-time gate (negotiation), i.e. is registrable.
    PluginRegistry(require_signature=False).register(manifest)


# --- publish -> resolve-by-digest -> rebuild -------------------------------------------------


def test_publish_signs_and_verifies(
    synthetic_dem: Path, synthetic_spice: Any, tmp_path: Path
) -> None:
    bundle = _full_bundle(synthetic_dem, tmp_path, synthetic_spice.window)
    private_pem, public_pem = generate_keypair()
    registry = Registry(tmp_path / "registry")

    artifact = publish_world_bundle(
        bundle, registry, private_key_pem=private_pem, name=_NAME, version=_VERSION
    )
    assert artifact.reference == f"{_NAME}:{_VERSION}"
    assert registry.read_manifest(artifact.digest)["artifactType"] == artifact_media_type(
        WORLD_ARTIFACT_KIND
    )

    # A client with the trusted key verifies the signature + SLSA + SBOM, fail-closed.
    client = HubClient(registry, trusted_public_key_pem=public_pem)
    config = client.pull(artifact.digest)  # re-verifies before returning bytes
    manifest = PluginManifest.model_validate_json(config)
    assert manifest.provenance is not None and manifest.provenance.digest == bundle.world_hash


def test_resolve_by_digest_rebuilds_live_provider(
    synthetic_dem: Path, synthetic_spice: Any, tmp_path: Path
) -> None:
    bundle = _full_bundle(synthetic_dem, tmp_path, synthetic_spice.window)
    private_pem, public_pem = generate_keypair()
    registry = Registry(tmp_path / "registry")
    artifact = publish_world_bundle(
        bundle, registry, private_key_pem=private_pem, name=_NAME, version=_VERSION
    )

    # Consumer path: pull + verify the manifest by content hash, fetch the layers, and rebuild the
    # provider through the entry point — no astro_mine.worlds import here.
    client = HubClient(registry, trusted_public_key_pem=public_pem)
    manifest = PluginManifest.model_validate_json(client.pull(artifact.digest))
    layers = _layers_by_media_type(registry, artifact.digest)
    assert set(layers) == {BUNDLE_LAYER_MEDIA_TYPE}

    provider = _provider_factory()(manifest, layers)
    check_world_provider(provider)

    position = _surface_position(provider)
    point = provider.sample(position)
    assert isinstance(point, SurfacePoint)
    los = provider.line_of_sight(position, tuple(2.0 * c for c in position))
    assert isinstance(los, bool)

    # PSR round-trip: the illumination mask survives publish -> pull byte-for-byte.
    pulled_psr = _read_bundle_member(layers[BUNDLE_LAYER_MEDIA_TYPE], "illumination/psr_mask.tif")
    assert pulled_psr == (bundle.path / "illumination" / "psr_mask.tif").read_bytes()

    # Horizon round-trip (issue #39): the rebuilt provider **adopts the persisted Zarr horizon map**
    # instead of re-deriving the whole skyline in-process, and reconstructs the *bundle's* model —
    # its recorded parameters (n_azimuth=16, max_radius_m=8000, abcorr=NONE), hence its
    # illumination_hash — rather than one built from library defaults.
    illumination = provider.illumination
    assert illumination.horizon_source == "stored"
    assert illumination.n_azimuth == 16
    assert illumination.max_radius_m == 8000.0
    assert illumination.abcorr == "NONE"
    source = IlluminationModel(
        bundle.path / "terrain", n_azimuth=16, max_radius_m=8000.0, abcorr="NONE"
    )
    assert illumination.illumination_hash == source.illumination_hash


def _read_bundle_member(tar_bytes: bytes, member: str) -> bytes:
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
        extracted = tar.extractfile(member)
        assert extracted is not None
        return extracted.read()


# --- determinism -----------------------------------------------------------------------------


def test_bundle_tar_is_deterministic(
    synthetic_dem: Path, synthetic_spice: Any, tmp_path: Path
) -> None:
    a = _full_bundle(synthetic_dem, tmp_path / "a", synthetic_spice.window)
    b = _full_bundle(synthetic_dem, tmp_path / "b", synthetic_spice.window)
    assert deterministic_bundle_tar(a.path) == deterministic_bundle_tar(b.path)

    # Same bundle -> same Hub layer digest across two independent registries.
    private_pem, _ = generate_keypair()
    art_a = publish_world_bundle(
        a, Registry(tmp_path / "ra"), private_key_pem=private_pem, name=_NAME, version=_VERSION
    )
    art_b = publish_world_bundle(
        b, Registry(tmp_path / "rb"), private_key_pem=private_pem, name=_NAME, version=_VERSION
    )
    assert art_a.digest == art_b.digest


# --- fail-closed ------------------------------------------------------------------------------


def test_pull_fails_closed_on_tampered_payload(
    synthetic_dem: Path, synthetic_spice: Any, tmp_path: Path
) -> None:
    bundle = _full_bundle(synthetic_dem, tmp_path, synthetic_spice.window)
    private_pem, public_pem = generate_keypair()
    registry = Registry(tmp_path / "registry")
    artifact = publish_world_bundle(
        bundle, registry, private_key_pem=private_pem, name=_NAME, version=_VERSION
    )

    # Corrupt the payload layer blob in place (its filename still claims the original digest).
    layer_digest = registry.read_manifest(artifact.digest)["layers"][0]["digest"]
    tampered = registry.path / "blobs" / "sha256" / layer_digest.split(":", 1)[1]
    tampered.write_bytes(tampered.read_bytes() + b"\x00tamper")

    # pull() re-runs the supply-chain check before returning bytes; the integrity failure surfaces
    # as SupplyChainError (fail-closed) — a compromised registry cannot serve tampered world bytes.
    client = HubClient(registry, trusted_public_key_pem=public_pem)
    with pytest.raises(SupplyChainError):
        client.pull(artifact.digest)


# --- from_bundle error paths -----------------------------------------------------------------


def test_from_bundle_rejects_missing_layer() -> None:
    manifest = PluginManifest(
        name=_NAME,
        version=_VERSION,
        kind=PluginKind.WORLD_PROVIDER,
        core_interfaces={"world_provider": "0.1.0"},
        provenance=Provenance(digest="sha256:deadbeef"),
    )
    with pytest.raises(ValueError, match=r"no .* layer"):
        DemWorldProvider.from_bundle(manifest, {})


def test_from_bundle_rejects_non_world_manifest() -> None:
    manifest = PluginManifest(
        name="not-a-world",
        version=_VERSION,
        kind=PluginKind.POLICY,
        core_interfaces={"policy": "0.1.0"},
    )
    with pytest.raises(ValueError, match="not a"):
        DemWorldProvider.from_bundle(manifest, {BUNDLE_LAYER_MEDIA_TYPE: b""})


def test_from_bundle_requires_regolith(synthetic_dem: Path, tmp_path: Path) -> None:
    # A terrain-only bundle has no regolith product, so the provider cannot be rebuilt.
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    bundle = build_world_bundle(_spec(), terrain=terrain, out_dir=tmp_path / "bundle")
    private_pem, _ = generate_keypair()
    registry = Registry(tmp_path / "registry")
    artifact = publish_world_bundle(
        bundle, registry, private_key_pem=private_pem, name=_NAME, version=_VERSION
    )
    manifest = build_world_manifest(bundle, name=_NAME, version=_VERSION)
    layers = _layers_by_media_type(registry, artifact.digest)
    with pytest.raises(ValueError, match="regolith"):
        DemWorldProvider.from_bundle(manifest, layers)


# --- the `worlds` CLI ------------------------------------------------------------------------


def test_cli_publishes_a_signed_bundle(
    synthetic_dem: Path, synthetic_spice: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from astro_mine.hub.supply_chain import generate_keypair
    from astro_mine.worlds.cli import main

    bundle = _full_bundle(synthetic_dem, tmp_path, synthetic_spice.window)
    # The signing key comes from `astro-mine-hub keygen` (the one signing-key command); here we mint
    # one directly from the same primitive rather than routing through another package's CLI.
    key = tmp_path / "cosign.key"
    key.write_bytes(generate_keypair()[0])

    registry = tmp_path / "registry"
    code = main(
        [
            "publish",
            str(bundle.path),
            "--registry",
            str(registry),
            "--key",
            str(key),
            "--name",
            _NAME,
            "--version",
            _VERSION,
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert f"published {_NAME}:{_VERSION}" in out
    # The published artifact is resolvable in the registry the CLI wrote.
    assert Registry(registry).references() == [f"{_NAME}:{_VERSION}"]


def test_cli_requires_a_subcommand() -> None:
    from astro_mine.worlds.cli import main

    with pytest.raises(SystemExit):
        main([])
