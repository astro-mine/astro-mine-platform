"""RM-P1-PROSPECT-13 — Hub-publish the parametric prior + reopen it as a live ResourceField.

Proves the acceptance criteria (prospect.md §3, §4, §6; hub.md §3, §9; conventions.md §1.1):

- **Serialize + publish** — ``load_prior('shackleton_water_ice_v1')`` serializes to a
  content-addressed bundle and publishes to a local ``Registry`` with a Core
  ``resource_field_backend`` manifest + signature/SLSA/SBOM; ``verify``/``pull`` round-trip
  **fail-closed** on tamper.
- **Resolve without importing prospect** — a consumer resolves the field by digest and rebuilds a
  live ``ResourceField`` (uncertainty-first ``posterior``/``mean``/``variance``) through the
  ``astro_mine.providers`` entry point, using only Core + Hub.
- **Security-class invariant (``RM-P0-PROSPECT-05``)** — only the public belief prior is published;
  the sealed ``GroundTruthField`` never appears in the bundle.
- **Determinism** — the bundle bytes and the published digest are identical across clean publishes.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pytest

from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.core.resource import ResourceField, check_resource_field
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Registry, open_registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import SupplyChainError, generate_keypair
from astro_mine.prospect.belief import sample_ground_truth
from astro_mine.prospect.field import FieldGrid
from astro_mine.prospect.isolation import GROUND_TRUTH_ACCESS
from astro_mine.prospect.priors import artifact_name_for, load_prior
from astro_mine.prospect.publish import (
    BUNDLE_MEDIA_TYPE,
    METADATA_MEMBER,
    PROVENANCE_MEMBER,
    PROVIDER_ENTRY_POINT,
    build_field_manifest,
    bundle_digest,
    from_bundle,
    publish_prior,
    serialize_bundle,
)

_ANCHOR = "shackleton_water_ice_v1"
_MEMBERS = {"mean.npy", "variance.npy", METADATA_MEMBER, PROVENANCE_MEMBER}
_PROBES = [(0.0, 0.0, 0.0), (500.0, -500.0, 0.0), (-900.0, 900.0, 0.0)]


def _small_grid() -> FieldGrid:
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


def _resource_field_factory():  # type: ignore[no-untyped-def]
    """The published factory, resolved the way a consumer would (no prospect import by name)."""
    (ep,) = entry_points(group="astro_mine.providers", name=PROVIDER_ENTRY_POINT)
    return ep.load()


def _layers_from_registry(registry: Registry, digest: str) -> dict[str, bytes]:
    """Build the ``media_type -> bytes`` layer mapping a resolver hands to ``from_bundle``."""
    image = registry.read_manifest(digest)
    return {layer["mediaType"]: registry.pull_blob(layer["digest"]) for layer in image["layers"]}


# --- serialization: determinism + shape ------------------------------------------------------


def test_bundle_is_byte_deterministic() -> None:
    a, b = serialize_bundle(load_prior(_ANCHOR)), serialize_bundle(load_prior(_ANCHOR))
    assert a == b  # two independent fits → identical bytes → a stable content address


def test_bundle_holds_exactly_the_four_members() -> None:
    with tarfile.open(fileobj=io.BytesIO(serialize_bundle(load_prior(grid=_small_grid())))) as tar:
        assert {m.name for m in tar.getmembers()} == _MEMBERS


def test_tar_entries_are_normalized_for_reproducibility() -> None:
    with tarfile.open(fileobj=io.BytesIO(serialize_bundle(load_prior(grid=_small_grid())))) as tar:
        for member in tar.getmembers():
            assert member.mtime == 0
            assert member.uid == 0 and member.gid == 0
            assert member.uname == "" and member.gname == ""


# --- from_bundle: rebuild a live field without re-running the recipe --------------------------


def test_from_bundle_round_trips_the_field() -> None:
    prior = load_prior(grid=_small_grid())
    manifest = build_field_manifest(prior, bundle_sha256=bundle_digest(serialize_bundle(prior)))
    field = from_bundle(manifest, {BUNDLE_MEDIA_TYPE: serialize_bundle(prior)})
    reference = prior.as_field()

    assert isinstance(field, ResourceField)
    assert check_resource_field(field) is None
    assert field.species == reference.species and field.unit == reference.unit
    for probe in _PROBES:
        post = field.posterior(probe)
        assert post.mean == pytest.approx(reference.mean(probe))
        assert post.variance == pytest.approx(reference.variance(probe))
        assert post.variance > 0.0  # uncertainty is preserved (a belief prior, not sealed truth)
        assert field.quantile(probe, 0.05) < post.mean < field.quantile(probe, 0.95)


def test_from_bundle_missing_layer_fails_loudly() -> None:
    # `from_bundle` now resolves *either* the .npy tar or a Zarr store (prospect.md §5), so the
    # error names both media types it would have accepted.
    manifest = build_field_manifest(load_prior(grid=_small_grid()), bundle_sha256="sha256:0")
    with pytest.raises(ValueError, match="no resource-field layer found"):
        from_bundle(manifest, {})


def test_from_bundle_missing_member_fails_loudly() -> None:
    prior = load_prior(grid=_small_grid())
    manifest = build_field_manifest(prior, bundle_sha256="sha256:0")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = b"{}"
        info = tarfile.TarInfo(METADATA_MEMBER)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="missing required members"):
        from_bundle(manifest, {BUNDLE_MEDIA_TYPE: buf.getvalue()})


def test_from_bundle_ignores_non_file_members() -> None:
    prior = load_prior(grid=_small_grid())
    with tarfile.open(fileobj=io.BytesIO(serialize_bundle(prior))) as src:
        payloads = {m.name: src.extractfile(m).read() for m in src.getmembers()}  # type: ignore[union-attr]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        directory = tarfile.TarInfo("nested")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)  # a non-regular member the reader must skip, not choke on
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    manifest = build_field_manifest(prior, bundle_sha256="sha256:0")
    field = from_bundle(manifest, {BUNDLE_MEDIA_TYPE: buf.getvalue()})
    assert check_resource_field(field) is None


# --- the Core resource_field_backend manifest ------------------------------------------------


def test_manifest_declares_the_resource_field_backend_contract() -> None:
    prior = load_prior(_ANCHOR)
    manifest = build_field_manifest(prior, bundle_sha256=bundle_digest(serialize_bundle(prior)))
    assert manifest.kind is PluginKind.RESOURCE_FIELD_BACKEND
    assert manifest.core_interfaces == {"resource_field": "0.1.0"}
    assert manifest.license == "Apache-2.0"
    assert manifest.provenance is not None
    assert manifest.provenance.digest == f"sha256:{prior.content_hash}"
    assert manifest.capability_tags == []  # open-commons: no gated tag on a public belief prior
    assert manifest.attributes["bundle_media_type"] == BUNDLE_MEDIA_TYPE
    assert manifest.attributes["provider_entry_point"] == PROVIDER_ENTRY_POINT


# --- the security-class invariant: ground truth is never in the bundle (RM-P0-PROSPECT-05) ----


def test_ground_truth_is_never_serialized_into_the_bundle() -> None:
    grant = (GROUND_TRUTH_ACCESS,)
    prior = load_prior(grid=_small_grid())
    truth = sample_ground_truth(prior, seed=7, capabilities=grant)
    realization = truth.reveal(capabilities=grant)

    bundle = serialize_bundle(prior)

    # 1. No member carries the sealed realization's bytes (however it might be shaped/dtyped).
    assert realization.tobytes() not in bundle
    assert np.ascontiguousarray(realization, dtype=np.float64).tobytes() not in bundle

    # 2. The bundle exposes only the four belief-prior members — nothing named for ground truth.
    with tarfile.open(fileobj=io.BytesIO(bundle)) as tar:
        names = {m.name for m in tar.getmembers()}
    assert names == _MEMBERS
    assert not any("truth" in n or "seal" in n or "realiz" in n for n in names)

    # 3. What reopens is the belief prior (variance > 0 everywhere), never the degenerate
    #    zero-variance sealed truth — the published arrays are the prior's, not the realization.
    manifest = build_field_manifest(prior, bundle_sha256=bundle_digest(bundle))
    field = from_bundle(manifest, {BUNDLE_MEDIA_TYPE: bundle})
    for probe in _PROBES:
        assert field.variance(probe) > 0.0
        assert field.mean(probe) == pytest.approx(prior.as_field().mean(probe))
        assert field.mean(probe) != pytest.approx(truth.mean(probe))  # not the realized truth


# --- end-to-end through Hub: publish, verify, resolve-by-digest, rebuild ----------------------


def test_publish_verify_pull_and_reopen_via_entry_point(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair()
    prior = load_prior(grid=_small_grid())
    artifact = publish_prior(prior, registry=open_registry(str(tmp_path / "reg")),
        private_key_pem=private_pem)
    # `publish_prior` defaults to the recipe's *published artifact name*, not its registry
    # key -- the key is a Python-side identifier and stays snake_case, while §13 governs what
    # the bytes are addressed by. Defaulting to the key is what put `shackleton_water_ice_pds_v1`
    # in the registry (astro-mine-platform#34).
    assert artifact.reference == f"{artifact_name_for(_ANCHOR)}:1.0.0"
    assert artifact.reference == "shackleton-water-ice:1.0.0"

    # A consumer opens the *same* registry with the trusted key — only Core + Hub, no prospect.
    consumer = HubClient(Registry(tmp_path / "reg"), trusted_public_key_pem=public_pem)
    assert consumer.verify(artifact.digest) == artifact.digest  # verify-twice, signed, fail-closed
    manifest = PluginManifest.model_validate_json(consumer.pull(artifact.digest))
    assert manifest.kind is PluginKind.RESOURCE_FIELD_BACKEND

    layers = _layers_from_registry(consumer.registry, artifact.digest)
    field = _resource_field_factory()(manifest, layers)
    assert check_resource_field(field) is None
    for probe in _PROBES:
        assert field.posterior(probe).mean == pytest.approx(prior.as_field().mean(probe))
        assert field.posterior(probe).variance == pytest.approx(prior.as_field().variance(probe))


def test_two_clean_publishes_resolve_the_identical_digest(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    prior = load_prior(_ANCHOR)
    one = publish_prior(prior, registry=open_registry(str(tmp_path / "a")),
        private_key_pem=private_pem)
    two = publish_prior(prior, registry=open_registry(str(tmp_path / "b")),
        private_key_pem=private_pem)
    assert one.digest == two.digest  # reproducible: two checkouts pin the same field


def test_tampered_bundle_fails_closed(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair()
    prior = load_prior(grid=_small_grid())
    artifact = publish_prior(prior, registry=open_registry(str(tmp_path / "reg")),
        private_key_pem=private_pem)

    registry = Registry(tmp_path / "reg")
    config_digest = registry.read_manifest(artifact.digest)["config"]["digest"]
    blob_path(registry.path, config_digest).write_bytes(b'{"tampered":true}')

    consumer = HubClient(registry, trusted_public_key_pem=public_pem)
    with pytest.raises(SupplyChainError):
        consumer.pull(artifact.digest)


# --- the `prospect publish` CLI --------------------------------------------------------------






def test_from_bundle_import_is_torch_free() -> None:
    # RM-P1-PROSPECT-13: the reconstruction path must import with only numpy + Core, so a
    # lightweight consumer (Sim's resolver) rebuilds the field without prospect's inference stack.
    # A fresh interpreter is required — other tests in this process have already loaded torch.
    probe = (
        "import sys\n"
        "from astro_mine.prospect.publish import from_bundle\n"
        "heavy = [m for m in ('torch', 'gpytorch') if m in sys.modules]\n"
        "assert not heavy, f'from_bundle pulled the inference stack: {heavy}'\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
