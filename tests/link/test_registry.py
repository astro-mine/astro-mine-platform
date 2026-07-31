"""RM-P0-LINK-04 — Link as a Core ``comms_model`` plugin, published to Hub by content hash.

Proves the deliverable and its acceptance (link.md §3, §6; hub.md §3, §9):

- Link declares a Core :class:`~astro_mine.core.registry.PluginManifest` with
  ``kind=comms_model``, which passes the full Core registry load-time gate (negotiation), i.e. is
  registrable exactly like the Worlds/Prospect plugin manifests;
- :func:`~astro_mine.link.registry.publish_contact_plan` stores a ContactPlan to a **local
  OCI-layout** ``Registry`` as a signed artifact whose config is that manifest
  (``provenance.digest == plan_digest``); a ``HubClient`` verifies + pulls it, **failing closed** on
  a tampered blob;
- a consumer resolves the comms model **by content hash** and rebuilds a live
  ``ConnectivitySampler`` through the ``astro_mine.providers`` entry point — **without importing**
  ``astro_mine.link`` in the resolve path;
- the bundle is byte-stable, so the same plan yields the same artifact digest ("two clean checkouts
  resolve the identical comms model") — the property the Bench anchor pin depends on.

Fully offline — a local registry directory under ``tmp_path``; no hosted Hub / Cloud.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from astro_mine.core.messages import ContactInterval, ContactNode, ContactPlan
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.registry import PluginKind, PluginManifest, PluginRegistry
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import (
    ArtifactExistsError,
    Registry,
    artifact_media_type,
    open_registry,
)
from astro_mine.hub.supply_chain import SupplyChainError, generate_keypair
from astro_mine.link.cache import plan_digest
from astro_mine.link.products import ConnectivitySampler
from astro_mine.link.registry import (
    BUNDLE_MEDIA_TYPE,
    COMMS_MODEL_ARTIFACT_KIND,
    LinkRegistryError,
    build_comms_model_manifest,
    bundle_digest,
    from_bundle,
    plan_from_bundle,
    publish_contact_plan,
    scenario_from_bundle,
    serialize_bundle,
)

_NAME = "astro-mine.link.test-comms-model"
_VERSION = "0.1.0"
_SCENARIO = "unit-test-scenario"


def _plan() -> ContactPlan:
    nodes = [
        ContactNode(id="rover", role=NodeRole.SPACE, kind="surface_agent"),
        ContactNode(id="relay", role=NodeRole.SPACE, kind="relay_orbiter"),
        ContactNode(id="dss", role=NodeRole.GROUND, kind="ground_station"),
    ]
    intervals = [
        ContactInterval(
            node_a="rover",
            node_b="relay",
            start_tdb_s=0.0,
            end_tdb_s=600.0,
            max_rate_bps=2.0e6,
            min_latency_s=0.01,
            mean_latency_s=0.01,
            margin_db=6.0,
            modcod="qpsk_r1_2",
        ),
        ContactInterval(
            node_a="relay", node_b="dss", start_tdb_s=300.0, end_tdb_s=1800.0, max_rate_bps=1.0e7
        ),
    ]
    return ContactPlan(
        nodes=nodes, intervals=intervals, epoch_start_tdb_s=0.0, epoch_end_tdb_s=3600.0
    )


def _input_hashes() -> dict[str, str]:
    return {"kernels": "sha256:aa", "terrain": "sha256:bb", "config": "sha256:cc"}


def _publish(registry_path: Path, key: bytes) -> Any:
    return publish_contact_plan(
        _plan(),
        registry=open_registry(str(registry_path)),
        name=_NAME,
        version=_VERSION,
        scenario_id=_SCENARIO,
        input_hashes=_input_hashes(),
        private_key_pem=key,
    )


def _layers_by_media_type(registry: Registry, digest: str) -> dict[str, bytes]:
    """Reconstruct the ``mediaType -> bytes`` layer map a consumer feeds ``from_bundle``."""
    image = registry.read_manifest(digest)
    return {layer["mediaType"]: registry.pull_blob(layer["digest"]) for layer in image["layers"]}


def _entry_point_factory() -> Callable[[PluginManifest, Mapping[str, bytes]], Any]:
    """Load the comms_model factory the way a consumer does — via the entry point only."""
    return importlib.metadata.entry_points(group="astro_mine.providers")["comms_model"].load()


# --- the Core comms_model manifest -----------------------------------------------------------


def test_manifest_declares_comms_model() -> None:
    plan = _plan()
    bundle = serialize_bundle(plan, scenario={"scenario_id": _SCENARIO})
    manifest = build_comms_model_manifest(
        plan,
        name=_NAME,
        version=_VERSION,
        bundle_sha256=bundle_digest(bundle),
        scenario_id=_SCENARIO,
        input_hashes=_input_hashes(),
    )

    assert manifest.kind == PluginKind.COMMS_MODEL
    assert manifest.core_interfaces == {"messages": "0.1.0", "env": "0.1.0"}
    assert manifest.license == "Apache-2.0"
    assert "CommsObservationMask" in manifest.outputs
    assert manifest.provenance is not None
    assert manifest.provenance.digest == f"sha256:{plan_digest(plan)}"
    assert manifest.provenance.source_content_hashes == _input_hashes()
    assert manifest.attributes["bundle_media_type"] == BUNDLE_MEDIA_TYPE
    assert manifest.attributes["space_nodes"] == ["rover", "relay"]
    assert manifest.attributes["ground_nodes"] == ["dss"]
    # A published comms model is open commons — it declares no gated capability tag (RFC-0003).
    assert manifest.capability_tags == []
    # The manifest passes the full Core load-time gate (negotiation), i.e. is registrable.
    PluginRegistry(require_signature=False).register(manifest)


# --- the content-addressed bundle -------------------------------------------------------------


def test_bundle_round_trips_the_plan() -> None:
    plan = _plan()
    bundle = serialize_bundle(plan, scenario={"scenario_id": _SCENARIO, "nodes": ["rover"]})

    assert plan_from_bundle(bundle) == plan
    assert scenario_from_bundle(bundle)["scenario_id"] == _SCENARIO
    # The bundle's content address is the OCI layer digest form (sha256:<hex>).
    assert bundle_digest(bundle).startswith("sha256:")


def test_bundle_bytes_are_deterministic() -> None:
    scenario = {"scenario_id": _SCENARIO}
    assert serialize_bundle(_plan(), scenario=scenario) == serialize_bundle(
        _plan(), scenario=scenario
    )


def test_plan_from_bundle_rejects_a_non_bundle() -> None:
    with pytest.raises(LinkRegistryError, match="readable tar"):
        plan_from_bundle(b"not a tar")


# --- publish -> resolve-by-digest -> rebuild ---------------------------------------------------


def test_publish_signs_and_verifies(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair()
    artifact = _publish(tmp_path / "registry", private_pem)
    registry = Registry(tmp_path / "registry")

    assert artifact.reference == f"{_NAME}:{_VERSION}"
    assert registry.read_manifest(artifact.digest)["artifactType"] == artifact_media_type(
        COMMS_MODEL_ARTIFACT_KIND
    )

    # A client with the trusted key verifies the signature + SLSA + SBOM, fail-closed.
    client = HubClient(registry, trusted_public_key_pem=public_pem)
    manifest = PluginManifest.model_validate_json(client.pull(artifact.digest))
    assert manifest.kind == PluginKind.COMMS_MODEL
    assert manifest.provenance is not None
    assert manifest.provenance.digest == f"sha256:{plan_digest(_plan())}"


def test_resolve_by_digest_rebuilds_a_live_sampler(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair()
    artifact = _publish(tmp_path / "registry", private_pem)
    registry = Registry(tmp_path / "registry")

    # Consumer path: pull + verify by content hash, fetch the layers, rebuild through the entry
    # point — no astro_mine.link import here (this is exactly what Sim's ContentResolver does).
    client = HubClient(registry, trusted_public_key_pem=public_pem)
    manifest = PluginManifest.model_validate_json(client.pull(artifact.digest))
    layers = _layers_by_media_type(registry, artifact.digest)
    assert set(layers) == {BUNDLE_MEDIA_TYPE}

    sampler = _entry_point_factory()(manifest, layers)
    assert isinstance(sampler, ConnectivitySampler)
    assert set(sampler.nodes) == {"rover", "relay", "dss"}


def test_artifact_is_immutable(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    _publish(tmp_path / "registry", private_pem)
    # name:version is immutable — a re-publish is refused (hub.md §2.1).
    with pytest.raises(ArtifactExistsError):
        _publish(tmp_path / "registry", private_pem)


# --- determinism: the property the Bench anchor pin rests on ----------------------------------


def test_two_clean_publishes_resolve_the_identical_digest(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    one = _publish(tmp_path / "a", private_pem)
    two = _publish(tmp_path / "b", private_pem)
    assert one.digest == two.digest


def test_signing_does_not_perturb_the_digest(tmp_path: Path) -> None:
    # Signatures ride as OCI referrers, not inside the image manifest, so publishes of the same
    # plan share a digest regardless of *who* signed it — a Bench pin is stable across publishers.
    # (This used to compare a signed publish against an unsigned one; Hub admits no unsigned
    # content now, so two distinct keys make the same point.)
    first_pem, _ = generate_keypair()
    second_pem, _ = generate_keypair()
    assert first_pem != second_pem
    assert _publish(tmp_path / "a", first_pem).digest == _publish(tmp_path / "b", second_pem).digest


# --- fail-closed -------------------------------------------------------------------------------


def test_pull_fails_closed_on_tampered_payload(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair()
    artifact = _publish(tmp_path / "registry", private_pem)
    registry = Registry(tmp_path / "registry")

    layer_digest = registry.read_manifest(artifact.digest)["layers"][0]["digest"]
    tampered = registry.path / "blobs" / "sha256" / layer_digest.split(":", 1)[1]
    tampered.write_bytes(tampered.read_bytes() + b"\x00tamper")

    # pull() re-runs the supply-chain check before returning bytes; the integrity failure surfaces
    # as SupplyChainError — a compromised registry cannot serve a tampered comms model.
    client = HubClient(registry, trusted_public_key_pem=public_pem)
    with pytest.raises(SupplyChainError):
        client.pull(artifact.digest)


def test_from_bundle_rejects_a_non_comms_manifest() -> None:
    manifest = PluginManifest(
        name="not-a-comms-model",
        version=_VERSION,
        kind=PluginKind.POLICY,
        core_interfaces={"policy": "0.1.0"},
    )
    with pytest.raises(LinkRegistryError, match="not a"):
        from_bundle(manifest, {BUNDLE_MEDIA_TYPE: b""})


def test_from_bundle_rejects_a_missing_layer() -> None:
    manifest = build_comms_model_manifest(
        _plan(), name=_NAME, version=_VERSION, bundle_sha256="sha256:00", scenario_id=_SCENARIO
    )
    with pytest.raises(LinkRegistryError, match=r"no .* layer"):
        from_bundle(manifest, {})
