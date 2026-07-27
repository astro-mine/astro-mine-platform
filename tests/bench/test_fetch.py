"""Fetching a scenario's pinned content into a local store (G1.2; bench#56).

The property under test is not that ``fetch`` *moves bytes* but that it **refuses the wrong ones**,
and that what it leaves behind stands on its own:

- a mirrored artifact must reproduce its **pinned digest** — the mirror re-publishes from the
  artifact's own bytes, and if that ever stopped rebuilding an identical manifest every downstream
  reference to the pin would break silently, so it fails loudly instead (:class:`DigestMismatch`);
- the artifact's attestations must travel with it, so the fetched store **re-verifies offline**;
- a tampered source must fail closed, not warn;
- a second fetch must succeed with the source gone (CX-LOCAL) and must not trust bytes merely for
  being on disk.

Everything runs local-to-local against registries built in-fixture — no network, no GHCR, no
account (CX-LOCAL). The source registry stands in for ``ghcr.io/astro-mine``; ``open_registry``
dispatches on the string either way, so the transport under test is the same one production uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from astro_mine.bench.content import (
    DigestMismatch,
    FetchError,
    default_store_path,
    fetch_scenario_content,
    resolve_store_path,
)
from astro_mine.bench.content._fetch import STORE_ENV
from astro_mine.bench.scenario import ContentPins, ContentRef, ScenarioSpec
from astro_mine.core.registry.enums import PluginKind
from astro_mine.core.registry.model import ManifestDocument, PluginManifest
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import ArtifactNotFound, Blob, Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import SupplyChainError, attest, generate_keypair

from ._factories import make_scenario_spec

WORLD_MEDIA_TYPE = "application/vnd.astro-mine.world.bundle.v1"
WORLD_BYTES = b"world-bundle-bytes"


def _publish_asset(
    registry: Registry,
    *,
    name: str,
    version: str = "1.0.0",
    kind: PluginKind = PluginKind.ASSET,
    artifact_kind: str = "asset",
    private_pem: bytes | None = None,
    payload: bytes = WORLD_BYTES,
) -> str:
    """Publish an **attested** artifact into ``registry`` and return its manifest digest.

    Attested because a fetch runs the full supply-chain gate: an artifact with no signature, SLSA
    provenance or SBOM never counts as fetched (hub.md §9).
    """
    manifest = PluginManifest(
        name=name,
        version=version,
        kind=kind,
        core_interfaces={"env": "0.1.0"},
        inputs=["Observation"],
        outputs=["ActionBatch"],
    )
    published = registry.publish(
        name=name,
        version=version,
        kind=artifact_kind,
        config=ManifestDocument(manifest_version="0.1", manifest=manifest).model_dump(mode="json"),
        layers=[Blob(WORLD_MEDIA_TYPE, payload)],
    )
    key = private_pem if private_pem is not None else generate_keypair()[0]
    attest(registry, published.digest, private_key_pem=key, name=name, version=version)
    return published.digest


@pytest.fixture
def source(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "source")


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "store"


def _spec_pinning(world: tuple[str, str], *fleet: tuple[str, str]) -> ScenarioSpec:
    """A ScenarioSpec pinning ``world`` and at least one fleet asset, as ``(id, digest)`` pairs.

    Both are required because ``ContentPins`` requires them — a scenario has a place *and* robots —
    so there is no such thing as a one-pin spec to test against.
    """
    return make_scenario_spec(
        content=ContentPins(
            world=ContentRef(id=world[0], content_hash=world[1]),
            fleet=tuple(ContentRef(id=i, content_hash=d) for i, d in fleet),
        )
    )


@pytest.fixture
def anchor(source: Registry) -> tuple[ScenarioSpec, str, str]:
    """A two-pin scenario — a world and a fleet asset — published and attested in ``source``."""
    world = _publish_asset(source, name="shackleton-v1", artifact_kind="world")
    rover = _publish_asset(source, name="astro-mine.fleet.rover", artifact_kind="asset")
    spec = _spec_pinning(("shackleton-v1", world), ("astro-mine.fleet.rover", rover))
    return spec, world, rover


# --- the happy path ---------------------------------------------------------------------


def test_mirrors_every_pin_and_preserves_the_digest(
    source: Registry, store: Path, anchor: tuple[ScenarioSpec, str, str]
) -> None:
    """The whole design rests on this: re-publishing an artifact's own bytes reproduces its digest.

    If this ever fails, ``fetch`` cannot be built on Hub's public API and needs a Hub-side copy().
    """
    spec, world, rover = anchor

    pins = fetch_scenario_content(spec, source=source.path, store=store)

    assert {pin.digest for pin in pins} == {world, rover}
    assert all(pin.mirrored for pin in pins)
    mirrored = Registry(store)
    assert mirrored.read_manifest(world) == source.read_manifest(world)


def test_fetched_store_verifies_offline(
    source: Registry, store: Path, anchor: tuple[ScenarioSpec, str, str]
) -> None:
    """Attestations travel with the artifact, so the copy stands on its own with no network."""
    spec, digest, _ = anchor
    fetch_scenario_content(spec, source=source.path, store=store)

    mirrored = Registry(store)
    assert len(mirrored.referrers(digest)) == len(source.referrers(digest)) == 3
    HubClient(mirrored).verify(digest)  # raises if any evidence is missing or broken


def test_reports_sizes_and_references(
    source: Registry, store: Path, anchor: tuple[ScenarioSpec, str, str]
) -> None:
    spec, world, _ = anchor
    pins = fetch_scenario_content(spec, source=source.path, store=store)
    world_pin = next(pin for pin in pins if pin.digest == world)
    assert world_pin.reference == "shackleton-v1:1.0.0"
    assert world_pin.size_bytes >= len(WORLD_BYTES)


def test_kind_is_recovered_from_the_artifact_type_not_the_core_kind(
    source: Registry, store: Path
) -> None:
    """A Worlds bundle is Hub kind ``world`` but Core kind ``world_provider`` (astro-mine-hub#33).

    Publishing under the Core kind would emit a different ``artifactType``, change the manifest and
    break the digest — so this is the regression guarding that distinction.
    """
    digest = _publish_asset(
        source, name="shackleton-v1", kind=PluginKind.WORLD_PROVIDER, artifact_kind="world"
    )
    rover = _publish_asset(source, name="astro-mine.fleet.rover", artifact_kind="asset")
    config = json.loads(source.read_config(digest))
    assert config["manifest"]["kind"] == "world_provider"  # Core vocabulary
    assert source.read_manifest(digest)["artifactType"].endswith("world.v1")  # Hub vocabulary

    pins = fetch_scenario_content(
        _spec_pinning(("shackleton-v1", digest), ("astro-mine.fleet.rover", rover)),
        source=source.path,
        store=store,
    )
    assert {pin.digest for pin in pins} == {digest, rover}


# --- idempotency and the offline path ---------------------------------------------------


def test_second_fetch_is_idempotent_and_needs_no_source(
    source: Registry, store: Path, anchor: tuple[ScenarioSpec, str, str]
) -> None:
    """A populated store re-verifies without touching the source at all (CX-LOCAL)."""
    spec, world, rover = anchor
    fetch_scenario_content(spec, source=source.path, store=store)

    # Point the source at a path that does not exist: a second fetch must not reach for it.
    pins = fetch_scenario_content(spec, source=store.parent / "gone", store=store)
    assert [pin.mirrored for pin in pins] == [False, False]
    assert {pin.digest for pin in pins} == {world, rover}


def test_a_present_but_unverifiable_pin_is_refetched_not_trusted(
    source: Registry, store: Path, anchor: tuple[ScenarioSpec, str, str]
) -> None:
    """Presence on disk is not trust: a store whose blob was tampered with must not be accepted."""
    spec, digest, _ = anchor
    fetch_scenario_content(spec, source=source.path, store=store)

    layer = Registry(store).read_manifest(digest)["layers"][0]["digest"]
    blob_path(store, layer).write_bytes(b"tampered")

    # The tampered store cannot be re-used, and re-publishing over it is refused rather than
    # silently preferring either copy.
    with pytest.raises(FetchError):
        fetch_scenario_content(spec, source=source.path, store=store)


# --- the refusals -----------------------------------------------------------------------


def test_a_pin_the_source_does_not_have_fails_closed(source: Registry, store: Path) -> None:
    rover = _publish_asset(source, name="astro-mine.fleet.rover", artifact_kind="asset")
    absent = "sha256:" + ("e" * 64)
    with pytest.raises(FetchError):
        fetch_scenario_content(
            _spec_pinning(("ghost", absent), ("astro-mine.fleet.rover", rover)),
            source=source.path,
            store=store,
        )
    assert not (store / "index.json").exists() or not Registry(store).references()


def test_a_tampered_source_blob_fails_closed(
    source: Registry, store: Path, anchor: tuple[ScenarioSpec, str, str]
) -> None:
    """A source that serves bytes other than the ones it committed to must never be mirrored.

    **Which gate catches it depends on the transport, and only one of them is the transport's.**
    A *remote* source re-hashes every blob on the way out (``_remote._checked``), so tampering dies
    there as an ``IntegrityError``. A *local* OCI layout does **not** — ``Registry.pull_blob`` reads
    the file without re-deriving its hash — so tampered bytes reach the mirror unchallenged and the
    only thing standing between them and the store is this module's own digest re-derivation. That
    is exactly why the reproduced digest is asserted rather than assumed: it is a real safety net,
    not a belt-and-braces check, and it is load-bearing for local and file-backed sources.
    """
    spec, digest, _ = anchor
    layer = source.read_manifest(digest)["layers"][0]["digest"]
    blob_path(source.path, layer).write_bytes(b"tampered-in-flight")

    with pytest.raises(DigestMismatch):
        fetch_scenario_content(spec, source=source.path, store=store)

    # Nothing tampered became resolvable: the pinned digest is not in the store.
    with pytest.raises((SupplyChainError, ArtifactNotFound)):
        HubClient(Registry(store)).verify(digest)


def test_digest_mismatch_is_its_own_error(
    source: Registry,
    store: Path,
    monkeypatch: pytest.MonkeyPatch,
    anchor: tuple[ScenarioSpec, str, str],
) -> None:
    """If the mirror ever stopped reproducing digests, it must be unmistakable in the traceback."""
    spec, digest, _ = anchor

    real_publish = Registry.publish

    def drifting(self: Registry, **kwargs: Any) -> Any:
        published = real_publish(self, **{**kwargs, "annotations": {"drift": "1"}})
        return published

    monkeypatch.setattr(Registry, "publish", drifting)
    with pytest.raises(DigestMismatch) as excinfo:
        fetch_scenario_content(spec, source=source.path, store=store)
    assert digest in str(excinfo.value)


def test_signer_pinning_rejects_a_foreign_key(source: Registry, store: Path) -> None:
    """``--trusted-key`` is optional, but when supplied a mismatch is a hard error (bench#56 D6)."""
    signer_pem, _ = generate_keypair()
    world = _publish_asset(
        source, name="shackleton-v1", artifact_kind="world", private_pem=signer_pem
    )
    rover = _publish_asset(
        source, name="astro-mine.fleet.rover", artifact_kind="asset", private_pem=signer_pem
    )
    _, foreign_pub = generate_keypair()

    with pytest.raises(FetchError):
        fetch_scenario_content(
            _spec_pinning(("shackleton-v1", world), ("astro-mine.fleet.rover", rover)),
            source=source.path,
            store=store,
            trusted_key_pem=foreign_pub,
        )


def test_signer_pinning_accepts_the_matching_key(source: Registry, store: Path) -> None:
    signer_pem, signer_pub = generate_keypair()
    world = _publish_asset(
        source, name="shackleton-v1", artifact_kind="world", private_pem=signer_pem
    )
    rover = _publish_asset(
        source, name="astro-mine.fleet.rover", artifact_kind="asset", private_pem=signer_pem
    )
    pins = fetch_scenario_content(
        _spec_pinning(("shackleton-v1", world), ("astro-mine.fleet.rover", rover)),
        source=source.path,
        store=store,
        trusted_key_pem=signer_pub,
    )
    assert {pin.digest for pin in pins} == {world, rover}


# --- store-path resolution --------------------------------------------------------------


def test_store_path_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--registry`` > ``$ASTRO_MINE_HUB_REGISTRY`` > the XDG default (bench#56 D5)."""
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "from-env"))
    assert resolve_store_path(tmp_path / "explicit") == tmp_path / "explicit"
    assert resolve_store_path() == tmp_path / "from-env"

    monkeypatch.delenv(STORE_ENV)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert resolve_store_path() == tmp_path / "cache" / "astro-mine" / "hub-registry"


def test_default_store_is_never_the_workspace_convention(monkeypatch: pytest.MonkeyPatch) -> None:
    """``files/hub-registry`` is a dev-workspace path, not a product default (bench#56 D5)."""
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert "files/hub-registry" not in str(default_store_path())
    assert default_store_path().is_absolute()
