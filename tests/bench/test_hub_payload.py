"""Verified payload retrieval — Bench's one door onto an artifact's bytes (bench#44).

Bench consumes no payload bytes itself; a deployment-injected loader does (an ONNX policy, a
payload-layer metric). :func:`pull_verified_layer` is the only route Bench's Hub seam offers it, and
the property that matters is not that it *returns* bytes but that it *refuses* the wrong ones: every
layer is re-hashed against the digest the verified manifest commits to before a byte comes back
(hub.md §2.3; conventions.md §9).

The two refusals are tested separately because they fire at different depths:

- a **tampered blob on disk** is caught by Hub's artifact-integrity gate, inside the supply-chain
  check that runs before any layer is fetched (:class:`SupplyChainError`);
- a **registry that serves bytes other than the ones it stores** passes that gate — nothing on disk
  is wrong — and is caught only by the re-hash at the fetch itself (:class:`IntegrityError`). This
  is the choke point the raw-blob route did not have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from astro_mine.bench.leaderboard import (
    PayloadRetrievalError,
    ResolvedSubmission,
    pull_verified_layer,
    resolve_submission,
)
from astro_mine.core.registry.enums import PluginKind
from astro_mine.core.registry.model import PluginManifest
from astro_mine.hub.registry import Blob, IntegrityError, Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import SupplyChainError, attest, generate_keypair

ONNX_MEDIA_TYPE = "application/vnd.astro-mine.policy.onnx.v1"
TOKENIZER_MEDIA_TYPE = "application/vnd.astro-mine.policy.tokenizer.v1"
ONNX_BYTES = b"onnx-model-bytes"
TOKENIZER_BYTES = b"tokenizer-bytes"


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "hub-registry")


def _publish(
    registry: Registry,
    *,
    name: str = "acme/policy",
    version: str = "1.0.0",
    layers: list[Blob] | None = None,
) -> str:
    """Publish an **attested** policy artifact and return its image-manifest digest.

    Attested because verified retrieval runs the same supply-chain gate the leaderboard applies at
    admission (bench.md §9): an artifact carrying no cosign signature / SLSA provenance / SBOM never
    reaches the layer-fetch path at all.
    """
    manifest = PluginManifest(
        name=name,
        version=version,
        kind=PluginKind.POLICY,
        core_interfaces={"observation": "0.1.0"},
        inputs=["Observation"],
        outputs=["ActionBatch"],
        attributes={"entrypoint": "tests.bench._factories:BASELINE_INSTANCE"},
    )
    published = registry.publish(
        name=name,
        version=version,
        kind=PluginKind.POLICY.value,
        # The bare manifest, which is what every publisher in this platform stores (hub.md §2
        # principle 2). This helper wrote a ManifestDocument envelope, and that is why the suite
        # was green over a path no real artifact could take (astro-mine-platform#14): the fixture
        # and the intake agreed with each other and with nothing else.
        config=manifest.model_dump(mode="json"),
        layers=[Blob(ONNX_MEDIA_TYPE, ONNX_BYTES)] if layers is None else layers,
    )
    private_pem, _ = generate_keypair()
    attest(registry, published.digest, private_key_pem=private_pem, name=name, version=version)
    return published.digest


def _resolved(registry: Registry, digest: str) -> ResolvedSubmission:
    """The real intake type — a frozen dataclass, structurally a ``ResolvedArtifact``."""
    return resolve_submission(registry, digest)


def test_pull_returns_the_published_bytes(registry: Registry) -> None:
    resolved = _resolved(registry, _publish(registry))
    assert pull_verified_layer(registry, resolved) == ONNX_BYTES


def test_media_type_selects_the_layer(registry: Registry) -> None:
    digest = _publish(
        registry,
        layers=[Blob(ONNX_MEDIA_TYPE, ONNX_BYTES), Blob(TOKENIZER_MEDIA_TYPE, TOKENIZER_BYTES)],
    )
    resolved = _resolved(registry, digest)
    assert pull_verified_layer(registry, resolved, media_type=ONNX_MEDIA_TYPE) == ONNX_BYTES
    assert (
        pull_verified_layer(registry, resolved, media_type=TOKENIZER_MEDIA_TYPE) == TOKENIZER_BYTES
    )


def test_tampered_layer_blob_is_refused(registry: Registry) -> None:
    # Swap the stored layer for bytes that do not hash to their content address. Hub's
    # artifact-integrity gate fires inside the supply-chain check, before the fetch — fail-closed,
    # and the error propagates unwrapped so the alarm is not blunted into a Bench-shaped error.
    digest = _publish(registry)
    resolved = _resolved(registry, digest)
    layer_digest = registry.read_manifest(digest)["layers"][0]["digest"]
    blob_path(registry.path, layer_digest).write_bytes(b"tampered-onnx-bytes")

    with pytest.raises(SupplyChainError, match="integrity"):
        pull_verified_layer(registry, resolved)


def test_a_registry_that_serves_other_bytes_is_refused(registry: Registry) -> None:
    """The re-hash choke point: bytes that do not hash to the manifest's digest never come back.

    Every blob on disk is intact here, so the supply-chain gate passes and cannot mask the failure —
    the registry simply *hands back* something else, which is precisely what a raw blob read cannot
    detect and what verified retrieval refuses (hub.md §2.3).
    """
    digest = _publish(registry)
    resolved = _resolved(registry, digest)
    layer_digest = registry.read_manifest(digest)["layers"][0]["digest"]

    class _SwappingRegistry:
        """Stores the published bytes, serves someone else's for the payload layer."""

        def __init__(self, inner: Registry) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def pull_blob(self, requested: str) -> bytes:
            if requested == layer_digest:
                return b"substituted-onnx-bytes"
            return bytes(self._inner.pull_blob(requested))

    with pytest.raises(IntegrityError, match="content-address mismatch"):
        pull_verified_layer(_SwappingRegistry(registry), resolved)


def test_artifact_without_a_payload_layer_is_an_error(registry: Registry) -> None:
    resolved = _resolved(registry, _publish(registry, layers=[]))
    with pytest.raises(PayloadRetrievalError, match="no payload layer"):
        pull_verified_layer(registry, resolved)


def test_unmatched_media_type_is_an_error(registry: Registry) -> None:
    resolved = _resolved(registry, _publish(registry))
    with pytest.raises(PayloadRetrievalError, match="of media type"):
        pull_verified_layer(registry, resolved, media_type=TOKENIZER_MEDIA_TYPE)


def test_ambiguous_multi_layer_pull_is_an_error(registry: Registry) -> None:
    # Two layers and no media type named: guessing which one the loader meant is not a service Bench
    # should provide — the caller names it.
    digest = _publish(
        registry,
        layers=[Blob(ONNX_MEDIA_TYPE, ONNX_BYTES), Blob(TOKENIZER_MEDIA_TYPE, TOKENIZER_BYTES)],
    )
    with pytest.raises(PayloadRetrievalError, match="name the media_type"):
        pull_verified_layer(registry, _resolved(registry, digest))
