"""Attestation tests (RM-P1-SEAL-03): SLSA / SBOM shape, byte-stable goldens, and ``attest``.

The builders are a behavior-preserving port of Hub's ``supply_chain/_attest.py``. The frozen content
digests below pin **byte-stability** (deterministic, no clock) via the platform canonical form
(``astro_mine.core.hashing.content_hash_json``) — any change to the document layout turns CI red.

``attest`` is the publish-side orchestrator; it is exercised here over the registry-agnostic
``MemoryStore`` (``conftest.py``), which is a dict, not a registry. The consumer side (``verify``)
and its fail-closed negatives live in ``test_supply_chain.py``.
"""

from __future__ import annotations

import json

import pytest

from astro_mine.core.hashing import content_hash, content_hash_json
from astro_mine.core.registry import Signature, SignatureScheme
from astro_mine.seal import (
    ARTIFACT_TYPE_SBOM,
    ARTIFACT_TYPE_SIGNATURE,
    ARTIFACT_TYPE_SLSA,
    MEDIA_SBOM,
    MEDIA_SIGNATURE,
    MEDIA_SLSA,
    attest,
    build_cyclonedx_sbom,
    build_slsa_provenance,
    verify_signature,
)
from astro_mine.seal._attest import _digest_parts
from tests.seal.conftest import MemoryStore

# Frozen determinism goldens: content_hash_json(build_*(...)) over the fixed inputs below.
SLSA_GOLDEN = "sha256:fd7406747f403617116e86254adc73937fefd2faf40427545cdd1810b69c6848"
SBOM_GOLDEN = "sha256:9f489c933efeabe772fe1258d9b8219e0753945192f24f8f6f37f3b52e18e519"


def test_slsa_and_sbom_shape() -> None:
    slsa = build_slsa_provenance(
        subject_name="a:1",
        subject_digest=content_hash(b"artifact-bytes"),
        builder_id="x",
        inputs=[content_hash(b"i")],
    )
    assert slsa["predicateType"].endswith("provenance/v1")
    assert slsa["subject"][0]["digest"]["sha256"]
    assert slsa["predicate"]["buildDefinition"]["resolvedDependencies"]

    sbom = build_cyclonedx_sbom(
        name="a", version="1", components=[{"name": "numpy", "version": "2"}]
    )
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["components"][0]["name"] == "numpy"


def test_digest_parts_rejects_bad() -> None:
    with pytest.raises(ValueError):
        _digest_parts("md5:x")
    with pytest.raises(ValueError):
        _digest_parts("sha256:")


def test_build_slsa_provenance_is_byte_stable() -> None:
    slsa = build_slsa_provenance(
        subject_name="astro-mine-seal-demo:1.0.0",
        subject_digest="sha256:" + "ab" * 32,
        builder_id="astro-mine-seal-test",
        inputs=["sha256:" + "cd" * 32],
    )
    assert content_hash_json(slsa) == SLSA_GOLDEN


def test_build_cyclonedx_sbom_is_byte_stable() -> None:
    sbom = build_cyclonedx_sbom(
        name="astro-mine-seal-demo",
        version="1.0.0",
        components=[{"name": "cryptography", "version": "42.0"}],
    )
    assert content_hash_json(sbom) == SBOM_GOLDEN


# -- attest: the publish-side orchestrator, over a store that is not a registry --------------------


def test_attest_attaches_all_three_attestations(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """One call attaches the signature, the SLSA provenance, and the SBOM, each under its type."""
    private_pem, _ = keypair
    subject = store.put_artifact(b"payload")

    attested = attest(
        store,
        subject,
        private_key_pem=private_pem,
        name="astro-mine.fleet.rover",
        version="0.1.0",
        builder_id="astro-mine-seal-test",
        inputs=[content_hash(b"input")],
        components=[{"name": "numpy", "version": "2.0"}],
    )

    # The returned set is plain content digests — no registry descriptor type crosses the seam.
    assert (
        attested.signature
        == store.attestation_digests(subject, artifact_type=ARTIFACT_TYPE_SIGNATURE)[0]
    )
    assert attested.slsa == store.attestation_digests(subject, artifact_type=ARTIFACT_TYPE_SLSA)[0]
    assert attested.sbom == store.attestation_digests(subject, artifact_type=ARTIFACT_TYPE_SBOM)[0]

    # Each attestation carries its documented media type.
    assert store.media[attested.signature] == MEDIA_SIGNATURE
    assert store.media[attested.slsa] == MEDIA_SLSA
    assert store.media[attested.sbom] == MEDIA_SBOM


def test_attest_signs_the_subject_digest(store: MemoryStore, keypair: tuple[bytes, bytes]) -> None:
    """The attached signature is a cosign Core ``Signature`` that verifies over the subject."""
    private_pem, public_pem = keypair
    subject = store.put_artifact(b"payload")

    attested = attest(
        store,
        subject,
        private_key_pem=private_pem,
        name="a",
        version="1",
        builder_id="astro-mine-seal-test",
    )

    signature = Signature.model_validate_json(store.read_attestation(attested.signature))
    assert signature.scheme == SignatureScheme.SIGSTORE_COSIGN
    verify_signature(signature, subject, trusted_public_key_pem=public_pem)  # no raise


def test_attest_records_the_builder_and_inputs(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """``builder_id`` is a provenance claim: it must land in the stored SLSA document verbatim."""
    private_pem, _ = keypair
    subject = store.put_artifact(b"payload")
    build_input = content_hash(b"input")

    attested = attest(
        store,
        subject,
        private_key_pem=private_pem,
        name="rover",
        version="2.0",
        builder_id="astro-mine-bench",
        inputs=[build_input],
        components=[{"name": "numpy", "version": "2.0"}],
    )

    slsa = json.loads(store.read_attestation(attested.slsa))
    assert slsa["predicate"]["runDetails"]["builder"]["id"] == "astro-mine-bench"
    assert slsa["subject"][0]["name"] == "rover:2.0"
    assert slsa["subject"][0]["digest"]["sha256"] == subject.removeprefix("sha256:")
    assert slsa["predicate"]["buildDefinition"]["resolvedDependencies"] == [
        {"digest": {"sha256": build_input.removeprefix("sha256:")}}
    ]

    sbom = json.loads(store.read_attestation(attested.sbom))
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["components"] == [{"type": "library", "name": "numpy", "version": "2.0"}]


def test_attest_is_deterministic(store: MemoryStore, keypair: tuple[bytes, bytes]) -> None:
    """Same artifact + metadata ⇒ byte-identical documents (no clock), so the digests repeat."""
    private_pem, _ = keypair
    subject = store.put_artifact(b"payload")

    kwargs = {
        "private_key_pem": private_pem,
        "name": "rover",
        "version": "2.0",
        "builder_id": "astro-mine-seal-test",
    }
    first = attest(store, subject, **kwargs)  # type: ignore[arg-type]
    second = attest(store, subject, **kwargs)  # type: ignore[arg-type]

    # The SLSA/SBOM documents are content-addressed and clock-free: they land on one address.
    assert first.slsa == second.slsa
    assert first.sbom == second.sbom


def test_attest_propagates_a_store_failure(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """Attesting an artifact the store does not hold must raise, not attach orphan evidence."""
    private_pem, _ = keypair

    with pytest.raises(KeyError):
        attest(
            store,
            content_hash(b"never stored"),
            private_key_pem=private_pem,
            name="a",
            version="1",
            builder_id="astro-mine-seal-test",
        )
