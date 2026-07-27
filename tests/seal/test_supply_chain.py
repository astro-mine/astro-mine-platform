"""Verify-twice tests (RM-P1-SEAL-03): required-evidence policy + the fail-closed orchestrator.

Two halves:

- the **pure document predicates** (``verify_slsa_document`` / ``verify_sbom_document``) and the
  ``DEFAULT_REQUIRED`` policy;
- the **registry-agnostic orchestrator** ``verify(store, subject, ...)``, driven over the
  ``MemoryStore`` fixture — a dict that has never heard of OCI or Hub (``conftest.py``).

The negative half is the point. ``verify`` returns ``None`` *only* when every required piece of
evidence is present, intact, and valid; every other path — a tampered artifact, a tampered
attestation, a missing or bad signature, an untrusted key, missing provenance, a missing SBOM, a
garbage document, an unknown policy token, a failing store — MUST raise ``SupplyChainError``
(seal.md §9; conventions.md §9). Each is asserted below.
"""

from __future__ import annotations

import pytest

from astro_mine.core.hashing import canonical_json, content_hash
from astro_mine.seal import (
    ARTIFACT_TYPE_SBOM,
    ARTIFACT_TYPE_SIGNATURE,
    ARTIFACT_TYPE_SLSA,
    DEFAULT_REQUIRED,
    MEDIA_SBOM,
    MEDIA_SIGNATURE,
    MEDIA_SLSA,
    AttestationError,
    AttestationStore,
    SupplyChainError,
    attest,
    build_cyclonedx_sbom,
    build_slsa_provenance,
    generate_keypair,
    sign_digest,
    verify,
    verify_sbom_document,
    verify_slsa_document,
)
from tests.seal.conftest import MemoryStore

ARTIFACT = b"a signed astro-mine artifact"


def _publish(store: MemoryStore, private_pem: bytes) -> str:
    """Store the artifact and attest it — the publish side. Returns the subject digest."""
    subject = store.put_artifact(ARTIFACT)
    attest(
        store,
        subject,
        private_key_pem=private_pem,
        name="astro-mine.fleet.rover",
        version="0.1.0",
        builder_id="astro-mine-seal-test",
    )
    return subject


# -- the port is structural ----------------------------------------------------------------------


def test_memory_store_satisfies_the_port(store: MemoryStore) -> None:
    """A plain dict-backed store *is* an ``AttestationStore`` — no registry type in the seam."""
    assert isinstance(store, AttestationStore)


# -- policy + pure document predicates -----------------------------------------------------------


def test_default_required_policy() -> None:
    # The evidence a verified/curated artifact must carry (LUNAR-SR-002).
    assert DEFAULT_REQUIRED == ("signature", "slsa", "sbom")


def test_verify_slsa_document_accepts_well_shaped() -> None:
    doc = build_slsa_provenance(
        subject_name="a:1", subject_digest="sha256:" + "ab" * 32, builder_id="x"
    )
    verify_slsa_document(doc)  # no raise


def test_verify_slsa_document_fail_closed() -> None:
    with pytest.raises(AttestationError):
        verify_slsa_document({"predicateType": "https://example.com/wrong"})
    with pytest.raises(AttestationError):
        verify_slsa_document({})  # missing predicateType


def test_verify_sbom_document_accepts_cyclonedx() -> None:
    verify_sbom_document(build_cyclonedx_sbom(name="a", version="1"))  # no raise


def test_verify_sbom_document_fail_closed() -> None:
    with pytest.raises(AttestationError):
        verify_sbom_document({"bomFormat": "SPDX"})
    with pytest.raises(AttestationError):
        verify_sbom_document({})  # missing bomFormat


# -- the happy path (verify twice) ---------------------------------------------------------------


def test_verify_accepts_an_attested_artifact(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """attest → verify at admission → verify again at pull: the same check, green both times."""
    private_pem, public_pem = keypair
    subject = _publish(store, private_pem)

    verify(store, subject, trusted_public_key_pem=public_pem)  # admission
    verify(store, subject, trusted_public_key_pem=public_pem)  # pull — idempotent, no state


def test_verify_without_a_pinned_key_still_checks_integrity_and_binding(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """Omitting the trusted key is the documented opt-out: intact + bound, but not *trusted*."""
    private_pem, _ = keypair
    subject = _publish(store, private_pem)

    verify(store, subject)  # no raise — but this proves integrity, not signer trust


# -- fail closed: the artifact itself -------------------------------------------------------------


def test_verify_rejects_a_tampered_artifact(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """The bytes at the subject address no longer hash to it — a compromised registry."""
    private_pem, public_pem = keypair
    subject = _publish(store, private_pem)

    store.tamper(subject, b"malicious payload")

    with pytest.raises(SupplyChainError, match="integrity check failed"):
        verify(store, subject, trusted_public_key_pem=public_pem)


def test_verify_rejects_an_unknown_subject(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """An artifact that is not there is a refusal, not an empty pass."""
    _, public_pem = keypair

    with pytest.raises(SupplyChainError, match="integrity check failed"):
        verify(store, content_hash(b"never published"), trusted_public_key_pem=public_pem)


# -- fail closed: the signature -------------------------------------------------------------------


def test_verify_rejects_a_missing_signature(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """An artifact with no signature attached at all."""
    _, public_pem = keypair
    subject = store.put_artifact(ARTIFACT)  # published without attest()

    with pytest.raises(SupplyChainError, match="no cosign signature attached"):
        verify(store, subject, trusted_public_key_pem=public_pem)


def test_verify_rejects_an_untrusted_key(store: MemoryStore, keypair: tuple[bytes, bytes]) -> None:
    """A perfectly valid signature — by the *wrong signer*. The pinned key is what decides trust."""
    private_pem, _ = keypair
    attacker_public_pem = generate_keypair()[1]
    subject = _publish(store, private_pem)

    with pytest.raises(SupplyChainError, match="signature verification failed"):
        verify(store, subject, trusted_public_key_pem=attacker_public_pem)


def test_verify_rejects_a_tampered_signature_blob(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """The signature attestation's own bytes were swapped — the store raises, and we refuse."""
    private_pem, public_pem = keypair
    subject = _publish(store, private_pem)
    sig_digest = store.attestation_digests(subject, artifact_type=ARTIFACT_TYPE_SIGNATURE)[0]

    store.tamper(sig_digest, b'{"scheme": "sigstore_cosign"}')

    with pytest.raises(SupplyChainError, match="signature integrity failed"):
        verify(store, subject, trusted_public_key_pem=public_pem)


def test_verify_rejects_a_signature_over_another_artifact(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """A real signature by the trusted key — but bound to a *different* digest (lift-and-shift)."""
    private_pem, public_pem = keypair
    subject = store.put_artifact(ARTIFACT)
    other = sign_digest(content_hash(b"some other artifact"), private_pem)

    store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SIGNATURE,
        media_type=MEDIA_SIGNATURE,
        payload=other.model_dump_json().encode("utf-8"),
    )

    with pytest.raises(SupplyChainError, match="signature verification failed"):
        verify(store, subject, trusted_public_key_pem=public_pem, require=("signature",))


def test_verify_rejects_an_appended_bad_signature(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """*Every* attached signature must verify — a bad one cannot hide behind a good one."""
    private_pem, public_pem = keypair
    subject = _publish(store, private_pem)
    forged = sign_digest(subject, generate_keypair()[0])  # right digest, attacker's key

    store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SIGNATURE,
        media_type=MEDIA_SIGNATURE,
        payload=forged.model_dump_json().encode("utf-8"),
    )

    with pytest.raises(SupplyChainError, match="signature verification failed"):
        verify(store, subject, trusted_public_key_pem=public_pem)


def test_verify_rejects_a_malformed_signature_envelope(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """Garbage where a Core ``Signature`` should be — a parse failure is a refusal."""
    _, public_pem = keypair
    subject = store.put_artifact(ARTIFACT)

    store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SIGNATURE,
        media_type=MEDIA_SIGNATURE,
        payload=b"not a signature envelope",
    )

    with pytest.raises(SupplyChainError, match="signature verification failed"):
        verify(store, subject, trusted_public_key_pem=public_pem, require=("signature",))


# -- fail closed: the attestation documents -------------------------------------------------------


def test_verify_rejects_missing_provenance(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    private_pem, public_pem = keypair
    subject = store.put_artifact(ARTIFACT)
    signature = sign_digest(subject, private_pem)
    store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SIGNATURE,
        media_type=MEDIA_SIGNATURE,
        payload=signature.model_dump_json().encode("utf-8"),
    )

    with pytest.raises(SupplyChainError, match="no SLSA provenance attached"):
        verify(store, subject, trusted_public_key_pem=public_pem)


def test_verify_rejects_a_missing_sbom(store: MemoryStore, keypair: tuple[bytes, bytes]) -> None:
    private_pem, public_pem = keypair
    subject = _publish(store, private_pem)
    # Drop the SBOM the producer attached — an under-attested artifact.
    store.attachments[subject] = [
        (kind, digest) for kind, digest in store.attachments[subject] if kind != ARTIFACT_TYPE_SBOM
    ]

    with pytest.raises(SupplyChainError, match="no SBOM attached"):
        verify(store, subject, trusted_public_key_pem=public_pem)


def test_verify_rejects_a_tampered_provenance_blob(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    private_pem, public_pem = keypair
    subject = _publish(store, private_pem)
    slsa_digest = store.attestation_digests(subject, artifact_type=ARTIFACT_TYPE_SLSA)[0]

    store.tamper(slsa_digest, canonical_json({"predicateType": "https://evil.example/v1"}))

    with pytest.raises(SupplyChainError, match="SLSA provenance integrity failed"):
        verify(store, subject, trusted_public_key_pem=public_pem)


def test_verify_rejects_wrong_shaped_provenance(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """Provenance that is intact but not SLSA — attached honestly, still refused."""
    _, public_pem = keypair
    subject = store.put_artifact(ARTIFACT)
    store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SLSA,
        media_type=MEDIA_SLSA,
        payload=canonical_json({"predicateType": "https://example.com/not-slsa"}),
    )

    with pytest.raises(SupplyChainError, match="wrong predicateType"):
        verify(store, subject, trusted_public_key_pem=public_pem, require=("slsa",))


def test_verify_rejects_a_non_cyclonedx_sbom(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    _, public_pem = keypair
    subject = store.put_artifact(ARTIFACT)
    store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SBOM,
        media_type=MEDIA_SBOM,
        payload=canonical_json({"bomFormat": "SPDX"}),
    )

    with pytest.raises(SupplyChainError, match="not CycloneDX"):
        verify(store, subject, trusted_public_key_pem=public_pem, require=("sbom",))


def test_verify_rejects_an_unparseable_document(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    _, public_pem = keypair
    subject = store.put_artifact(ARTIFACT)
    store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SLSA,
        media_type=MEDIA_SLSA,
        payload=b"<xml>not json</xml>",
    )

    with pytest.raises(SupplyChainError, match="not valid JSON"):
        verify(store, subject, trusted_public_key_pem=public_pem, require=("slsa",))


def test_verify_rejects_a_non_object_document(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """Valid JSON, but a list — ``doc.get(...)`` would explode; refuse before it can."""
    _, public_pem = keypair
    subject = store.put_artifact(ARTIFACT)
    store.attach_attestation(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SBOM,
        media_type=MEDIA_SBOM,
        payload=b"[]",
    )

    with pytest.raises(SupplyChainError, match="not a JSON object"):
        verify(store, subject, trusted_public_key_pem=public_pem, require=("sbom",))


# -- fail closed: the policy and the store itself --------------------------------------------------


def test_verify_rejects_an_unknown_required_kind(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """A typo'd policy token is refused, never ignored — a silent typo is a silent bypass."""
    private_pem, public_pem = keypair
    subject = _publish(store, private_pem)

    with pytest.raises(SupplyChainError, match="unknown required-evidence kind"):
        verify(
            store,
            subject,
            trusted_public_key_pem=public_pem,
            require=("signature", "slsa", "sbom", "attestation"),
        )


def test_verify_rejects_when_the_store_fails(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """A store that errors is a refusal — never "the lookup failed, so nothing is wrong"."""
    private_pem, public_pem = keypair
    subject = _publish(store, private_pem)
    store.fail_with = RuntimeError("registry unreachable")

    with pytest.raises(SupplyChainError, match="integrity check failed"):
        verify(store, subject, trusted_public_key_pem=public_pem)


def test_verify_wraps_a_foreign_store_error_on_lookup(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """A store error type Seal cannot import still lands as ``SupplyChainError``, not a pass."""
    private_pem, public_pem = keypair
    subject = _publish(store, private_pem)

    class HostileStore(MemoryStore):
        def attestation_digests(self, subject: str, *, artifact_type: str) -> list[str]:
            raise RuntimeError("referrer lookup exploded")

    hostile = HostileStore()
    hostile.blobs = store.blobs
    hostile.attachments = store.attachments

    with pytest.raises(SupplyChainError, match="could not read signature attestations"):
        verify(hostile, subject, trusted_public_key_pem=public_pem)


def test_verify_with_no_required_evidence_still_checks_integrity(
    store: MemoryStore, keypair: tuple[bytes, bytes]
) -> None:
    """``require=()`` is an explicit caller opt-out — integrity is *still* enforced underneath."""
    private_pem, _ = keypair
    subject = _publish(store, private_pem)

    verify(store, subject, require=())  # no raise: the artifact is intact

    store.tamper(subject, b"swapped")
    with pytest.raises(SupplyChainError, match="integrity check failed"):
        verify(store, subject, require=())
