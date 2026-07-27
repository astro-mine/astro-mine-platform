"""Supply-chain tests (RM-P1-HUB-03): sign/verify, SLSA/SBOM, verify-twice fail-closed negatives."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from astro_mine import seal
from astro_mine.core.registry import (
    PluginKind,
    PluginManifest,
    Provenance,
    SignatureScheme,
)
from astro_mine.hub._content import canonical_json, content_hash
from astro_mine.hub.registry import Blob, Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import (
    ARTIFACT_TYPE_SBOM,
    ARTIFACT_TYPE_SIGNATURE,
    ARTIFACT_TYPE_SLSA,
    AttestationSet,
    RegistryAttestationStore,
    SignatureError,
    SupplyChainError,
    attest,
    build_cyclonedx_sbom,
    build_slsa_provenance,
    generate_keypair,
    make_verifier,
    sign_digest,
    verify,
    verify_signature,
)
from astro_mine.seal import MEDIA_SBOM, MEDIA_SIGNATURE, MEDIA_SLSA
from astro_mine.seal._attest import _digest_parts

DIGEST = content_hash(b"artifact-bytes")


def _publish(reg: Registry) -> str:
    art = reg.publish(
        name="pol",
        version="1.0.0",
        kind="policy",
        config={"n": 1},
        layers=[Blob("application/octet-stream", b"payload")],
    )
    return art.digest


def _rsa_private_pem() -> bytes:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


# --- signing --------------------------------------------------------------------------------


def test_sign_and_verify_roundtrip() -> None:
    priv, pub = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    verify_signature(sig, DIGEST)  # untrusted (self-attested) — no raise
    verify_signature(sig, DIGEST, trusted_public_key_pem=pub)  # pinned — no raise


def test_verify_signature_wrong_digest() -> None:
    priv, _ = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    with pytest.raises(SignatureError):
        verify_signature(sig, content_hash(b"other"))


def test_verify_signature_untrusted_key() -> None:
    priv, _ = generate_keypair()
    _, other_pub = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    with pytest.raises(SignatureError):
        verify_signature(sig, DIGEST, trusted_public_key_pem=other_pub)


def test_verify_signature_bad_signature() -> None:
    priv, _ = generate_keypair()
    # a signature over a different digest, re-labelled as covering DIGEST → ECDSA verify fails
    forged = sign_digest(content_hash(b"different"), priv).model_copy(update={"payload": DIGEST})
    with pytest.raises(SignatureError):
        verify_signature(forged, DIGEST)


def test_verify_signature_incomplete_and_wrong_scheme() -> None:
    priv, _ = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    with pytest.raises(SignatureError):
        verify_signature(sig.model_copy(update={"value": None}), DIGEST)
    with pytest.raises(SignatureError):
        verify_signature(sig.model_copy(update={"scheme": SignatureScheme.UNSIGNED}), DIGEST)


def test_sign_with_non_ec_key_rejected() -> None:
    with pytest.raises(SignatureError):
        sign_digest(DIGEST, _rsa_private_pem())


def test_verify_with_non_ec_or_malformed_cert_rejected() -> None:
    priv, _ = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    with pytest.raises(SignatureError):
        verify_signature(sig.model_copy(update={"certificate": "not a pem"}), DIGEST)
    rsa_pub = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    with pytest.raises(SignatureError):
        verify_signature(sig.model_copy(update={"certificate": rsa_pub}), DIGEST)


# --- make_verifier (the pre-Core-load hook) -------------------------------------------------


def _manifest(digest: str | None, *, sig: object = None) -> PluginManifest:
    provenance = (
        Provenance(input_hashes=[], source_content_hashes={}, digest=digest) if digest else None
    )
    return PluginManifest(
        name="p", version="1.0.0", kind=PluginKind.POLICY, provenance=provenance, signature=sig
    )


def test_make_verifier_ok_and_failures() -> None:
    priv, pub = generate_keypair()
    sig = sign_digest(DIGEST, priv)
    make_verifier(trusted_public_key_pem=pub)(_manifest(DIGEST, sig=sig))  # ok

    with pytest.raises(SignatureError):  # no provenance digest to bind to
        make_verifier()(_manifest(None))
    with pytest.raises(SignatureError):  # provenance but no cosign signature
        make_verifier()(_manifest(DIGEST))
    _, other = generate_keypair()
    with pytest.raises(SignatureError):  # signed by an untrusted key
        make_verifier(trusted_public_key_pem=other)(_manifest(DIGEST, sig=sig))


# --- attestation documents ------------------------------------------------------------------


def test_slsa_and_sbom_shape() -> None:
    slsa = build_slsa_provenance(
        subject_name="a:1", subject_digest=DIGEST, builder_id="x", inputs=[content_hash(b"i")]
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


# --- attest + verify (verify-twice) ---------------------------------------------------------


def test_attest_and_verify_roundtrip(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    priv, pub = generate_keypair()
    aset = attest(
        reg,
        subject,
        private_key_pem=priv,
        name="pol",
        version="1.0.0",
        inputs=[content_hash(b"in")],
        components=[{"name": "numpy", "version": "2.0"}],
    )
    assert isinstance(aset, AttestationSet)
    verify(reg, subject, trusted_public_key_pem=pub)  # admission AND pull: no raise


def test_verify_missing_signature_fails_closed(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    with pytest.raises(SupplyChainError):  # nothing attached → no signature
        verify(reg, subject)


def test_verify_missing_slsa_and_sbom(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    priv, _ = generate_keypair()
    sig = sign_digest(subject, priv)
    reg.attach(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SIGNATURE,
        blob=Blob("application/json", sig.model_dump_json().encode()),
    )
    verify(reg, subject, require=("signature",))  # only signature required → ok
    with pytest.raises(SupplyChainError):  # slsa required but absent
        verify(reg, subject, require=("signature", "slsa"))
    with pytest.raises(SupplyChainError):  # sbom required but absent
        verify(reg, subject, require=("sbom",))


def test_verify_untrusted_key_fails_closed(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    priv, _ = generate_keypair()
    _, other = generate_keypair()
    attest(reg, subject, private_key_pem=priv, name="p", version="1")
    with pytest.raises(SupplyChainError):
        verify(reg, subject, trusted_public_key_pem=other)


def test_verify_tampered_artifact_fails_closed(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    priv, _ = generate_keypair()
    attest(reg, subject, private_key_pem=priv, name="p", version="1")
    layer = reg.read_manifest(subject)["layers"][0]["digest"]
    blob_path(reg.path, layer).write_bytes(b"HACKED")
    with pytest.raises(SupplyChainError):
        verify(reg, subject)


def test_verify_tampered_signature_blob_fails_closed(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    priv, _ = generate_keypair()
    aset = attest(reg, subject, private_key_pem=priv, name="p", version="1")
    sig_layer = reg.read_manifest(aset.signature)["layers"][0]["digest"]
    blob_path(reg.path, sig_layer).write_bytes(b'{"scheme":"unsigned"}')
    with pytest.raises(SupplyChainError):
        verify(reg, subject)


def test_verify_bad_slsa_predicate_fails_closed(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    reg.attach(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SLSA,
        blob=Blob(MEDIA_SLSA, canonical_json({"predicateType": "wrong"})),
    )
    with pytest.raises(SupplyChainError):
        verify(reg, subject, require=("slsa",))


def test_verify_bad_sbom_format_fails_closed(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    reg.attach(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SBOM,
        blob=Blob(MEDIA_SBOM, canonical_json({"bomFormat": "SPDX"})),
    )
    with pytest.raises(SupplyChainError):
        verify(reg, subject, require=("sbom",))


def test_verify_tampered_slsa_blob_fails_closed(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    priv, _ = generate_keypair()
    aset = attest(reg, subject, private_key_pem=priv, name="p", version="1")
    slsa_layer = reg.read_manifest(aset.slsa)["layers"][0]["digest"]
    blob_path(reg.path, slsa_layer).write_bytes(b"{}")  # corrupt the provenance blob
    with pytest.raises(SupplyChainError):
        verify(reg, subject, require=("slsa",))


def test_verify_tampered_sbom_blob_fails_closed(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    priv, _ = generate_keypair()
    aset = attest(reg, subject, private_key_pem=priv, name="p", version="1")
    sbom_layer = reg.read_manifest(aset.sbom)["layers"][0]["digest"]
    blob_path(reg.path, sbom_layer).write_bytes(b"{}")  # corrupt the SBOM blob
    with pytest.raises(SupplyChainError):
        verify(reg, subject, require=("sbom",))


# --- the Seal binding (RM-P1-SEAL-03) -------------------------------------------------------


def test_verify_unknown_require_token_refused(tmp_path: Path) -> None:
    """A typo in ``require`` must refuse, not silently skip the check it names.

    Hub's old orchestration decided each check with ``"sbom" in require``, so an unrecognised token
    disabled a check instead of failing: ``require=("signature", "slsa", "sbomm")`` verified an
    artifact carrying **no SBOM at all**. Seal refuses the unknown token outright.
    """
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    priv, pub = generate_keypair()
    attest(reg, subject, private_key_pem=priv, name="p", version="1")

    with pytest.raises(SupplyChainError, match="unknown required-evidence kind"):
        verify(reg, subject, trusted_public_key_pem=pub, require=("signature", "slsa", "sbomm"))


def test_appended_forged_signature_fails_closed(tmp_path: Path) -> None:
    """*Every* attached signature is checked — a bad one cannot hide behind a good one."""
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    priv, pub = generate_keypair()
    attest(reg, subject, private_key_pem=priv, name="p", version="1")
    verify(reg, subject, trusted_public_key_pem=pub)  # the honest signature alone verifies

    other_priv, _ = generate_keypair()  # an attacker appends a second, untrusted signature
    reg.attach(
        subject=subject,
        artifact_type=ARTIFACT_TYPE_SIGNATURE,
        blob=Blob(MEDIA_SIGNATURE, sign_digest(subject, other_priv).model_dump_json().encode()),
    )
    with pytest.raises(SupplyChainError):
        verify(reg, subject, trusted_public_key_pem=pub)


def test_supply_chain_error_is_seals(tmp_path: Path) -> None:
    """Hub's ``SupplyChainError`` **is** Seal's — the class downstream ``except`` clauses catch.

    Bench's leaderboard gate and Fleet both do ``except SupplyChainError`` on the class imported
    from ``astro_mine.hub.supply_chain``. If Hub ever reintroduced a local class while Seal's
    orchestrator raised its own, every one of those handlers would miss and the gate would fail
    **open**. This pins the alias.
    """
    assert SupplyChainError is seal.SupplyChainError

    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    with pytest.raises(seal.SupplyChainError):  # unsigned → refused, catchable as Seal's class
        verify(reg, subject)


def test_registry_binds_to_seals_attestation_store(tmp_path: Path) -> None:
    """Hub's adapter satisfies Seal's port — the seam that keeps Seal out of the OCI plane."""
    store = RegistryAttestationStore(Registry(tmp_path / "reg"))
    assert isinstance(store, seal.AttestationStore)
