"""The fail-closed signed-load gate: refuse unsigned/tampered specs & models (RM-P1-GUARD-05).

Safety-critical (guard.md §9.5; LUNAR-SR-002): the shield's correctness depends on the integrity of
its ``SafetySpec`` / ``CompiledSafetyModel`` inputs, so an unsigned, tampered, mismatched-digest, or
wrong-signer artifact must **refuse to load** — before the trusted core ever sees the bytes. The
gate is untrusted Python; the Rust TCB is unchanged (LUNAR-SR-004).
"""

from __future__ import annotations

from importlib import resources

import pytest
import yaml

from astro_mine.core.registry import (
    PluginManifest,
    Provenance,
    Signature,
    SignatureScheme,
)
from astro_mine.core.registry.enums import PluginKind
from astro_mine.core.registry.registry import UnsignedManifest
from astro_mine.guard.spec import (
    SafetyDocument,
    compile_spec,
    compiled_to_wire,
    generate_keypair,
    guard_verifier,
    load_signed_compiled_model,
    load_signed_safety_spec,
    register_safety_spec,
    sign_digest,
    sign_safety_manifest,
    signer_id,
)
from astro_mine.seal import SignatureError

ANCHOR_PATH = resources.files("astro_mine.guard.reference").joinpath(
    "safety_specs", "anchor.safety.yaml"
)


def _keys() -> tuple[bytes, bytes]:
    return generate_keypair()


# --- SafetySpec documents ------------------------------------------------------------------


def _anchor_doc(source: str) -> SafetyDocument:
    return SafetyDocument.model_validate(yaml.safe_load(source))


def test_signed_spec_round_trips_and_records_provenance() -> None:
    source = ANCHOR_PATH.read_text(encoding="utf-8")
    private_pem, public_pem = _keys()
    # Sign the artifact's content hash (the identity a manifest/verdict binds to).
    doc = _anchor_doc(source)
    sig = sign_digest(doc.content_hash(), private_pem)

    loaded, prov = load_signed_safety_spec(source, sig, trusted_public_key_pem=public_pem)
    assert loaded.content_hash() == doc.content_hash()
    assert prov.verified is True
    assert prov.pinned is True
    assert prov.content_hash == doc.content_hash()
    assert prov.signer_id == signer_id(public_pem)


def test_unpinned_load_still_verifies_the_signature() -> None:
    source = ANCHOR_PATH.read_text(encoding="utf-8")
    private_pem, _ = _keys()
    sig = sign_digest(_anchor_doc(source).content_hash(), private_pem)
    _loaded, prov = load_signed_safety_spec(source, sig)  # no pinned key
    assert prov.verified is True
    assert prov.pinned is False


def test_refuse_unsigned_scheme_spec() -> None:
    source = ANCHOR_PATH.read_text(encoding="utf-8")
    unsigned = Signature(scheme=SignatureScheme.UNSIGNED)
    with pytest.raises(SignatureError, match="not a cosign signature"):
        load_signed_safety_spec(source, unsigned)


def test_refuse_signature_over_a_different_digest() -> None:
    source = ANCHOR_PATH.read_text(encoding="utf-8")
    private_pem, public_pem = _keys()
    # A perfectly valid signature — but over a DIFFERENT artifact's digest.
    sig = sign_digest("sha256:" + "0" * 64, private_pem)
    with pytest.raises(SignatureError, match="payload does not match"):
        load_signed_safety_spec(source, sig, trusted_public_key_pem=public_pem)


def test_refuse_tampered_spec_bytes() -> None:
    source = ANCHOR_PATH.read_text(encoding="utf-8")
    private_pem, public_pem = _keys()
    sig = sign_digest(_anchor_doc(source).content_hash(), private_pem)  # for the pristine spec
    # Tamper: weaken the lander keep-out margin from 3.0 m to 0.0 m. The recomputed hash no longer
    # matches the signed digest, so the gate refuses the edited spec (fail closed).
    tampered = source.replace("margin_m: 3.0", "margin_m: 0.0")
    assert tampered != source
    with pytest.raises(SignatureError, match="payload does not match"):
        load_signed_safety_spec(tampered, sig, trusted_public_key_pem=public_pem)


def test_refuse_wrong_signer_spec() -> None:
    source = ANCHOR_PATH.read_text(encoding="utf-8")
    private_pem, _ = _keys()
    _, other_public = _keys()
    sig = sign_digest(_anchor_doc(source).content_hash(), private_pem)
    with pytest.raises(SignatureError, match="not the trusted key"):
        load_signed_safety_spec(source, sig, trusted_public_key_pem=other_public)


# --- CompiledSafetyModel wire payloads -----------------------------------------------------


def test_signed_compiled_model_round_trips(anchor_document: SafetyDocument) -> None:
    private_pem, public_pem = _keys()
    model = compile_spec(anchor_document, sample_period_s=120_960.0)
    wire = compiled_to_wire(model)
    sig = sign_digest(model.content_hash(), private_pem)
    loaded, prov = load_signed_compiled_model(wire, sig, trusted_public_key_pem=public_pem)
    assert loaded.content_hash() == model.content_hash()
    assert prov.content_hash == model.content_hash()
    assert prov.verified is True


def test_refuse_tampered_compiled_wire(anchor_document: SafetyDocument) -> None:
    private_pem, public_pem = _keys()
    model = compile_spec(anchor_document, sample_period_s=120_960.0)
    sig = sign_digest(model.content_hash(), private_pem)  # signature for the pristine model
    # Tamper the compiled model (drop a keep-out term), re-encode, load with the OLD signature.
    tampered = model.model_copy(update={"keep_out_terms": model.keep_out_terms[:-1]})
    assert tampered.content_hash() != model.content_hash()
    with pytest.raises(SignatureError, match="payload does not match"):
        load_signed_compiled_model(
            compiled_to_wire(tampered), sig, trusted_public_key_pem=public_pem
        )


# --- registry gate: register_safety_spec ---------------------------------------------------


def test_require_signature_refuses_unsigned(anchor_document: SafetyDocument) -> None:
    # A signature-requiring registry refuses an unsigned manifest (never fail-open).
    with pytest.raises(UnsignedManifest):
        register_safety_spec(anchor_document, require_signature=True)


def test_signed_registration_passes_the_gate(anchor_document: SafetyDocument) -> None:
    private_pem, public_pem = _keys()
    man = register_safety_spec(
        anchor_document,
        private_pem=private_pem,
        require_signature=True,
        trusted_public_key_pem=public_pem,
    )
    assert man.signature is not None
    assert man.signature.scheme == SignatureScheme.SIGSTORE_COSIGN
    # the signature binds the manifest's provenance.digest (the spec content hash)
    assert man.provenance is not None
    assert man.signature.payload == man.provenance.digest


def test_signed_registration_rejects_wrong_trusted_key(anchor_document: SafetyDocument) -> None:
    private_pem, _ = _keys()
    _, other_public = _keys()
    with pytest.raises(SignatureError, match="not the trusted key"):
        register_safety_spec(
            anchor_document,
            private_pem=private_pem,
            require_signature=True,
            trusted_public_key_pem=other_public,
        )


def test_default_registration_is_unsigned_dev(anchor_document: SafetyDocument) -> None:
    # Back-compat: the local/dev default stays require_signature=False (RFC-0004, opt-in signing).
    man = register_safety_spec(anchor_document)
    assert man.signature is None


def test_sign_manifest_without_digest_fails_closed() -> None:
    # Nothing to bind a signature to → refuse to sign (fail closed), never a dangling signature.
    private_pem, _ = _keys()
    manifest = PluginManifest(name="x", version="0.1", kind=PluginKind.POLICY)
    with pytest.raises(SignatureError, match=r"no provenance\.digest"):
        sign_safety_manifest(manifest, private_pem)


# --- guard_verifier fail-closed branches ---------------------------------------------------


def test_guard_verifier_refuses_manifest_without_digest() -> None:
    verifier = guard_verifier()
    manifest = PluginManifest(name="x", version="0.1", kind=PluginKind.POLICY)
    with pytest.raises(SignatureError, match=r"no provenance\.digest"):
        verifier(manifest)


def test_guard_verifier_refuses_manifest_without_cosign_signature() -> None:
    verifier = guard_verifier()
    digest = "sha256:" + "e" * 64
    manifest = PluginManifest(
        name="x",
        version="0.1",
        kind=PluginKind.POLICY,
        provenance=Provenance(digest=digest),
        signature=Signature(scheme=SignatureScheme.UNSIGNED),
    )
    with pytest.raises(SignatureError, match="no cosign signature"):
        verifier(manifest)
