"""Fail-closed signed loading of SafetySpec / CompiledSafetyModel artifacts (RM-P1-GUARD-05).

The shield's correctness depends on its inputs, so their integrity is part of the safety
guarantee (guard.md §9.5; ``LUNAR-SR-002``): Guard **refuses to load** an unsigned or tampered
``SafetySpec`` document or ``CompiledSafetyModel``. This is a **fail-closed Python load gate that
runs before the trusted core ever sees the bytes** — it is *not* a change to the Rust TCB
(``LUNAR-SR-004``; the crypto stays out of the minimal, Kani-amenable core, guard.md §9.1).

Two independent integrity re-derivations (verify-twice):

1. **This gate** recomputes the artifact's ``content_hash`` from the loaded/validated bytes and
   verifies the cosign :class:`~astro_mine.core.registry.Signature` covers *that* hash under a
   pinned trusted key — refusing (raising :class:`~astro_mine.seal.SignatureError`)
   on an unsigned / ``unsigned``-scheme / tampered / mismatched-digest / wrong-signer artifact,
   **before** ``CoreConfig.build(wire)``.
2. **The trusted core** independently re-derives and reports the ``spec_content_hash`` it enforced
   into every :class:`~astro_mine.guard.audit.model.SafetyVerdict` (``rust/src/model.rs``), so a
   model that slipped past the gate but does not match still fail-closes inside the TCB.

Tampering is caught structurally: the signature binds a *content* hash, and this gate recomputes
that hash from the bytes it actually loaded — so any edit to the spec/model changes the recomputed
hash, which then no longer matches the signed payload, and verification fails.
"""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.core.registry import (
    PluginManifest,
    Signature,
    SignatureScheme,
    Verifier,
)
from astro_mine.guard.spec.ir import CompiledSafetyModel
from astro_mine.guard.spec.keyid import signer_id
from astro_mine.guard.spec.loader import load_safety_spec
from astro_mine.guard.spec.model import SafetyDocument
from astro_mine.guard.spec.wire import compiled_from_wire
from astro_mine.seal import SignatureError, verify_signature

__all__ = [
    "LoadedArtifactProvenance",
    "guard_verifier",
    "load_signed_compiled_model",
    "load_signed_safety_spec",
]


@dataclass(frozen=True, slots=True)
class LoadedArtifactProvenance:
    """The provenance a verified-load records for a downstream verdict (guard.md §9.5).

    ``content_hash`` is the ``sha256:<hex>`` identity recomputed from the loaded bytes (equal to the
    signed ``payload``); ``signer_id`` fingerprints the key that signed it; ``verified`` is the
    fail-closed outcome (always ``True`` when returned — a failed verification raises instead) and
    ``pinned`` records whether the load pinned a specific trusted key. Bound to the same hashes the
    shield already stamps onto :class:`~astro_mine.guard.audit.model.SafetyVerdict`
    (``spec_content_hash`` / ``compiled_content_hash``)."""

    content_hash: str
    signer_id: str
    verified: bool
    pinned: bool


def _verify_artifact(
    signature: Signature, content_hash: str, trusted_public_key_pem: bytes | None
) -> LoadedArtifactProvenance:
    """Verify ``signature`` covers ``content_hash`` (fail closed); build the provenance record."""
    verify_signature(signature, content_hash, trusted_public_key_pem=trusted_public_key_pem)
    # verify_signature has already proven the envelope is complete and covers this artifact.
    assert signature.certificate is not None
    return LoadedArtifactProvenance(
        content_hash=content_hash,
        signer_id=signer_id(signature.certificate.encode()),
        verified=True,
        pinned=trusted_public_key_pem is not None,
    )


def load_signed_safety_spec(
    source: str | bytes,
    signature: Signature,
    *,
    trusted_public_key_pem: bytes | None = None,
) -> tuple[SafetyDocument, LoadedArtifactProvenance]:
    """Parse + validate a SafetySpec, then verify its cosign signature — refuse on any failure.

    Runs the full structural + fail-safe :func:`~astro_mine.guard.spec.loader.load_safety_spec`
    pipeline, recomputes the document's ``content_hash``, and verifies ``signature`` covers it under
    ``trusted_public_key_pem`` (when pinned). Raises
    :class:`~astro_mine.seal.SignatureError` on an unsigned / tampered / mismatched /
    wrong-signer artifact, or :class:`~astro_mine.guard.spec.loader.SafetySpecValidationError` on a
    malformed spec — the artifact never loads unverified (fail closed)."""
    document = load_safety_spec(source)
    provenance = _verify_artifact(signature, document.content_hash(), trusted_public_key_pem)
    return document, provenance


def load_signed_compiled_model(
    wire: bytes,
    signature: Signature,
    *,
    trusted_public_key_pem: bytes | None = None,
) -> tuple[CompiledSafetyModel, LoadedArtifactProvenance]:
    """Decode a CompiledSafetyModel wire payload, then verify its cosign signature — refuse on any
    failure — **before** the bytes are handed to ``CoreConfig.build`` / the trusted core.

    Decodes ``wire`` (a ``compiled_to_wire`` protobuf), recomputes the model's ``content_hash``, and
    verifies ``signature`` covers it under ``trusted_public_key_pem`` (when pinned). Any tamper to
    ``wire`` changes the recomputed hash, so verification fails. Raises
    :class:`~astro_mine.seal.SignatureError` on an unsigned / tampered / mismatched /
    wrong-signer artifact (fail closed)."""
    model = compiled_from_wire(wire)
    provenance = _verify_artifact(signature, model.content_hash(), trusted_public_key_pem)
    return model, provenance


def guard_verifier(*, trusted_public_key_pem: bytes | None = None) -> Verifier:
    """A Core ``Verifier`` that checks a SafetySpec manifest's own cosign signature at load.

    The ``Callable[[PluginManifest], None]`` shape ``PluginRegistry`` invokes during its signature
    gate (``core/registry/registry.py``): it finds a cosign signature among the manifest's
    :attr:`~astro_mine.core.registry.PluginManifest.all_signatures` and verifies it against the
    manifest's ``provenance.digest`` — the artifact's content identity (RFC-0004: the cosign bind
    target). Raises :class:`~astro_mine.seal.SignatureError` on any failure, aborting
    the load (fail closed). Mirrors astro-mine-hub's ``make_verifier``."""

    def _verify(manifest: PluginManifest) -> None:
        digest = manifest.provenance.digest if manifest.provenance else None
        if digest is None:
            raise SignatureError("manifest has no provenance.digest to bind a signature to")
        cosign = [s for s in manifest.all_signatures if s.scheme == SignatureScheme.SIGSTORE_COSIGN]
        if not cosign:
            raise SignatureError("manifest carries no cosign signature")
        for signature in cosign:
            verify_signature(signature, digest, trusted_public_key_pem=trusted_public_key_pem)

    return _verify
