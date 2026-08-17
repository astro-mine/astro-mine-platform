# SPDX-License-Identifier: Apache-2.0
"""Content-addressing + Core-catalog registration for the SafetySpec.

"Core-catalogued" means the Guard-**owned** SafetySpec schema is registered *through* the
Core plugin registry (:class:`~astro_mine.core.registry.PluginManifest`) — Guard never edits
``astro_mine.core.messages`` (guard.md §5; the surrogate ``build_surrogate_manifest``
precedent). A manifest declares the Core interface versions the spec is built against
(negotiated at load), names the ``SafetySpec`` / ``CompiledSafetyModel`` types it produces,
and pins the spec by content hash in ``provenance.digest`` so a signed manifest (GUARD-05)
commits to the exact reviewed contract.

Content-addressing uses the one platform primitive
(:func:`astro_mine.core.hashing.content_hash_json`, via
:meth:`~astro_mine.guard.spec.model.SafetyDocument.content_hash`): a spec's identity is the
``sha256:<hex>`` of its canonical JSON, so a design-time safety claim and an operational
reading of the same spec reproduce (guard.md §5, "content-addressed").

Signed loading (RM-P1-GUARD-05): :func:`register_safety_spec` keeps ``require_signature=False`` as
the local/dev default (RFC-0004 — signing is opt-in), and adds an opt-in signed path.
:func:`sign_safety_manifest` attaches a cosign :class:`~astro_mine.core.registry.Signature` over
the manifest's ``provenance.digest`` (the spec's content hash — the RFC-0004 cosign bind target),
and passing ``private_pem`` (+ optionally ``require_signature=True`` and a pinned
``trusted_public_key_pem``) makes the registry gate the load fail-closed via
:func:`~astro_mine.guard.spec.signed.guard_verifier`.
"""

from __future__ import annotations

from astro_mine.core.registry import (
    PluginKind,
    PluginManifest,
    PluginRegistry,
    Provenance,
)
from astro_mine.guard import __version__ as _GUARD_VERSION
from astro_mine.guard.spec.ir import CompiledSafetyModel
from astro_mine.guard.spec.model import SafetyDocument
from astro_mine.guard.spec.signed import guard_verifier
from astro_mine.seal import SignatureError, sign_digest

__all__ = [
    "COMPILED_MODEL_OUTPUT",
    "SAFETY_SPEC_INTERFACE_VERSIONS",
    "SAFETY_SPEC_OUTPUT",
    "build_safety_manifest",
    "compiled_content_hash",
    "register_safety_spec",
    "sign_safety_manifest",
    "spec_content_hash",
]

#: The Core interfaces the SafetySpec schema is built against — the input the registry
#: negotiates against *this* Core at load (:func:`astro_mine.core.compat.assert_core_compatible`).
#: ``messages`` (the SafetySpec embeds the Core ``Volume``), ``sadf`` (constraint sources are
#: referenced abstractly as SADF budget paths, resolved in GUARD-04), and ``registry`` (the
#: manifest itself). Held at the frozen Core 0.1.0 line (VERSIONING: Core interfaces frozen
#: through Phase 1).
SAFETY_SPEC_INTERFACE_VERSIONS: dict[str, str] = {
    "messages": "0.1.0",
    "sadf": "0.1.0",
    "registry": "0.1.0",
}

#: The type names the manifest declares it produces (the Core-catalogued Guard-owned types).
SAFETY_SPEC_OUTPUT = "astro_mine.guard.spec.SafetySpec"
COMPILED_MODEL_OUTPUT = "astro_mine.guard.spec.CompiledSafetyModel"


def spec_content_hash(document: SafetyDocument) -> str:
    """The ``sha256:<hex>`` content address of a SafetySpec document (its immutable identity)."""
    return document.content_hash()


def compiled_content_hash(model: CompiledSafetyModel) -> str:
    """The ``sha256:<hex>`` content address of a compiled safety model."""
    return model.content_hash()


def build_safety_manifest(
    document: SafetyDocument,
    *,
    name: str | None = None,
    version: str | None = None,
    code_version: str | None = None,
    toolchain_version: str | None = None,
    input_hashes: list[str] | None = None,
) -> PluginManifest:
    """Build the Core ``PluginManifest`` that catalogues a SafetySpec (unsigned; caller signs).

    Identity defaults to the spec's ``id`` and the running Guard version. The manifest declares
    the Core interfaces the spec is built against (:data:`SAFETY_SPEC_INTERFACE_VERSIONS`, the
    registry negotiates them), names the Guard-owned ``SafetySpec`` / ``CompiledSafetyModel``
    output types, and carries the spec's content hash as ``provenance.digest`` — the identity a
    signature (GUARD-05) binds to. It declares **no capability tags**: a generic declarative
    safety contract is open-commons science (guard.md §9.6), and no gated tag applies. It reuses
    the existing ``policy`` :class:`~astro_mine.core.registry.PluginKind` (Guard's headline
    ``PolicyShield`` is a Core Policy/Planner) — no new kind, no Core change (the surrogate
    precedent)."""
    spec = document.safety
    return PluginManifest(
        name=name or spec.id,
        version=version or _GUARD_VERSION,
        kind=PluginKind.POLICY,
        core_interfaces=dict(SAFETY_SPEC_INTERFACE_VERSIONS),
        inputs=["astro_mine.core.messages.Volume"],
        outputs=[SAFETY_SPEC_OUTPUT, COMPILED_MODEL_OUTPUT],
        license="Apache-2.0",
        description=(
            f"SafetySpec {spec.id!r} — a declarative Guard safety contract "
            f"(keep-out, budget, kinematic, and STL/MTL temporal constraints)."
        ),
        provenance=Provenance(
            digest=document.content_hash(),
            code_version=code_version or _GUARD_VERSION,
            toolchain_version=toolchain_version or f"astro-mine-guard {_GUARD_VERSION}",
            input_hashes=list(input_hashes) if input_hashes is not None else [],
        ),
    )


def sign_safety_manifest(manifest: PluginManifest, private_pem: bytes) -> PluginManifest:
    """Return a copy of ``manifest`` with a cosign :class:`~astro_mine.core.registry.Signature` over
    its ``provenance.digest`` (the spec's content hash — the RFC-0004 cosign bind target).

    The signature is set as the manifest's singular ``signature`` block — the field
    ``PluginRegistry``'s signature gate keys on (and which
    :attr:`~astro_mine.core.registry.PluginManifest.all_signatures`, hence
    :func:`~astro_mine.guard.spec.signed.guard_verifier`, also reads). Raises
    :class:`~astro_mine.seal.SignatureError` if the manifest carries no
    ``provenance.digest`` to bind to (fail closed — nothing to sign)."""
    digest = manifest.provenance.digest if manifest.provenance else None
    if digest is None:
        raise SignatureError("manifest has no provenance.digest to sign")
    return manifest.model_copy(update={"signature": sign_digest(digest, private_pem)})


def register_safety_spec(
    document: SafetyDocument,
    *,
    registry: PluginRegistry | None = None,
    name: str | None = None,
    version: str | None = None,
    private_pem: bytes | None = None,
    require_signature: bool = False,
    trusted_public_key_pem: bytes | None = None,
) -> PluginManifest:
    """Build and register a SafetySpec manifest through the Core plugin registry.

    Negotiates the declared Core interface versions against the registry's Core (raising if
    incompatible) and stores the manifest. When ``private_pem`` is given the manifest is cosign
    signed (:func:`sign_safety_manifest`) before registration.

    ``registry`` overrides the registry to register through; otherwise a fresh
    ``PluginRegistry(require_signature=..., verifier=...)`` is built from ``require_signature``
    (default ``False`` — the local/dev posture, RFC-0004) and, when ``trusted_public_key_pem`` is
    given, a fail-closed :func:`~astro_mine.guard.spec.signed.guard_verifier`. A signature-requiring
    registry refuses an unsigned manifest (raising
    :class:`~astro_mine.core.registry.UnsignedManifest`)."""
    manifest = build_safety_manifest(document, name=name, version=version)
    if private_pem is not None:
        manifest = sign_safety_manifest(manifest, private_pem)
    if registry is not None:
        reg = registry
    else:
        verifier = (
            guard_verifier(trusted_public_key_pem=trusted_public_key_pem)
            if trusted_public_key_pem is not None
            else None
        )
        reg = PluginRegistry(require_signature=require_signature, verifier=verifier)
    return reg.register(manifest)
