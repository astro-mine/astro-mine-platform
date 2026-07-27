"""Fail-closed load of a served surrogate (RM-P1-SURR-04; surrogate.md §9).

The signed :class:`~astro_mine.core.registry.PluginManifest` is verified **before** the ONNX
artifact is trusted — mirroring Core's manifest-signing rule. The gate is three fail-closed checks
in order: (1) the registry's signature gate rejects an unsigned / tampered-digest / untrusted-key
manifest; (2) the bundle's content hash must equal the ``provenance.digest`` the signature binds to
— so a swapped artifact is caught; (3) the bundle's ``ErrorReport`` must hash to the
``error_report_digest`` the manifest declares — so the served bound is exactly the signed one. Any
mismatch raises :class:`ServedIntegrityError`; only then is the ONNX Runtime session built.

:func:`resolve_and_load` is the Hub path — resolve a ``name:version`` (or digest) reference, pull
the signed manifest + the ONNX-bundle layer by content hash, and run the same gate. Hub is imported
lazily (the ``[publish]`` extra); the in-memory :func:`load_served_surrogate` needs only Core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.core.hashing import content_hash
from astro_mine.core.registry import PluginManifest, PluginRegistry
from astro_mine.surrogate.serve.bundle import ONNX_BUNDLE_MEDIA_TYPE, OnnxBundle
from astro_mine.surrogate.serve.runtime import OnnxServedSurrogate

if TYPE_CHECKING:
    from astro_mine.core.registry import Verifier
    from astro_mine.hub.registry import Registry

__all__ = ["ServedIntegrityError", "load_served_surrogate", "resolve_and_load"]


class ServedIntegrityError(Exception):
    """The signed manifest does not match the bundle — a swapped artifact or report.

    Distinct from the registry's signature errors (which reject a bad *signature*): this fires when
    the signature is valid but the bytes it commits to (the ONNX bundle, the ErrorReport) do not
    match what was delivered. Fail-closed: the surrogate is never constructed.
    """


def load_served_surrogate(
    *,
    bundle_bytes: bytes,
    manifest: PluginManifest,
    verifier: Verifier | None = None,
) -> OnnxServedSurrogate:
    """Verify a signed manifest against ``bundle_bytes`` and build the served surrogate.

    Registers ``manifest`` in a signature-requiring :class:`PluginRegistry` (raising
    ``UnsignedManifest`` / ``SignatureError`` on a bad signature), then checks the bundle hash and
    the embedded ErrorReport hash against the manifest before constructing the ONNX Runtime tier.
    Pass a ``verifier`` (a trusted-key verifier from ``astro_mine.hub.supply_chain.make_verifier``)
    for a real gate; without one the registry only checks that a signature is *present*.
    """
    registry = PluginRegistry(require_signature=True, verifier=verifier)
    registry.register(manifest)  # fail-closed on unsigned / tampered-digest / untrusted-key

    if manifest.provenance is None:
        raise ServedIntegrityError("manifest carries no provenance.digest to bind the artifact to")
    if content_hash(bundle_bytes) != manifest.provenance.digest:
        raise ServedIntegrityError(
            f"bundle hash {content_hash(bundle_bytes)} does not match the signed "
            f"provenance.digest {manifest.provenance.digest}"
        )
    bundle = OnnxBundle.parse(bundle_bytes)
    declared = manifest.attributes.get("error_report_digest")
    actual = bundle.error_report.content_hash()
    if actual != declared:
        raise ServedIntegrityError(
            f"bundle ErrorReport hash {actual} does not match the manifest's "
            f"error_report_digest {declared!r}"
        )
    return OnnxServedSurrogate(bundle)


def resolve_and_load(
    registry: Registry,
    reference: str,
    *,
    verifier: Verifier | None = None,
) -> OnnxServedSurrogate:
    """Resolve a Hub reference, pull the signed manifest + ONNX bundle, and load fail-closed.

    ``reference`` is a ``name:version`` tag or a ``sha256:`` digest. Reads the artifact's config
    blob as the signed :class:`PluginManifest`, pulls the single ``ONNX_BUNDLE_MEDIA_TYPE`` layer by
    content hash, and runs :func:`load_served_surrogate`.
    """
    descriptor = registry.resolve(reference)
    manifest = PluginManifest.model_validate_json(registry.read_config(descriptor.digest))
    image_manifest = registry.read_manifest(descriptor.digest)
    layers = [
        layer for layer in image_manifest["layers"] if layer["mediaType"] == ONNX_BUNDLE_MEDIA_TYPE
    ]
    if len(layers) != 1:
        raise ServedIntegrityError(
            f"expected exactly one {ONNX_BUNDLE_MEDIA_TYPE} layer, found {len(layers)}"
        )
    bundle_bytes = registry.pull_blob(layers[0]["digest"])
    return load_served_surrogate(bundle_bytes=bundle_bytes, manifest=manifest, verifier=verifier)
