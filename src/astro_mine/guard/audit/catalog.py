# SPDX-License-Identifier: Apache-2.0
"""Core-catalog registration for the SafetyVerdict message (RM-P1-GUARD-06).

"Core-catalogued" means the Guard-**owned** SafetyVerdict schema is registered *through* the
Core plugin registry (:class:`~astro_mine.core.registry.PluginManifest`) — Guard never edits
``astro_mine.core.messages`` (guard.md §5, §6; the :mod:`astro_mine.guard.spec.catalog`
precedent). The manifest declares the Core interface versions the message is built against
(``messages`` — it is a runtime message; ``policy`` — it is the audit surface of the
``PolicyShield`` Policy; ``registry`` — the manifest itself), names the ``SafetyVerdict`` type
it produces, and pins the *schema* by content hash in ``provenance.digest`` so a signed
manifest (GUARD-05) commits to the exact reviewed record shape.

It declares **no capability tags**: an auditable safety-verdict record is open-commons science
(guard.md §9.6). It reuses the ``policy`` :class:`~astro_mine.core.registry.PluginKind` — the
verdict is the ``PolicyShield``'s output surface — so no new kind and no Core change.
"""

from __future__ import annotations

from astro_mine.core.hashing import content_hash_json
from astro_mine.core.registry import (
    PluginKind,
    PluginManifest,
    PluginRegistry,
    Provenance,
)
from astro_mine.guard import __version__ as _GUARD_VERSION
from astro_mine.guard.audit.model import load_schema

__all__ = [
    "SAFETY_VERDICT_INTERFACE_VERSIONS",
    "SAFETY_VERDICT_OUTPUT",
    "build_verdict_manifest",
    "register_verdict_schema",
    "verdict_schema_content_hash",
]

#: The Core interfaces the SafetyVerdict message is built against — the input the registry
#: negotiates against *this* Core at load (:func:`astro_mine.core.compat.assert_core_compatible`).
#: Held at the frozen Core 0.1.0 line (VERSIONING: Core interfaces frozen through Phase 1).
SAFETY_VERDICT_INTERFACE_VERSIONS: dict[str, str] = {
    "messages": "0.1.0",
    "policy": "0.1.0",
    "registry": "0.1.0",
}

#: The type name the manifest declares it produces (the Core-catalogued Guard-owned type).
SAFETY_VERDICT_OUTPUT = "astro_mine.guard.audit.SafetyVerdict"

#: The registered plugin name (stable identity for the catalogued verdict schema).
_VERDICT_MANIFEST_NAME = "astro-mine-guard-safety-verdict"


def verdict_schema_content_hash() -> str:
    """The ``sha256:<hex>`` content address of the canonical SafetyVerdict JSON Schema.

    The schema is the reviewed contract, so its content hash is the identity a signature
    (GUARD-05) binds to — the verdict analogue of a SafetySpec's document content hash."""
    return content_hash_json(load_schema())


def build_verdict_manifest(
    *,
    name: str | None = None,
    version: str | None = None,
    code_version: str | None = None,
    toolchain_version: str | None = None,
) -> PluginManifest:
    """Build the Core ``PluginManifest`` that catalogues the SafetyVerdict schema (unsigned).

    Identity defaults to the stable verdict-schema name and the running Guard version. The
    manifest declares the Core interfaces the message is built against
    (:data:`SAFETY_VERDICT_INTERFACE_VERSIONS`), names the Guard-owned ``SafetyVerdict`` output
    type, and carries the schema's content hash as ``provenance.digest``. It declares **no
    capability tags** and reuses the ``policy`` :class:`~astro_mine.core.registry.PluginKind`
    (the verdict is the ``PolicyShield``'s audit surface) — no new kind, no Core change."""
    return PluginManifest(
        name=name or _VERDICT_MANIFEST_NAME,
        version=version or _GUARD_VERSION,
        kind=PluginKind.POLICY,
        core_interfaces=dict(SAFETY_VERDICT_INTERFACE_VERSIONS),
        inputs=[],
        outputs=[SAFETY_VERDICT_OUTPUT],
        license="Apache-2.0",
        description=(
            "SafetyVerdict — the auditable per-tick output of the Guard arbiter "
            "(certified action, intervention reason, invoked spec clause(s), active layer, "
            "barrier-margin certificate, and reproducibility provenance)."
        ),
        provenance=Provenance(
            digest=verdict_schema_content_hash(),
            code_version=code_version or _GUARD_VERSION,
            toolchain_version=toolchain_version or f"astro-mine-guard {_GUARD_VERSION}",
            input_hashes=[],
        ),
    )


def register_verdict_schema(
    *,
    registry: PluginRegistry | None = None,
    name: str | None = None,
    version: str | None = None,
) -> PluginManifest:
    """Build and register the SafetyVerdict manifest through the Core plugin registry.

    Negotiates the declared Core interface versions against the registry's Core (raising if
    incompatible) and stores the manifest. ``registry`` defaults to a fresh
    ``PluginRegistry(require_signature=False)`` — signed loading is GUARD-05, out of scope."""
    reg = registry if registry is not None else PluginRegistry(require_signature=False)
    manifest = build_verdict_manifest(name=name, version=version)
    return reg.register(manifest)
