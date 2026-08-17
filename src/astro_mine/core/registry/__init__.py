# SPDX-License-Identifier: Apache-2.0
"""Plugin manifest, registry, and version negotiation (RM-P0-CORE-05).

How content is discovered, version-negotiated, signed, and loaded — and the
capability-tag vocabulary that is the substrate for export-control gating. The plugin
**manifest** (``kind``, the Core interface versions it implements, inputs/outputs,
capability tags, provenance, signature) is authored as YAML/JSON and validated against
the canonical schema in ``schema/manifest.schema.json``; the :class:`PluginRegistry`
resolves manifests and **gates every load** — refusing an incompatible Core-version, an
unsigned manifest, or a reserved/gated capability tag with a clear, actionable error.

Core *describes, validates, and resolves*; it never executes plugin code (core.md §9).
The capability-tag vocabulary is reused from SADF (:class:`CapabilityTag` /
``GATED_CAPABILITY_TAGS``) so a capability means the same thing at the asset and the
plugin boundary. Load-time negotiation builds on the :mod:`astro_mine.core.compat`
primitive. No Protobuf wire form yet — the manifest is a control-plane document, not a
per-tick message; cross-language wire + Hub indexing are deferred (P1).

Public API:

- the document — :class:`PluginManifest`, :class:`ManifestDocument`, :class:`Provenance`,
  :class:`Signature`, and the vocabularies :class:`PluginKind` / :class:`SignatureScheme`
  (and reused :class:`CapabilityTag`);
- load + validate — :func:`load_manifest` for the authored *document*,
  :func:`load_plugin_manifest` for the bare manifest a publisher stores as a config blob, plus
  :func:`validate_manifest` / :func:`load_schema`;
- the registry — :class:`PluginRegistry` and its gates'
  :class:`RegistryError` / :class:`ManifestValidationError` / :class:`IncompatibleManifest`
  / :class:`UnsignedManifest`;
- the policy-plugin declaration — :class:`TierPlugin` (+ :data:`TierFactory`), the manifest/factory
  pair an instantiating host resolves from an entry point. Core owns it because Allocate and Guard
  both implement it (conventions.md §3.3); the host that gates and instantiates is Mind's
  :class:`~astro_mine.mind.registry.TierRegistry`.

Backlog: RM-P0-CORE-05 — astro-mine-core#5
"""

from __future__ import annotations

from astro_mine.core.registry import enums, loader, model, registry, tier
from astro_mine.core.registry.enums import (
    CapabilityTag,
    PluginKind,
    SignatureKind,
    SignatureScheme,
)
from astro_mine.core.registry.loader import (
    ManifestValidationError,
    RegistryError,
    load_manifest,
    load_plugin_manifest,
    load_schema,
    validate_manifest,
)
from astro_mine.core.registry.model import (
    CatalogRecord,
    CoreInterfaceVersion,
    ManifestDocument,
    PluginManifest,
    Provenance,
    Signature,
)
from astro_mine.core.registry.registry import (
    IncompatibleManifest,
    PluginRegistry,
    UnsignedManifest,
    Verifier,
)
from astro_mine.core.registry.tier import TierFactory, TierPlugin

__all__ = [
    "CapabilityTag",
    "CatalogRecord",
    "CoreInterfaceVersion",
    "IncompatibleManifest",
    "ManifestDocument",
    "ManifestValidationError",
    "PluginKind",
    "PluginManifest",
    "PluginRegistry",
    "Provenance",
    "RegistryError",
    "Signature",
    "SignatureKind",
    "SignatureScheme",
    "TierFactory",
    "TierPlugin",
    "UnsignedManifest",
    "Verifier",
    "enums",
    "load_manifest",
    "load_plugin_manifest",
    "load_schema",
    "loader",
    "model",
    "registry",
    "tier",
    "validate_manifest",
]
