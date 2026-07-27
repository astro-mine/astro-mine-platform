"""Plugin manifest v0.1 — typed Pydantic models (RM-P0-CORE-05).

The manifest is how every extension declares itself to the platform: its ``kind``, the
Core interface versions it is built against (the input to load-time **version
negotiation**), its declared inputs/outputs, its **capability tags** (the export-control
substrate), and its build-time **provenance** and **signature**. Authored as YAML/JSON
and validated against the canonical JSON Schema in ``schema/manifest.schema.json``
(shipped in-package); these models mirror it and a consistency test
(``tests/test_registry_consistency.py``) plus the drift guard keep the two aligned.

Design note (mirrors SADF): the models are **purely structural**. The dual-use semantic
gate (a manifest must not declare a reserved/gated capability tag) and load-time
negotiation/signature checks live in :mod:`astro_mine.core.registry.loader` and
:mod:`astro_mine.core.registry.registry`, so the models stay behaviourally identical to
the canonical JSON Schema.

Core *describes, validates, and resolves* — it never executes plugin code (core.md §9).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core import compat
from astro_mine.core.registry.enums import (
    CapabilityTag,
    DeterminismClass,
    PluginKind,
    Regime,
    SignatureKind,
    SignatureScheme,
)

__all__ = [
    "CatalogRecord",
    "CoreInterfaceVersion",
    "ManifestDocument",
    "PluginManifest",
    "Provenance",
    "Signature",
]

MANIFEST_VERSION: Literal["0.1"] = "0.1"


class _Model(BaseModel):
    """Base for every manifest model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


class Provenance(_Model):
    """Build-time provenance of the plugin artifact (conventions.md §5; core.md §9).

    The lineage by which a plugin is reproducible and auditable: the content hashes of
    its inputs, the producing code/toolchain versions, the environment lockfile, and the
    seed. ``source_content_hashes`` records named source artifacts (e.g. the CAD/URDF/USD
    a Fleet asset was imported from); ``builder_version`` the importer/converter/builder
    tool; ``digest`` the plugin artifact's own content digest (its content-addressed
    identity — Bench "the content hash *is* the task identity"; Fleet OCI packaging).

    This is **build-time** provenance and is distinct from Cloud's run-time
    ``RunContext`` (the execution envelope: image digest, resolved inputs, run id). Do
    not conflate them — together they form the full reproducibility chain.
    """

    input_hashes: list[str] = Field(default_factory=list)
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None
    seed: int | None = None
    source_content_hashes: dict[str, str] = Field(default_factory=dict)
    builder_version: str | None = None
    digest: str | None = None


class Signature(_Model):
    """The manifest signature block (core.md §9; conventions.md §9).

    Sigstore/cosign-shaped. In Phase 0 Core enforces the **contract** — that a load
    gate can require a signature's *presence and shape* — and delegates the cryptographic
    chain-of-trust (Rekor transparency log, certificate/identity verification) to a
    pluggable verifier supplied by the host (Cloud/Fleet, Phase 1). ``scheme``
    ``unsigned`` is an explicit, visible dev marker, never a silent default.

    ``kind`` distinguishes the attestation this block carries — a cosign signature, an
    SLSA/in-toto build-provenance attestation, or an SBOM — so a manifest's
    ``signatures[]`` can hold the heterogeneous set Hub verifies (hub.md §3, §9).
    ``digest`` is the content digest by which that attestation is fetched/audited via the
    OCI Referrers API (Hub stores attestations by digest, not inline).
    """

    scheme: SignatureScheme
    kind: SignatureKind = SignatureKind.COSIGN_SIGNATURE
    value: str | None = None
    payload: str | None = None
    certificate: str | None = None
    digest: str | None = None


class PluginManifest(_Model):
    """A plugin's self-declaration: identity, the Core interfaces it implements, its
    inputs/outputs, capability tags, provenance, and signature.

    ``core_interfaces`` maps a Core interface name (``env``, ``policy``, ``sadf``, …) to
    the SemVer the plugin is built against — the input the registry negotiates against
    *this* Core (RM-P0-CORE-07's :func:`~astro_mine.core.compat.assert_core_compatible`).
    ``inputs``/``outputs`` name the message/interface types the plugin consumes/produces
    (declarative, for compatibility checks). ``capability_tags`` is the reused autonomy/
    export-control vocabulary; the loader rejects any gated tag (open-commons gate).
    Kind-specific descriptors that Core does not schematize (a regime engine's fidelity
    descriptor, a metric's expected trace channels, resource budgets) ride in the open
    ``attributes`` map — keeping the waist thin while never blocking a downstream plugin.
    """

    name: str
    version: str
    kind: PluginKind
    core_interfaces: dict[str, str] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    capability_tags: list[CapabilityTag] = Field(default_factory=list)
    determinism_class: DeterminismClass | None = None
    regimes: list[Regime] = Field(default_factory=list)
    description: str | None = None
    # SPDX license id (Apache-2.0 for the commons). The download-boundary export/license
    # gate keys on it (hub.md §9; charter §10.4), so Hub needs it as a manifest field.
    license: str | None = None
    provenance: Provenance | None = None
    # ``signature`` is the legacy single-attestation block; ``signatures`` is the plural
    # set Hub indexes (cosign + SLSA provenance + SBOM). Both are honored — see
    # :attr:`all_signatures` — so a single-signature manifest stays valid unchanged.
    signature: Signature | None = None
    signatures: list[Signature] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def all_signatures(self) -> list[Signature]:
        """Every attestation on this manifest — the legacy singular ``signature`` (if
        present) followed by the plural ``signatures`` — the set Hub verifies and indexes."""
        return ([self.signature] if self.signature is not None else []) + list(self.signatures)

    def satisfies(
        self,
        *,
        interfaces: Mapping[str, str] | None = None,
        capability_tags: Iterable[CapabilityTag | str] | None = None,
    ) -> bool:
        """Whether this manifest satisfies a discovery/resolution constraint set.

        The Core-owned negotiation primitive Hub's resolver *consumes* rather than
        reimplements (hub.md §2 "discovery is capability negotiation, not string
        matching"; §3): each required Core interface must be satisfied by the manifest's
        declared ``core_interfaces`` under the same SemVer rule the registry applies at
        load (:func:`astro_mine.core.compat.check_compatible`), and every required
        capability tag must be declared. Returns ``True`` only if every constraint holds
        (an undeclared interface or an incompatible major/minor fails)."""
        for name, required in (interfaces or {}).items():
            declared = self.core_interfaces.get(name)
            if declared is None or not compat.check_compatible(required, declared):
                return False
        declared_tags = {str(t) for t in self.capability_tags}
        return all(str(tag) in declared_tags for tag in (capability_tags or ()))

    def to_catalog_record(self) -> CatalogRecord:
        """Project this manifest into Hub's catalog record (hub.md §3).

        Hub derives its catalog model **purely from the Core manifest** — never a
        Hub-private schema (hub.md §2). This is that projection: the exact indexed fields,
        with ``core_interfaces`` reshaped into the ``core_interface_versions[]`` array Hub
        indexes and both signature forms folded into ``signatures[]``. Hub-side facets
        (downloads, publisher, namespace, embedding) are added by Hub, not Core."""
        return CatalogRecord(
            name=self.name,
            version=self.version,
            kind=self.kind,
            core_interface_versions=[
                CoreInterfaceVersion(interface=name, version=version)
                for name, version in sorted(self.core_interfaces.items())
            ],
            capability_tags=list(self.capability_tags),
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            license=self.license,
            provenance=self.provenance,
            signatures=self.all_signatures,
        )


class CoreInterfaceVersion(_Model):
    """One ``(interface, version)`` pair — the array element of the
    ``core_interface_versions[]`` projection Hub indexes (the manifest authors the same
    data as the ``core_interfaces`` map; Hub indexes it as a list)."""

    interface: str
    version: str


class CatalogRecord(_Model):
    """The projection of a plugin manifest into Hub's catalog record (hub.md §3).

    Every field is derived **purely from the Core manifest** (via
    :meth:`PluginManifest.to_catalog_record`) so Hub indexes by the Core schema and never
    a Hub-private one (hub.md §2, principle 2: "if discovery needs a field, the field
    belongs in the Core manifest via RFC"). This is a *derived view*, not an authored
    document — it has no ``manifest_version`` and is not loaded from a file; hence no JSON
    Schema / wire form of its own."""

    name: str
    version: str
    kind: PluginKind
    core_interface_versions: list[CoreInterfaceVersion] = Field(default_factory=list)
    capability_tags: list[CapabilityTag] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    license: str | None = None
    provenance: Provenance | None = None
    signatures: list[Signature] = Field(default_factory=list)


class ManifestDocument(_Model):
    """Top-level plugin-manifest document. ``manifest_version`` pins the schema minor."""

    manifest_version: Literal["0.1"]
    manifest: PluginManifest
