"""Plugin manifest v0.1 — closed vocabularies (Core-owned, RM-P0-CORE-05).

The *closed* part of the registry waist: the plugin **kind** vocabulary every
extension declares itself as, and the **signature scheme** the manifest is signed
under. Like the SADF vocabularies they are deliberately small and grow only by RFC
(conventions.md §3); adding a member is append-only — members are never removed or
repurposed.

The capability-tag vocabulary the manifest declares is **not** redefined here — the
registry reuses :class:`astro_mine.core.sadf.enums.CapabilityTag` and the gated subset
:data:`astro_mine.core.sadf.enums.GATED_CAPABILITY_TAGS`, so a capability means exactly
the same thing whether an *asset* (SADF) or a *plugin* (manifest) declares it. The same
holds for :class:`~astro_mine.core.sadf.enums.DeterminismClass` (an engine plugin's
declared determinism) and :class:`~astro_mine.core.sadf.enums.Regime` (which regimes a
plugin serves). They are re-exported here so the canonical manifest schema can reference
them by name and the drift guard checks them against the one SADF-owned definition.
"""

from __future__ import annotations

from enum import StrEnum

from astro_mine.core.sadf.enums import CapabilityTag, DeterminismClass, Regime

__all__ = [
    "CapabilityTag",
    "DeterminismClass",
    "PluginKind",
    "Regime",
    "SignatureKind",
    "SignatureScheme",
]


class PluginKind(StrEnum):
    """Closed, Core-owned vocabulary of the content kinds resolved through the registry.

    Most kinds implement a Core interface (Environment / Policy / SADF / ResourceField,
    …) and are discovered and version-negotiated through :class:`PluginRegistry`. Core
    only *describes, validates, and resolves* a manifest — it never executes plugin
    code (core.md §9); instantiation/sandboxing is the host component's job.

    Some kinds are **packaging metadata** for content nobody loads as code: an ``asset``
    manifest describes a SADF document instantiated by Sim's loader, not by the registry,
    and ``design``/``campaign`` describe frozen Studio artifacts whose bytes Core never
    parses (RFC-0008). The vocabulary names what Core *describes for discovery*, not only
    what it executes; :class:`~astro_mine.core.registry.PluginRegistry` resolves the former
    and ignores the latter.

    Members map to the extension surfaces named across the component backlogs:
    Sim (``regime_engine``/``sensor_model``/``coupling_scheme``), Worlds
    (``world_provider``/``body_pack``/``field_model``), Prospect
    (``resource_field_backend``/``observation_model``/``prior_recipe``/
    ``info_gain_objective``), Link (``comms_model``), Fleet (``asset``), Bench
    (``policy``/``metric``), and Studio (``design``/``campaign``).
    Mission-architecture kinds (RFC-0001 trajectory/mission/sizing) arrive by RFC in
    Phase 3.
    """

    # Sim extension points (sim.md §3)
    REGIME_ENGINE = "regime_engine"
    SENSOR_MODEL = "sensor_model"
    COUPLING_SCHEME = "coupling_scheme"
    # Worlds extension points (worlds.md §3)
    WORLD_PROVIDER = "world_provider"
    BODY_PACK = "body_pack"
    FIELD_MODEL = "field_model"
    # Prospect extension points (prospect.md §3)
    RESOURCE_FIELD_BACKEND = "resource_field_backend"
    OBSERVATION_MODEL = "observation_model"
    PRIOR_RECIPE = "prior_recipe"
    INFO_GAIN_OBJECTIVE = "info_gain_objective"
    # Link comms-environment extension point (link.md §3) — parallel to the Worlds
    # world_provider / field_model surfaces
    COMMS_MODEL = "comms_model"
    # Fleet asset packaging metadata (fleet.md §3) — instantiated by Sim, not the registry
    ASSET = "asset"
    # Autonomy & benchmark plugins (bench.md §3)
    POLICY = "policy"
    METRIC = "metric"
    # Studio design artifacts (studio.md §5, §6; RFC-0008) — like `asset`, packaging metadata for
    # content nobody loads as code. `design` is a frozen TradeStudy/EvaluatedCandidate; `campaign`
    # is the frozen Campaign Ops consumes unchanged. Both exist so Hub can index a published design
    # by the Core manifest rather than a Studio-private schema (hub.md §2 principle 2).
    DESIGN = "design"
    CAMPAIGN = "campaign"


class SignatureScheme(StrEnum):
    """How a manifest's :class:`~astro_mine.core.registry.model.Signature` was produced.

    ``sigstore_cosign`` is the platform default (conventions.md §9; core.md §9). The
    explicit ``unsigned`` marker exists so local/dev manifests fail *loudly and
    visibly* at a signature-requiring registry rather than by silent omission — it is
    never a default and a hardened registry refuses it.
    """

    SIGSTORE_COSIGN = "sigstore_cosign"
    UNSIGNED = "unsigned"


class SignatureKind(StrEnum):
    """What attestation a :class:`~astro_mine.core.registry.model.Signature` carries.

    A plugin artifact accrues more than one attestation, which Hub verifies twice (at
    admission and at pull) and serves independently by digest via the OCI Referrers API
    (hub.md §3, §9): a **cosign signature** over the artifact, an **SLSA / in-toto build
    provenance** attestation, and an **SBOM**. ``kind`` lets a manifest's ``signatures[]``
    hold the heterogeneous set and lets a verifier route each to the right check. Defaults
    to ``cosign_signature`` so a legacy single-signature manifest keeps its meaning.
    """

    COSIGN_SIGNATURE = "cosign_signature"
    SLSA_PROVENANCE = "slsa_provenance"
    SBOM = "sbom"
