# SPDX-License-Identifier: Apache-2.0
"""Publishing a frozen design or campaign to Hub (RM-P1-STUDIO-06; studio.md §6 →Hub).

Studio "can write back published designs/campaigns as **content-addressed, signed** artifacts"
(studio.md §6), and Hub indexes them "by the Core plugin manifest", refusing "to invent its own
parallel metadata schema for anything Core already describes" (hub.md §2 principle 2). Those two
sentences fix the whole design:

* The artifact is an ordinary OCI artifact whose **config is a Core** :class:`PluginManifest`, of
  kind ``campaign`` / ``design`` — the vocabulary RFC-0008 added to Core precisely so that Studio
  would not need a private one.
* The **payload is a layer** whose bytes Core never parses: the canonical JSON of Studio's own
  ``Campaign``/``TradeStudy``. Studio's schema stays Studio's (studio.md §12, "Studio adds no Core
  surface of its own"); what Core supplies is the *index*.
* Signing, SLSA provenance, and the SBOM are Hub's — ``HubClient.publish(private_key_pem=…)``
  delegates to ``astro_mine.seal``. Studio holds no crypto and no key material of its own.

Only **frozen** artifacts publish: "once a study runs or a campaign is handed off, the artifact is
frozen and content-addressed" (studio.md §5). The frozen bundle's own content hash is recorded
in the manifest's ``provenance.digest``, and :meth:`HubArtifactPublisher.pull_campaign`
re-derives it from the layer bytes — so a payload swapped underneath an otherwise valid signature
is still rejected.

**Capability tags are inherited, never invented.** A design's export-control posture is the union of
its assets' capability tags; Studio "honors but does not redefine that partition" (studio.md §9).
Every ``sadf_ref`` in the swarm is resolved against Hub and its tags folded in. A ref that does not
resolve is a hard error: publishing while silently understating what a swarm can do would route it
past the OPA/capability gate those tags exist to feed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astro_mine.core.registry import CapabilityTag, PluginKind, PluginManifest, Provenance

from .._base import FrozenStudioModel
from ..campaign import CampaignBundle, load_campaign
from ..hashing import canonical_json, content_hash
from ..models import Campaign, TradeStudy
from ..provenance import ArtifactProvenance

if TYPE_CHECKING:  # pragma: no cover - the [hub] extra is not in the base wheel
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Registry

__all__ = [
    "CAMPAIGN_ARTIFACT_KIND",
    "CAMPAIGN_LAYER_MEDIA_TYPE",
    "DESIGN_ARTIFACT_KIND",
    "DESIGN_LAYER_MEDIA_TYPE",
    "ArtifactPublisher",
    "CapabilityResolver",
    "HubArtifactPublisher",
    "HubCapabilityResolver",
    "PublishError",
    "PublishedArtifactRef",
    "build_campaign_manifest",
    "build_design_manifest",
]

#: The Hub artifact kinds (RFC-0008) → ``application/vnd.astro-mine.<kind>.v1``.
CAMPAIGN_ARTIFACT_KIND = "campaign"
DESIGN_ARTIFACT_KIND = "design"

#: The payload layers' media types. Bytes Core never parses; Studio's canonical JSON.
CAMPAIGN_LAYER_MEDIA_TYPE = "application/vnd.astro-mine.campaign.bundle.v1.json"
DESIGN_LAYER_MEDIA_TYPE = "application/vnd.astro-mine.design.bundle.v1.json"


class PublishError(RuntimeError):
    """A design/campaign could not be published, or a pulled one could not be trusted."""


class PublishedArtifactRef(FrozenStudioModel):
    """Where a published artifact lives, and what it is.

    ``digest`` is the OCI image-manifest digest — the artifact's identity in Hub, and the handle a
    consumer (Ops, a colleague) pulls it back by. ``content_digest`` is the hash of the *payload*
    bytes, i.e. the frozen bundle's own content address. They differ because the former also covers
    the manifest and the layer descriptors.
    """

    reference: str
    digest: str
    content_digest: str
    kind: str


@runtime_checkable
class CapabilityResolver(Protocol):
    """Resolves a Fleet SADF reference to the capability tags its asset manifest declares."""

    def capability_tags(self, sadf_ref: str) -> Sequence[CapabilityTag]: ...


@runtime_checkable
class ArtifactPublisher(Protocol):
    """The →Hub seam. A deployment binds it to Hub; a test binds it to a temp registry."""

    def publish_campaign(
        self, bundle: CampaignBundle, *, name: str, version: str
    ) -> PublishedArtifactRef: ...

    def publish_trade_study(
        self, study: TradeStudy, *, name: str, version: str
    ) -> PublishedArtifactRef: ...

    def pull_campaign(self, reference: str) -> Campaign: ...


class HubCapabilityResolver:
    """Reads a Fleet asset's capability tags out of its published Core manifest.

    Studio never opens a SADF document: the tags it needs are already on the ``asset`` manifest Hub
    indexes, which is what "pure Core consumer, no sibling-package imports" means in practice.
    """

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def capability_tags(self, sadf_ref: str) -> Sequence[CapabilityTag]:
        try:
            digest = self._registry.resolve(sadf_ref).digest
            manifest = PluginManifest.model_validate_json(self._registry.read_config(digest))
        except Exception as exc:
            raise PublishError(
                f"cannot resolve asset {sadf_ref!r} in Hub, so its capability tags are unknown; "
                "refusing to publish a design that understates what its swarm can do"
            ) from exc
        return manifest.capability_tags


def _inherited_capability_tags(
    refs: Iterable[str], resolver: CapabilityResolver
) -> list[CapabilityTag]:
    """The union of a swarm's declared capabilities, sorted so the manifest is deterministic."""
    tags: set[CapabilityTag] = set()
    for ref in refs:
        tags.update(resolver.capability_tags(ref))
    return sorted(tags, key=lambda tag: tag.value)


def _core_provenance(provenance: ArtifactProvenance, *, content_digest: str) -> Provenance:
    """Project Studio's reproducibility envelope onto Core's ``Provenance``.

    Nothing is invented: `conventions.md` §5 requires a generated artifact to record "its inputs
    (content hashes), the producing code version, the environment lockfile, and the random seed",
    and Studio's ``ArtifactProvenance`` already carries exactly those. This is the shape Hub's
    catalog indexes them in.
    """
    return Provenance(
        input_hashes=list(provenance.input_hashes),
        code_version=provenance.code_version,
        toolchain_version=provenance.toolchain_version,
        env_lockfile=provenance.env_lockfile,
        seed=provenance.seed,
        digest=content_digest,
    )


def build_campaign_manifest(
    bundle: CampaignBundle,
    *,
    name: str,
    version: str,
    capability_tags: Sequence[CapabilityTag] = (),
) -> PluginManifest:
    """The Core manifest Hub indexes a published campaign by.

    ``core_interfaces`` is empty, and that is the point: a campaign implements no Core interface and
    nothing loads it as code (RFC-0008). ``outputs`` are the metric keys the chosen candidate was
    scored on, so a catalog search finds campaigns scored on a metric without parsing the payload.
    """
    campaign = bundle.campaign
    return PluginManifest(
        name=name,
        version=version,
        kind=PluginKind.CAMPAIGN,
        core_interfaces={},
        inputs=list(campaign.provenance.input_hashes),
        outputs=sorted(campaign.chosen.score.metric_scores),
        capability_tags=list(capability_tags),
        description=campaign.name,
        provenance=_core_provenance(campaign.provenance, content_digest=bundle.digest),
        attributes={
            "bundle_media_type": CAMPAIGN_LAYER_MEDIA_TYPE,
            "campaign_id": campaign.id,
            "objective_hash": campaign.objective_hash,
            "trade_study_ref": campaign.trade_study_ref,
            "world_ref": campaign.chosen.world_ref,
        },
    )


def build_design_manifest(
    study: TradeStudy,
    *,
    name: str,
    version: str,
    content_digest: str,
    capability_tags: Sequence[CapabilityTag] = (),
) -> PluginManifest:
    """The Core manifest Hub indexes a published trade study by.

    RFC-0008 leaves open whether a study publishes as one artifact or as a manifest referencing
    separately published candidates. Phase 1 publishes one artifact, one layer: candidates dedupe
    across studies rarely enough that the extra round trips are not yet worth it.
    """
    metrics: set[str] = set()
    for evaluated in study.evaluated:
        metrics.update(evaluated.score.metric_scores)
    return PluginManifest(
        name=name,
        version=version,
        kind=PluginKind.DESIGN,
        core_interfaces={},
        inputs=list(study.provenance.input_hashes),
        outputs=sorted(metrics),
        capability_tags=list(capability_tags),
        description=f"trade study {study.id} ({len(study.evaluated)} candidates)",
        provenance=_core_provenance(study.provenance, content_digest=content_digest),
        attributes={
            "bundle_media_type": DESIGN_LAYER_MEDIA_TYPE,
            "study_id": study.id,
            "objective_hash": study.objective_hash,
            "backend": study.backend,
            "pareto_front_size": len(study.pareto_front),
        },
    )


class HubArtifactPublisher:
    """Publishes through the ``astro-mine-hub`` client — ORAS + cosign, no bespoke protocol.

    ``private_key_pem`` is the ECDSA P-256 signing key (RFC-0005's offline default; keyless
    cosign is Phase 2). The client's ``trusted_public_key_pem`` pins which signer a pull will
    accept: :meth:`pull_campaign` re-verifies signature + SLSA + SBOM and **fails closed**, so a
    compromised
    registry cannot make Studio accept a campaign it did not publish (hub.md §2 principle 3).
    """

    def __init__(
        self,
        client: HubClient,
        *,
        capability_resolver: CapabilityResolver,
        private_key_pem: bytes,
    ) -> None:
        self._client = client
        self._capability_resolver = capability_resolver
        self._private_key_pem = private_key_pem

    # -- publish ---------------------------------------------------------------------------

    def publish_campaign(
        self, bundle: CampaignBundle, *, name: str, version: str
    ) -> PublishedArtifactRef:
        """Publish a frozen campaign as a signed, content-addressed artifact."""
        swarm = bundle.campaign.chosen.candidate.swarm
        tags = _inherited_capability_tags((s.sadf_ref for s in swarm), self._capability_resolver)
        manifest = build_campaign_manifest(bundle, name=name, version=version, capability_tags=tags)
        return self._publish(
            manifest,
            payload=bundle.payload(),
            content_digest=bundle.digest,
            kind=CAMPAIGN_ARTIFACT_KIND,
            media_type=CAMPAIGN_LAYER_MEDIA_TYPE,
            name=name,
            version=version,
            inputs=bundle.campaign.provenance.input_hashes,
        )

    def publish_trade_study(
        self, study: TradeStudy, *, name: str, version: str
    ) -> PublishedArtifactRef:
        """Publish a frozen trade study — its evaluated candidates and Pareto front."""
        payload = canonical_json(study.model_dump(mode="json"))
        content_digest = content_hash(payload)
        refs = [
            selection.sadf_ref
            for evaluated in study.evaluated
            for selection in evaluated.candidate.swarm
        ]
        tags = _inherited_capability_tags(refs, self._capability_resolver)
        manifest = build_design_manifest(
            study, name=name, version=version, content_digest=content_digest, capability_tags=tags
        )
        return self._publish(
            manifest,
            payload=payload,
            content_digest=content_digest,
            kind=DESIGN_ARTIFACT_KIND,
            media_type=DESIGN_LAYER_MEDIA_TYPE,
            name=name,
            version=version,
            inputs=study.provenance.input_hashes,
        )

    def _publish(
        self,
        manifest: PluginManifest,
        *,
        payload: bytes,
        content_digest: str,
        kind: str,
        media_type: str,
        name: str,
        version: str,
        inputs: Sequence[str],
    ) -> PublishedArtifactRef:
        from astro_mine.hub.registry import Blob

        try:
            artifact = self._client.publish(
                name=name,
                version=version,
                kind=kind,
                manifest=manifest,
                layers=[Blob(media_type, payload)],
                private_key_pem=self._private_key_pem,
                inputs=list(inputs),
            )
        except Exception as exc:
            raise PublishError(f"could not publish {kind} {name}:{version}: {exc}") from exc

        return PublishedArtifactRef(
            reference=f"{name}:{version}",
            digest=artifact.digest,
            content_digest=content_digest,
            kind=kind,
        )

    # -- pull ------------------------------------------------------------------------------

    def pull_campaign(self, reference: str) -> Campaign:
        """Pull a published campaign by reference or digest, re-verifying before trusting it.

        Ops consumes "the same Core-defined artifact Studio produced" (studio.md §9). Verification
        is threefold and fail-closed: Hub re-checks the signature/SLSA/SBOM, the payload layer's
        bytes are re-hashed against the content address the **verified** manifest commits to
        (:meth:`HubClient.pull_payload`), and the payload is re-hashed against the digest the
        manifest *recorded*. The third check is not redundant: a layer swapped at publish time is a
        legitimate layer of a validly signed manifest, so it passes the first two and only the
        recorded content digest catches it.
        """
        payload, manifest = self._pull_payload(
            reference, PluginKind.CAMPAIGN, CAMPAIGN_LAYER_MEDIA_TYPE
        )
        recorded = manifest.provenance.digest if manifest.provenance is not None else None
        if recorded is not None and content_hash(payload) != recorded:
            raise PublishError(
                f"campaign {reference} payload does not match its recorded content digest "
                f"{recorded} — refusing it"
            )
        return load_campaign(payload)

    def _pull_payload(
        self, reference: str, kind: PluginKind, media_type: str
    ) -> tuple[bytes, PluginManifest]:
        # Pin the reference to a digest once, then read both halves of the artifact by that digest:
        # config and payload are then guaranteed to come from the *same* artifact, where resolving
        # the tag twice would let a re-tag between the two reads pair a verified config with some
        # other artifact's layer.
        try:
            digest = self._client.registry.resolve(reference).digest
            config = self._client.pull(digest, verify=True)
        except Exception as exc:
            raise PublishError(f"refusing pulled artifact {reference}: {exc}") from exc

        manifest = PluginManifest.model_validate_json(config)
        if manifest.kind is not kind:
            raise PublishError(
                f"{reference} is a {manifest.kind.value} artifact, not a {kind.value}"
            )

        # `pull_payload` selects the layers off the *verified* manifest and re-hashes each one's
        # bytes against the digest that manifest commits to before returning them (hub.md §2.3;
        # conventions.md §9). Filtered to the payload media type, in one call — every verified call
        # re-runs the supply-chain check.
        try:
            layers = self._client.pull_payload(digest, media_type=media_type, verify=True)
        except Exception as exc:
            raise PublishError(f"{reference} has no readable payload layer: {exc}") from exc
        if not layers:
            raise PublishError(f"{reference} carries no {media_type} payload layer")
        return layers[0].data, manifest
