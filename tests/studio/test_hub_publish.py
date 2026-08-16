"""RM-P1-STUDIO-06 — publishing a design/campaign to Hub, and pulling it back by digest.

These tests drive a **real** OCI-layout registry and the real supply chain (ECDSA P-256 cosign
signature + SLSA provenance + CycloneDX SBOM), not a stub. The acceptance criteria are about
trust: that a published artifact is content-addressed and signed, that it is re-pullable by
digest, and that a consumer who cannot verify it gets nothing. A mock demonstrates none of that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.objective import ObjectiveDocument
from astro_mine.core.registry import CapabilityTag, PluginKind, PluginManifest
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import ArtifactExistsError, IntegrityError, Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import generate_keypair
from astro_mine.studio.campaign import author_campaign, freeze_campaign
from astro_mine.studio.hub import (
    CAMPAIGN_LAYER_MEDIA_TYPE,
    ArtifactPublisher,
    HubArtifactPublisher,
    HubCapabilityResolver,
    PublishError,
    build_campaign_manifest,
)
from astro_mine.studio.models import (
    AssetSelection,
    CampaignPhase,
    DesignCandidate,
    EvaluatedCandidate,
)
from astro_mine.studio.orchestrate import (
    LOCAL_STAND_IN_EVALUATOR_ID,
    SiblingClients,
    evaluate_candidate,
)

ROVER_REF = "prospecting-rover:0.1.0"
ROVER_TAGS = [CapabilityTag.MOBILITY_WHEELED, CapabilityTag.PROSPECTING_NEUTRON]


@pytest.fixture
def keys() -> tuple[bytes, bytes]:
    return generate_keypair()


@pytest.fixture
def registry(tmp_path: Path, keys: tuple[bytes, bytes]) -> Registry:
    """A registry holding one signed Fleet asset — the swarm's capability tags come from it."""
    private_pem, _ = keys
    reg = Registry(tmp_path / "registry")
    asset = PluginManifest(
        name="prospecting-rover",
        version="0.1.0",
        kind=PluginKind.ASSET,
        capability_tags=ROVER_TAGS,
    )
    HubClient(reg).publish(
        name=asset.name,
        version=asset.version,
        kind="asset",
        manifest=asset,
        private_key_pem=private_pem,
    )
    return reg


@pytest.fixture
def publisher(registry: Registry, keys: tuple[bytes, bytes]) -> HubArtifactPublisher:
    private_pem, public_pem = keys
    client = HubClient(registry, trusted_public_key_pem=public_pem)
    return HubArtifactPublisher(
        client,
        capability_resolver=HubCapabilityResolver(registry),
        private_key_pem=private_pem,
    )


@pytest.fixture
def chosen(objective_doc: ObjectiveDocument, clients: SiblingClients) -> EvaluatedCandidate:
    candidate = DesignCandidate(id="cand-a", swarm=[AssetSelection(sadf_ref=ROVER_REF, count=4)])
    return evaluate_candidate(candidate, objective_doc, clients=clients, seed=7)


@pytest.fixture
def bundle(objective_doc: ObjectiveDocument, chosen: EvaluatedCandidate):
    campaign = author_campaign(
        objective_doc, chosen, name="Lunar ice", phases=[CampaignPhase(id="p", name="Prospect")]
    )
    return freeze_campaign(campaign)


def test_publisher_satisfies_the_seam(publisher: HubArtifactPublisher) -> None:
    assert isinstance(publisher, ArtifactPublisher)


class TestPublishCampaign:
    def test_publishes_a_signed_content_addressed_artifact(
        self, publisher: HubArtifactPublisher, registry: Registry, bundle
    ) -> None:
        ref = publisher.publish_campaign(bundle, name="lunar-ice-campaign", version="0.1.0")

        assert ref.reference == "lunar-ice-campaign:0.1.0"
        assert ref.digest.startswith("sha256:")
        # The payload's own content hash is the frozen bundle's identity, not the OCI manifest's.
        assert ref.content_digest == bundle.digest != ref.digest

        image = registry.read_manifest(ref.digest)
        assert image["artifactType"] == "application/vnd.astro-mine.campaign.v1"
        assert image["layers"][0]["mediaType"] == CAMPAIGN_LAYER_MEDIA_TYPE

        # Signature + SLSA + SBOM are attached as OCI referrers, and verification passes.
        assert registry.referrers(ref.digest)
        HubClient(registry).verify(ref.reference)

    def test_indexes_by_the_core_manifest_not_a_studio_schema(
        self, publisher: HubArtifactPublisher, registry: Registry, bundle
    ) -> None:
        ref = publisher.publish_campaign(bundle, name="lunar-ice-campaign", version="0.1.0")
        manifest = PluginManifest.model_validate_json(registry.read_config(ref.digest))

        assert manifest.kind is PluginKind.CAMPAIGN
        # A campaign implements no Core interface; nothing loads it as code (RFC-0008).
        assert manifest.core_interfaces == {}
        assert manifest.attributes["campaign_id"] == bundle.campaign.id
        assert manifest.attributes["objective_hash"] == bundle.campaign.objective_hash

    def test_carries_the_provenance_a_puller_reproduces_from(
        self, publisher: HubArtifactPublisher, registry: Registry, bundle
    ) -> None:
        ref = publisher.publish_campaign(bundle, name="lunar-ice-campaign", version="0.1.0")
        manifest = PluginManifest.model_validate_json(registry.read_config(ref.digest))

        provenance = manifest.provenance
        assert provenance is not None
        # conventions.md §5: inputs, code version, environment lockfile, seed.
        assert provenance.input_hashes == list(bundle.campaign.provenance.input_hashes)
        assert provenance.code_version and provenance.toolchain_version and provenance.env_lockfile
        assert provenance.seed == bundle.campaign.chosen.seed
        assert provenance.digest == bundle.digest

    def test_inherits_capability_tags_from_the_swarm(
        self, publisher: HubArtifactPublisher, registry: Registry, bundle
    ) -> None:
        """studio.md §9: Studio honors, and never redefines, the export-control partition."""
        ref = publisher.publish_campaign(bundle, name="lunar-ice-campaign", version="0.1.0")
        manifest = PluginManifest.model_validate_json(registry.read_config(ref.digest))
        assert set(manifest.capability_tags) == set(ROVER_TAGS)

    def test_refuses_to_publish_a_swarm_whose_assets_it_cannot_resolve(
        self, publisher: HubArtifactPublisher, objective_doc, clients
    ) -> None:
        """Understating what a swarm can do would route it past the capability gate."""
        candidate = DesignCandidate(
            id="ghost", swarm=[AssetSelection(sadf_ref="sha256:not-in-hub", count=1)]
        )
        evaluated = evaluate_candidate(candidate, objective_doc, clients=clients, seed=1)
        campaign = author_campaign(
            objective_doc, evaluated, name="Ghost", phases=[CampaignPhase(id="p", name="P")]
        )
        with pytest.raises(PublishError, match="capability tags are unknown"):
            publisher.publish_campaign(freeze_campaign(campaign), name="ghost", version="0.1.0")

    def test_a_version_is_published_once(self, publisher: HubArtifactPublisher, bundle) -> None:
        """Digests are immutable: re-publishing a version is rejected, not silently clobbered."""
        publisher.publish_campaign(bundle, name="lunar-ice-campaign", version="0.1.0")
        with pytest.raises(PublishError) as excinfo:
            publisher.publish_campaign(bundle, name="lunar-ice-campaign", version="0.1.0")
        assert isinstance(excinfo.value.__cause__, ArtifactExistsError)


class TestPullCampaign:
    def test_round_trips_by_reference_and_by_digest(
        self, publisher: HubArtifactPublisher, bundle
    ) -> None:
        ref = publisher.publish_campaign(bundle, name="lunar-ice-campaign", version="0.1.0")

        for handle in (ref.reference, ref.digest):
            pulled = publisher.pull_campaign(handle)
            # The same Core-defined artifact Studio produced -- no translation (studio.md §9).
            assert pulled == bundle.campaign

    def test_a_tampered_payload_fails_closed(
        self,
        publisher: HubArtifactPublisher,
        registry: Registry,
        bundle,
        keys: tuple[bytes, bytes],
    ) -> None:
        """The pull is refused — and payload *retrieval* refuses the bytes on its own.

        The class-level refusal comes from the supply-chain re-verification, which would still fire
        if the layer were read with the unchecked `registry.pull_blob()`. The guarantee this test
        exists for is the one inside `pull_payload`: the layer's bytes are re-hashed against the
        content address the verified manifest commits to, so they are refused even with the
        supply-chain check switched off (hub.md §2.3; conventions.md §9).
        """
        _, public_pem = keys
        ref = publisher.publish_campaign(bundle, name="lunar-ice-campaign", version="0.1.0")
        layer_digest = registry.read_manifest(ref.digest)["layers"][0]["digest"]
        blob_path(registry.path, layer_digest).write_bytes(
            b'{"id": "not-the-campaign-you-published"}'
        )

        with pytest.raises(PublishError):
            publisher.pull_campaign(ref.reference)

        client = HubClient(registry, trusted_public_key_pem=public_pem)
        with pytest.raises(IntegrityError):
            client.pull_payload(ref.reference, media_type=CAMPAIGN_LAYER_MEDIA_TYPE, verify=False)

    def test_an_untrusted_signer_fails_closed(
        self, registry: Registry, bundle, keys: tuple[bytes, bytes]
    ) -> None:
        """A signature the client does not trust never yields a usable artifact.

        This used to fail at *pull*: the campaign was published, then refused on the way back.
        Since astro-mine-hub#32 the check runs at **admission** too, pinned to the client's
        trusted key — so an artifact the publisher itself would not accept is never indexed in
        the first place. Failing earlier is strictly better; the property is unchanged.
        """
        private_pem, _ = keys
        publisher = HubArtifactPublisher(
            HubClient(registry, trusted_public_key_pem=generate_keypair()[1]),
            capability_resolver=HubCapabilityResolver(registry),
            private_key_pem=private_pem,
        )
        # Signing with our key while the client pins a *different* trusted key.
        with pytest.raises(PublishError, match="signing key is not the trusted key"):
            publisher.publish_campaign(bundle, name="lunar-ice-campaign", version="0.1.0")

    def test_refuses_an_artifact_of_the_wrong_kind(
        self, publisher: HubArtifactPublisher, registry: Registry, keys
    ) -> None:
        with pytest.raises(PublishError, match="not a campaign"):
            publisher.pull_campaign(ROVER_REF)

    def test_refuses_a_payload_that_is_not_the_one_the_manifest_recorded(
        self, publisher: HubArtifactPublisher, registry: Registry, bundle, keys
    ) -> None:
        """Defense in depth, beyond the signature and the blob hashes.

        A manifest is signed, and every blob hashes to its content address — yet the manifest
        could still *point at a different, equally well-formed* payload than the one whose digest
        it claims under ``provenance.digest``. Studio re-derives that digest from the layer bytes
        and refuses a mismatch, so the campaign a puller loads is the one whose identity was
        published.
        """
        from astro_mine.hub.registry import Blob
        from astro_mine.studio.hub.publish import build_campaign_manifest

        private_pem, public_pem = keys
        manifest = build_campaign_manifest(bundle, name="swapped", version="0.1.0")
        other_payload = b'{"id":"a-different-campaign"}'
        assert manifest.provenance is not None
        assert manifest.provenance.digest == bundle.digest  # claims our campaign...

        HubClient(registry, trusted_public_key_pem=public_pem).publish(
            name="swapped",
            version="0.1.0",
            kind="campaign",
            manifest=manifest,
            layers=[Blob(CAMPAIGN_LAYER_MEDIA_TYPE, other_payload)],  # ...but carries another
            private_key_pem=private_pem,
        )

        with pytest.raises(PublishError, match="does not match its recorded content digest"):
            publisher.pull_campaign("swapped:0.1.0")


class TestPublishTradeStudy:
    def test_publishes_a_design_artifact(
        self, publisher: HubArtifactPublisher, registry: Registry, objective_doc, chosen
    ) -> None:
        from astro_mine.studio.models import TradeStudy
        from astro_mine.studio.provenance import capture_provenance

        study = TradeStudy(
            id="ts-1",
            objective_hash=chosen.score.objective_hash,
            backend="random",
            evaluator=LOCAL_STAND_IN_EVALUATOR_ID,
            seeds=[7],
            evaluated=[chosen],
            pareto_front=[chosen.candidate.id],
            provenance=capture_provenance(input_hashes=[chosen.digest()], seed=7),
        )
        ref = publisher.publish_trade_study(study, name="lunar-ice-study", version="0.1.0")

        assert ref.kind == "design"
        image = registry.read_manifest(ref.digest)
        assert image["artifactType"] == "application/vnd.astro-mine.design.v1"

        manifest = PluginManifest.model_validate_json(registry.read_config(ref.digest))
        assert manifest.kind is PluginKind.DESIGN
        assert manifest.attributes["study_id"] == "ts-1"
        assert manifest.attributes["pareto_front_size"] == 1
        assert set(manifest.capability_tags) == set(ROVER_TAGS)


def test_a_campaign_artifact_with_no_payload_layer_is_refused(
    publisher: HubArtifactPublisher, registry: Registry, bundle, keys: tuple[bytes, bytes]
) -> None:
    """A well-signed manifest of the right kind, carrying nothing to load."""
    private_pem, _ = keys
    manifest = build_campaign_manifest(bundle, name="hollow", version="0.1.0")
    HubClient(registry).publish(
        name="hollow",
        version="0.1.0",
        kind="campaign",
        manifest=manifest,
        private_key_pem=private_pem,
    )
    with pytest.raises(PublishError, match=r"carries no .* payload layer"):
        publisher.pull_campaign("hollow:0.1.0")
