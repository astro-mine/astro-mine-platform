# SPDX-License-Identifier: Apache-2.0
"""Publish a served surrogate to a Hub registry (RM-P1-SURR-04; surrogate.md §4, §6).

Surrogate produces the ONNX tier; it is discovered and reused as a **content-addressed OCI
artifact** (surrogate.md §6). This builds the Core :class:`~astro_mine.core.registry.PluginManifest`
for the bundle, signs it (keyed ECDSA over ``provenance.digest``, the tier identity), and publishes
the signed manifest as the artifact config with the ONNX bundle as its one layer — the round trip
:func:`~astro_mine.surrogate.serve.load.resolve_and_load` verifies fail-closed.

Hub is imported lazily (the ``[publish]`` extra): the surrogate package never imports Hub at import
time, matching the astro-mine-prospect ``[publish]`` precedent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astro_mine.surrogate.enums import ServedBackend
from astro_mine.surrogate.manifest import build_surrogate_manifest
from astro_mine.surrogate.serve.bundle import ONNX_BUNDLE_MEDIA_TYPE, OnnxBundle

if TYPE_CHECKING:
    from astro_mine.hub.registry import Registry

__all__ = ["PublishedSurrogate", "publish_served_surrogate"]


@dataclass(frozen=True)
class PublishedSurrogate:
    """The result of a publish: the tag, the OCI manifest digest, and the pinned content hashes."""

    name: str
    version: str
    reference: str
    manifest_digest: str
    artifact_digest: str
    error_report_digest: str


def publish_served_surrogate(
    bundle: OnnxBundle,
    registry: Registry,
    *,
    name: str,
    version: str,
    private_key_pem: bytes,
    code_version: str | None = None,
    toolchain_version: str | None = None,
    train_dataset_hash: str | None = None,
    sampling_policy_hash: str | None = None,
) -> PublishedSurrogate:
    """Sign, attest, and publish ``bundle`` to ``registry`` under an immutable ``name:version``.

    Builds the surrogate manifest (its ``provenance.digest`` is the bundle's content hash), signs
    it with ``private_key_pem`` (``astro_mine.hub.supply_chain.sign_digest`` over that digest), and
    publishes the signed manifest as the artifact config with the ONNX bundle as the single layer.

    Publishing goes through :class:`~astro_mine.hub.client.HubClient`, so the artifact also carries
    the **cosign signature, SLSA provenance, and SBOM attestations** Hub verifies at admission and
    again fail-closed at pull. The manifest-embedded ``signature`` binds the bundle hash and is
    kept; it is not a substitute for those attestations, which are what a verifier actually reads.

    ``train_dataset_hash`` and ``sampling_policy_hash`` record where the tier's **domain** came
    from. They matter more than they look: a surrogate's trust region is *derived* from the configs
    its fixture swept, so the sampling policy is the declaration that decides the contract Sim
    admits the tier under — and until surrogate#17 nothing in the published artifact said which box
    that was. Pinning the policy by hash here makes the tier's domain traceable to the declaration
    that set it instead of merely inferable from the box's edges. Both are optional (an
    RM-P1-SURR-04 caller that has neither publishes exactly as before).
    """
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Blob
    from astro_mine.hub.supply_chain import sign_digest

    bundle_bytes = bundle.serialize()
    artifact_digest = bundle.content_hash()
    manifest = build_surrogate_manifest(
        name=name,
        version=version,
        report=bundle.error_report,
        artifact_digest=artifact_digest,
        served_backend=ServedBackend.ONNX,
        native_graph_fallback=False,
        code_version=code_version,
        toolchain_version=toolchain_version,
        train_dataset_hash=train_dataset_hash,
        sampling_policy_hash=sampling_policy_hash,
    )
    # build_surrogate_manifest sets provenance.digest to artifact_digest; sign that directly.
    signed = manifest.model_copy(
        update={"signature": sign_digest(artifact_digest, private_key_pem)}
    )
    # Publish through Hub's *client*, not the raw registry. `registry.publish` only stores bytes;
    # it is `HubClient.publish` that calls `attest(...)` to attach the cosign signature, SLSA
    # provenance, and SBOM as OCI referrers — the evidence `hub verify` and every fail-closed pull
    # actually read. Storing directly left surrogates signed only *inside* the manifest: signed to
    # a human, unsigned to the verifier, and unable to pass Hub's admission gate (hub#32).
    #
    # `inputs` is what makes the SLSA statement say something: the bundle's content hash plus the
    # training-set and sampling-policy hashes are precisely "which inputs produced this tier", and
    # the sampling policy is the declaration that fixes the trust region Sim admits it under.
    inputs = [artifact_digest, *(h for h in (train_dataset_hash, sampling_policy_hash) if h)]
    published = HubClient(registry).publish(
        name=name,
        version=version,
        kind="surrogate",
        manifest=signed,
        layers=[Blob(ONNX_BUNDLE_MEDIA_TYPE, bundle_bytes)],
        private_key_pem=private_key_pem,
        inputs=inputs,
    )
    return PublishedSurrogate(
        name=name,
        version=version,
        reference=published.reference,
        manifest_digest=published.digest,
        artifact_digest=artifact_digest,
        error_report_digest=bundle.error_report.content_hash(),
    )
