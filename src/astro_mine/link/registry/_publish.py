"""Publish a contact plan to a Hub registry as a signed, content-addressed ``comms_model``.

The producer→Hub half of link.md §6 ("contact-plan products are shareable Hub artifacts,
content-addressed") and of hub.md §3, §9: serialize the plan into its deterministic bundle, build
the Core ``comms_model`` :class:`~astro_mine.core.registry.PluginManifest`, and store + sign +
attest it in a **local OCI-layout registry** through the ``astro-mine-hub`` client — the tier-1
offline path, no hosted Hub, no Cloud (hub.md principle 7; ``LUNAR-TR-004``).

``astro-mine-hub`` is a **publish-time** dependency (the ``hub`` extra), imported lazily here so the
library path — computing windows, budgets, plans, masks — stays offline and dependency-light, and so
resolving a plan through :func:`~astro_mine.link.registry.from_bundle` never needs the Hub client
either.

Backlog: RM-P0-LINK-04 -- https://github.com/astro-mine/astro-mine-link/issues/25
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from astro_mine.core.messages import ContactPlan
from astro_mine.link.registry._bundle import BUNDLE_MEDIA_TYPE, bundle_digest, serialize_bundle
from astro_mine.link.registry._errors import LinkRegistryError
from astro_mine.link.registry._manifest import (
    COMMS_MODEL_ARTIFACT_KIND,
    build_comms_model_manifest,
    normalize_digest,
)

if TYPE_CHECKING:
    from astro_mine.hub.registry import PublishedArtifact, RegistryClient

__all__ = ["publish_contact_plan"]


def publish_contact_plan(
    plan: ContactPlan,
    *,
    registry: RegistryClient,
    name: str,
    version: str,
    scenario_id: str,
    scenario: Mapping[str, Any] | None = None,
    input_hashes: Mapping[str, str] | None = None,
    description: str | None = None,
    private_key_pem: bytes,
    namespace: str = "open",
    publisher: str = "local",
) -> PublishedArtifact:
    """Serialize, manifest, sign, and publish ``plan`` to the OCI ``registry`` the caller supplies.

    The registry is **injected**, not opened from a path here: it has two implementations — the
    local OCI-layout store and the remote OCI Distribution client — and choosing between them is
    the caller's decision, not Link's (conventions.md §3.3). The ``HubClient`` wrapped around it is
    not injected, because there is exactly one signer and it is Hub's.

    Returns the :class:`~astro_mine.hub.registry.PublishedArtifact` — its immutable
    ``name:version`` reference and its **content digest** (the OCI image-manifest digest a consumer,
    and a Bench ``ScenarioSpec``, pins the comms model by). The artifact is signed with
    ``private_key_pem`` and gets its cosign signature / SLSA provenance / SBOM attestations,
    verified at admission and again fail-closed at pull.

    The key is **required**: ``hub.md`` §9 tiers artifacts as *open* (self-published, **signed**,
    unreviewed), *curated*, and *verified*, so there is no tier for unsigned content and Hub's
    admission gate refuses it (astro-mine-hub#32). Keyed ECDSA signing is offline and accountless
    (``astro-mine hub keygen``), so the local tier is unaffected (CX-LOCAL).

    ``scenario`` is the open descriptor written into the bundle (node ids, epoch window, pinned
    input hashes) — it defaults to the plan's own node/epoch summary. ``input_hashes`` is the
    LINK-05 pinned-input digest map (kernels/terrain/nodes/epoch/config) recorded in the manifest's
    provenance.

    Deterministic: the same plan + scenario descriptor yields byte-identical bundle bytes, hence
    the same layer digest and the same artifact digest, on any machine or checkout (conventions.md
    §1.5). Signatures ride as OCI *referrers*, so signing does not perturb that digest.
    """
    try:
        from astro_mine.hub.client import HubClient
        from astro_mine.hub.registry import Blob
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise LinkRegistryError(
            "publishing a contact plan needs the Hub client; install astro-mine-platform[link-hub] "
            "(astro-mine-hub is git-pinned in pyproject.toml [tool.uv.sources])"
        ) from exc

    sources = {key: normalize_digest(value) for key, value in sorted((input_hashes or {}).items())}
    descriptor = dict(scenario) if scenario is not None else default_scenario_descriptor(plan)
    descriptor.setdefault("scenario_id", scenario_id)
    if sources:
        descriptor.setdefault("input_hashes", sources)

    bundle = serialize_bundle(plan, scenario=descriptor)
    digest = bundle_digest(bundle)
    manifest = build_comms_model_manifest(
        plan,
        name=name,
        version=version,
        bundle_sha256=digest,
        scenario_id=scenario_id,
        description=description,
        input_hashes=sources,
    )
    client = HubClient(registry)
    return client.publish(
        name=name,
        version=version,
        # The Hub artifact category; the Core manifest kind is COMMS_MODEL (see _manifest.py).
        kind=COMMS_MODEL_ARTIFACT_KIND,
        manifest=manifest,
        layers=[Blob(BUNDLE_MEDIA_TYPE, bundle)],
        private_key_pem=private_key_pem,
        namespace=namespace,
        publisher=publisher,
        inputs=sorted(sources.values()),
    )


def default_scenario_descriptor(plan: ContactPlan) -> dict[str, Any]:
    """The self-describing summary written into a bundle when the caller supplies none.

    Node ids by role, the plan's validity window, and the interval count — enough for a consumer to
    tell *what comms graph* it pulled without decoding the plan.
    """
    return {
        "nodes": [
            {"id": node.id, "role": node.role.value, "kind": node.kind} for node in plan.nodes
        ],
        "epoch_start_tdb_s": plan.epoch_start_tdb_s,
        "epoch_end_tdb_s": plan.epoch_end_tdb_s,
        "n_intervals": len(plan.intervals),
    }
