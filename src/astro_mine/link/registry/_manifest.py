"""Link's Core ``comms_model`` plugin manifest — the comms environment declaring itself.

Per conventions.md §1.3 ("plugins over patches") and link.md §3, **Link is itself a Core
environment-model plugin**: :data:`~astro_mine.core.registry.PluginKind.COMMS_MODEL` is the
extension point Core reserves for it, parallel to Worlds' ``world_provider`` and Prospect's
``resource_field_backend``. This module builds that manifest — the document Hub indexes, the
registry version-negotiates, and a consumer resolves a published :class:`ContactPlan` through.

Link **consumes** Core's manifest schema; it does not invent one (hub.md §2 principle 2). Two
consequences worth stating plainly, because both are easy to get wrong:

- ``core_interfaces`` names the *Core interfaces the plugin is built against*, not the plugin kind.
  A comms model produces the Core **message catalog** (``ContactPlan`` / ``CommsObservationMask``)
  and delivers those masks through the Core **Environment API**, so it declares ``messages``
  and ``env`` - the two interfaces Core's ``CORE_INTERFACE_VERSIONS`` actually carries. There is
  no ``comms`` Core interface, and the registry rejects an interface Core does not provide.
- The manifest carries **no vcs-derived toolchain version**. The anchor artifact's digest is a
  Bench pin (bench#28), so it must not drift when Link's commit count does: everything hashed into
  the manifest is either pinned scenario data or a content address of a pinned input
  (conventions.md §1.5; hub.md §2.1). Provenance still records the plan's own digest and the
  kernel/terrain/config input hashes — the identity that actually determines the plan.

Backlog: RM-P0-LINK-04 -- https://github.com/astro-mine/astro-mine-link/issues/25
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.messages import ContactPlan
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.registry import PluginKind, PluginManifest, Provenance
from astro_mine.link.cache import plan_digest
from astro_mine.link.registry._bundle import BUNDLE_MEDIA_TYPE, COMMS_MODEL_ENTRY_POINT
from astro_mine.link.registry._errors import LinkRegistryError

__all__ = [
    "COMMS_MODEL_ARTIFACT_KIND",
    "CORE_INTERFACES",
    "build_comms_model_manifest",
    "normalize_digest",
]

#: The Hub OCI ``artifactType`` a comms model is stored under. Hub's ``ARTIFACT_KINDS`` vocabulary
#: is closed and grows only when Core's does (hub.md §2 principle 2), and it carries no
#: ``comms_model`` member — so, exactly like Prospect's ``resource_field_backend``, the artifact is
#: the generic Hub ``plugin`` kind while the **Core manifest** carries ``PluginKind.COMMS_MODEL``.
COMMS_MODEL_ARTIFACT_KIND = "plugin"

#: The Core interfaces a Link comms model is built against: it emits the Core message catalog
#: (``ContactPlan``, ``CommsObservationMask``) and feeds masks through the Core Environment API
#: (link.md §3, §6; ``LUNAR-TR-003``).
CORE_INTERFACES: Mapping[str, str] = {"messages": "0.1.0", "env": "0.1.0"}

#: The Core message types a comms model consumes / produces (declarative, for compatibility checks).
_INPUTS = ["ContactPlan"]
_OUTPUTS = ["ContactPlan", "CommsObservationMask"]


def normalize_digest(digest: str) -> str:
    """Normalize a content address to the platform ``sha256:<hex>`` form (conventions.md §5).

    Link's own cache digests (:mod:`astro_mine.link.cache`) are bare hex, while Core's
    ``Provenance`` carriers and Hub's supply chain speak the prefixed form. Accept either and emit
    the prefixed one, so a caller can hand a :class:`~astro_mine.link.cache.CacheKey` field straight
    to :func:`~astro_mine.link.registry.publish_contact_plan` without reformatting it.
    """
    algorithm, _, hexpart = digest.partition(":")
    if hexpart:
        if algorithm != "sha256":
            raise LinkRegistryError(f"unsupported digest algorithm {algorithm!r} in {digest!r}")
        return digest
    if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        raise LinkRegistryError(f"not a sha256 content address: {digest!r}")
    return f"sha256:{digest}"


def build_comms_model_manifest(
    plan: ContactPlan,
    *,
    name: str,
    version: str,
    bundle_sha256: str,
    scenario_id: str,
    description: str | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> PluginManifest:
    """The ``kind=comms_model`` manifest for ``plan`` (unsigned; the publisher attaches signatures).

    ``provenance.digest`` is the plan's own content address
    (:func:`~astro_mine.link.cache.plan_digest` over Core's byte-stable wire form) — the identity a
    Bench scenario pins and a Link cache key agrees with. ``bundle_sha256`` is the ``sha256:<hex>``
    address of the serialized bundle layer, folded into ``attributes`` so a catalog can cross-check
    the payload against the manifest. ``input_hashes`` maps a pinned-input name (``kernels``,
    ``terrain``, ``nodes``, ``epoch``, ``config``) to its digest — the LINK-05 cache key,
    recorded so a consumer can tell *why* two plans differ (link.md §5, §9).

    The manifest declares **no capability tags**: a contact plan predicted from public ephemerides
    and parametric antenna models is open commons. The gated
    :data:`~astro_mine.link.products.LIVE_MISSION_LINK_PREDICTION` tag never appears here, because a
    live-mission schedule is never published (RFC-0003; link.md §9).
    """
    sources = {key: normalize_digest(value) for key, value in sorted((input_hashes or {}).items())}
    nodes = {role: _node_ids(plan, role) for role in (NodeRole.SPACE, NodeRole.GROUND)}
    return PluginManifest(
        name=name,
        version=version,
        kind=PluginKind.COMMS_MODEL,
        core_interfaces=dict(CORE_INTERFACES),
        inputs=list(_INPUTS),
        outputs=list(_OUTPUTS),
        license="Apache-2.0",
        description=description
        or (
            f"Astro-Mine comms model {scenario_id!r}: a content-addressed ContactPlan "
            f"({len(plan.nodes)} nodes, {len(plan.intervals)} contact intervals) that reopens as a "
            "ConnectivitySampler and emits per-tick CommsObservationMasks."
        ),
        provenance=Provenance(
            digest=f"sha256:{plan_digest(plan)}",
            input_hashes=sorted(sources.values()),
            source_content_hashes=sources,
        ),
        attributes={
            "scenario_id": scenario_id,
            "bundle_media_type": BUNDLE_MEDIA_TYPE,
            "bundle_sha256": bundle_sha256,
            "provider_entry_point": COMMS_MODEL_ENTRY_POINT,
            "space_nodes": nodes[NodeRole.SPACE],
            "ground_nodes": nodes[NodeRole.GROUND],
            "n_intervals": len(plan.intervals),
        },
    )


def _node_ids(plan: ContactPlan, role: NodeRole) -> list[str]:
    """The plan's node ids with ``role``, in declaration order — an open discovery facet."""
    return [node.id for node in plan.nodes if node.role == role]
