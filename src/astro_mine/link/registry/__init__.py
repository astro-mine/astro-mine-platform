"""Link as a Core plugin: the ``comms_model`` manifest + the Hub publish/resolve path.

link.md §3 names this module — "``registry/`` — Core plugin manifest: declares Link as an
environment-model plugin" — and link.md §6 names its artifact: "contact-plan products are shareable
Hub artifacts (content-addressed)". Both live here, because they are one contract:

- :func:`build_comms_model_manifest` — the Core :class:`~astro_mine.core.registry.PluginManifest`
  (``kind=comms_model``) Link declares itself with; it version-negotiates against Core and is what
  Hub indexes.
- :func:`serialize_bundle` / :func:`bundle_digest` — the deterministic, content-addressed
  contact-plan bundle (the manifest's payload layer).
- :func:`publish_contact_plan` — signed, digest-resolvable publish to a local OCI-layout Hub
  registry (offline; no hosted Hub).
- :func:`from_bundle` — the ``astro_mine.providers`` → ``comms_model`` entry point: rebuilds a live
  :class:`~astro_mine.link.products.ConnectivitySampler` from pulled bytes, so Sim/Bench resolve a
  comms model **by content hash** without importing :mod:`astro_mine.link` (conventions.md §1.1).

Backlog: RM-P0-LINK-04 -- https://github.com/astro-mine/astro-mine-link/issues/25
"""

from __future__ import annotations

from astro_mine.link.registry._bundle import (
    BUNDLE_MEDIA_TYPE,
    COMMS_MODEL_ENTRY_POINT,
    PLAN_JSON_MEMBER,
    PLAN_WIRE_MEMBER,
    SCENARIO_MEMBER,
    bundle_digest,
    from_bundle,
    plan_from_bundle,
    scenario_from_bundle,
    serialize_bundle,
)
from astro_mine.link.registry._errors import LinkRegistryError
from astro_mine.link.registry._manifest import (
    COMMS_MODEL_ARTIFACT_KIND,
    CORE_INTERFACES,
    build_comms_model_manifest,
)
from astro_mine.link.registry._publish import default_scenario_descriptor, publish_contact_plan

__all__ = [
    "BUNDLE_MEDIA_TYPE",
    "COMMS_MODEL_ARTIFACT_KIND",
    "COMMS_MODEL_ENTRY_POINT",
    "CORE_INTERFACES",
    "PLAN_JSON_MEMBER",
    "PLAN_WIRE_MEMBER",
    "SCENARIO_MEMBER",
    "LinkRegistryError",
    "build_comms_model_manifest",
    "bundle_digest",
    "default_scenario_descriptor",
    "from_bundle",
    "plan_from_bundle",
    "publish_contact_plan",
    "scenario_from_bundle",
    "serialize_bundle",
]
