# SPDX-License-Identifier: Apache-2.0
"""The content-addressed contact-plan bundle: the wire form of a published comms model.

The payload half of the Link→Hub publish path (link.md §6 — "contact-plan products are shareable
Hub artifacts (content-addressed)"). A bundle is a **byte-stable** USTAR tar carrying:

- ``contact_plan.pb`` — the Core :class:`~astro_mine.core.messages.ContactPlan` in its byte-stable
  Protobuf wire form (``contact_plan_to_wire``). This is the artifact's content identity: its
  SHA-256 is exactly :func:`~astro_mine.link.cache.plan_digest`, the digest LINK-05 already
  content-addresses a plan by, so a Bench scenario and a Link cache key agree on what "this plan"
  means (link.md §5; conventions.md §1.5).
- ``contact_plan.json`` — the same plan as canonical JSON, so a non-Python consumer (or a human)
  reads the artifact without a protobuf toolchain.
- ``comms_scenario.json`` — the scenario descriptor: the node set, the epoch window, and the pinned
  input hashes (kernels / terrain / config) the plan was derived from.

The tar is canonical — sorted member names, ``mtime``/``uid``/``gid`` zeroed, USTAR format — so the
same plan yields byte-identical bundle bytes on any machine or checkout, hence a stable OCI layer
digest and a stable artifact digest ("two clean checkouts resolve the identical comms model").

:func:`from_bundle` is the **entry-point factory** (``astro_mine.providers`` → ``comms_model``): a
consumer resolves the plan through Hub + the Core manifest and rebuilds a live
:class:`~astro_mine.link.products.ConnectivitySampler` from the pulled bytes — **without importing**
:mod:`astro_mine.link` by name, exactly as Sim rebuilds a Worlds ``WorldProvider`` or a Prospect
``ResourceField`` (conventions.md §1.1; hub.md §3).

Backlog: RM-P0-LINK-04 -- astro-mine-link#25
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Mapping
from typing import Any

from astro_mine.core.hashing import canonical_json, content_hash
from astro_mine.core.messages import ContactPlan, contact_plan_from_wire, contact_plan_to_wire
from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.link.products import ConnectivitySampler
from astro_mine.link.registry._errors import LinkRegistryError

__all__ = [
    "BUNDLE_MEDIA_TYPE",
    "COMMS_MODEL_ENTRY_POINT",
    "PLAN_JSON_MEMBER",
    "PLAN_WIRE_MEMBER",
    "SCENARIO_MEMBER",
    "bundle_digest",
    "from_bundle",
    "plan_from_bundle",
    "scenario_from_bundle",
    "serialize_bundle",
]

#: The single OCI layer's media type — the contact-plan bundle tar (hub.md §3). A consumer keys the
#: pulled ``layers`` mapping by media type and looks this one up.
BUNDLE_MEDIA_TYPE = "application/vnd.astro-mine.contact-plan.bundle.v1.tar"

#: The ``astro_mine.providers`` entry-point name :func:`from_bundle` registers under — the Core
#: :class:`~astro_mine.core.registry.PluginKind` value, so a consumer's resolver keys the factory by
#: the manifest ``kind`` it just pulled (Sim's ``PROVIDER_ENTRY_POINT_GROUP`` convention).
COMMS_MODEL_ENTRY_POINT = "comms_model"

PLAN_WIRE_MEMBER = "contact_plan.pb"
PLAN_JSON_MEMBER = "contact_plan.json"
SCENARIO_MEMBER = "comms_scenario.json"


def serialize_bundle(plan: ContactPlan, *, scenario: Mapping[str, Any]) -> bytes:
    """The deterministic bundle bytes for ``plan`` + its ``scenario`` descriptor.

    ``scenario`` is the open, JSON-serializable descriptor of what produced the plan (scenario id,
    node ids, epoch window, pinned kernel/terrain/config hashes). It rides in the bundle rather than
    the manifest so the artifact is **self-describing**: pulled bytes alone say which scenario,
    which nodes, and which pinned inputs the plan came from (link.md §5).
    """
    members = {
        PLAN_WIRE_MEMBER: contact_plan_to_wire(plan),
        PLAN_JSON_MEMBER: canonical_json(plan.model_dump(mode="json", exclude_none=True)),
        SCENARIO_MEMBER: canonical_json(dict(scenario)),
    }
    return _deterministic_tar(members)


def bundle_digest(data: bytes) -> str:
    """The ``sha256:<hex>`` content address of a bundle's bytes (matches the OCI layer digest)."""
    return content_hash(data)


def plan_from_bundle(data: bytes) -> ContactPlan:
    """Rebuild the Core :class:`~astro_mine.core.messages.ContactPlan` from bundle bytes.

    Reads the byte-stable Protobuf member (the plan's content identity), never the JSON convenience
    projection — a bundle whose two projections disagree must not silently resolve to the JSON one.
    """
    return contact_plan_from_wire(_member(data, PLAN_WIRE_MEMBER))


def scenario_from_bundle(data: bytes) -> dict[str, Any]:
    """The scenario descriptor recorded in a bundle (node ids, epoch window, input hashes)."""
    decoded: Any = json.loads(_member(data, SCENARIO_MEMBER))
    if not isinstance(decoded, dict):
        raise LinkRegistryError(f"bundle member {SCENARIO_MEMBER!r} is not a JSON object")
    return decoded


def from_bundle(manifest: PluginManifest, layers: Mapping[str, bytes]) -> ConnectivitySampler:
    """Rebuild a live :class:`~astro_mine.link.products.ConnectivitySampler` from pulled Hub layers.

    The ``astro_mine.providers`` → ``comms_model`` entry point: a consumer (Sim, Bench) that pulled
    an artifact by content hash hands over the Core manifest plus the ``mediaType -> bytes`` layer
    map and gets back the comms model that answers ``comms_mask(agent, epoch)`` — Sim's
    ``ConnectivitySource`` contract — without importing :mod:`astro_mine.link`.

    Fails loudly (:class:`LinkRegistryError`) on a manifest that is not a ``comms_model`` or a layer
    set with no contact-plan bundle: a resolver must never fall back to an empty (fully connected,
    or fully denied) comms model.
    """
    if manifest.kind != PluginKind.COMMS_MODEL:
        raise LinkRegistryError(
            f"manifest {manifest.name!r} is kind {manifest.kind.value!r}, not a "
            f"{PluginKind.COMMS_MODEL.value!r} plugin; it cannot rebuild a comms model"
        )
    data = layers.get(BUNDLE_MEDIA_TYPE)
    if data is None:
        raise LinkRegistryError(
            f"artifact {manifest.name!r} carries no {BUNDLE_MEDIA_TYPE!r} layer; "
            f"a comms_model artifact must ship its contact-plan bundle"
        )
    return ConnectivitySampler(plan_from_bundle(data))


def _member(data: bytes, name: str) -> bytes:
    """One member's bytes from a bundle tar; a missing member fails loudly."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            extracted = tar.extractfile(name)
            if extracted is None:
                raise LinkRegistryError(f"contact-plan bundle has no {name!r} member")
            return extracted.read()
    except tarfile.TarError as exc:
        raise LinkRegistryError(f"contact-plan bundle is not a readable tar: {exc}") from exc


def _deterministic_tar(members: Mapping[str, bytes]) -> bytes:
    """A reproducible USTAR tar of ``members``: sorted names, zeroed mtime/uid/gid/owner.

    USTAR (not the default pax) keeps the headers free of the extended attributes that would
    reintroduce nondeterminism, so two builds of the same plan produce byte-identical tars — hence
    an identical Hub layer digest (hub.md §2.1; conventions.md §1.5).
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()
