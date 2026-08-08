"""Publish a world bundle to Hub as a signed ``world`` OCI artifact (RM-P1-WORLDS-15).

:func:`build_world_bundle` (RM-P0-WORLDS-07) writes a content-addressed bundle **to disk only** —
no Core manifest, no Hub publish — so a consumer (Sim, Bench) cannot resolve the anchor's world by
content hash. This module closes that producer→Hub gap (worlds.md §3, §5; hub.md §3, §9):

- :func:`deterministic_bundle_tar` packs the **whole** bundle directory (terrain/regolith COGs and
  their ``manifest.json``, the PSR mask, thermal curves, STAC catalog, ``world.json``) into one
  **byte-stable** tar — sorted member names, ``mtime=0``, ``uid/gid=0``, no extended attrs — so the
  same bundle always yields the same layer digest ("two clean checkouts resolve the same world").
- :func:`build_world_manifest` emits the Core ``world_provider`` :class:`PluginManifest` whose
  ``provenance.digest`` is the bundle's ``world_hash`` — the manifest Hub indexes and Sim/Bench
  instantiate by digest through the Core registry (no ``astro_mine.worlds`` import by the consumer).
- :func:`publish_world_bundle` stores the artifact (config = the manifest, one payload layer) to a
  local OCI-layout ``Registry`` and signs+attests it (cosign ECDSA-P256 + SLSA + SBOM).

The reverse — rebuilding a live provider from the pulled bytes — is
:meth:`~astro_mine.worlds.provider.DemWorldProvider.from_bundle`, wired as the
``astro_mine.providers`` → ``world_provider`` entry point. ``astro-mine-hub`` is imported **lazily**
inside :func:`publish_world_bundle` so the base package stays dependency-light (offline, no hosted
Hub; the publish path needs the ``[hub]`` extra).

Backlog: RM-P1-WORLDS-15 — astro-mine-worlds#30
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from astro_mine.core.registry import PluginKind, PluginManifest, Provenance
from astro_mine.worlds.spec._bundle import WorldBundle

if TYPE_CHECKING:
    from astro_mine.hub.registry import PublishedArtifact, RegistryClient

__all__ = [
    "BUNDLE_LAYER_MEDIA_TYPE",
    "WORLD_ARTIFACT_KIND",
    "WORLD_PROVIDER_INTERFACE",
    "WORLD_PROVIDER_INTERFACE_VERSION",
    "build_world_manifest",
    "deterministic_bundle_tar",
    "extract_bundle_tar",
    "publish_world_bundle",
]

#: Hub OCI ``artifactType`` for a world bundle (Hub ``ARTIFACT_KINDS`` vocabulary; hub.md §3).
WORLD_ARTIFACT_KIND = "world"
#: The one payload layer's media type — a deterministic tar of the full bundle directory.
BUNDLE_LAYER_MEDIA_TYPE = "application/vnd.astro-mine.world.bundle.v1.tar"
#: The Core interface the world provider implements, and the version it is built against.
WORLD_PROVIDER_INTERFACE = "world_provider"
WORLD_PROVIDER_INTERFACE_VERSION = "0.1.0"


def deterministic_bundle_tar(bundle_dir: str | Path) -> bytes:
    """Pack a world-bundle directory into a **reproducible** uncompressed tar (its bytes).

    Every regular file under ``bundle_dir`` is added under its POSIX-relative name, in sorted
    order, with ``mtime=0``, ``mode=0o644``, ``uid=gid=0`` and empty owner names — so two builds of
    the same bundle produce byte-identical tars (hence an identical Hub layer digest). USTAR format
    keeps the headers free of the pax extended attributes that would reintroduce nondeterminism.
    """
    root = Path(bundle_dir)
    members = sorted(
        (path.relative_to(root).as_posix(), path) for path in root.rglob("*") if path.is_file()
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, path in members:
            data = path.read_bytes()
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


def extract_bundle_tar(data: bytes, dest: str | Path) -> None:
    """Unpack a :func:`deterministic_bundle_tar` payload into ``dest`` (path-traversal safe)."""
    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        tar.extractall(destination, filter="data")


def build_world_manifest(bundle: WorldBundle, *, name: str, version: str) -> PluginManifest:
    """The Core ``world_provider`` :class:`PluginManifest` for ``bundle`` (Hub's config blob).

    ``provenance.digest`` is the bundle's content-addressed ``world_hash`` (the identity Bench and
    Sim pin), ``source_content_hashes``/``input_hashes`` record the component layer hashes, and
    ``core_interfaces`` declares the ``world_provider`` interface the registry negotiates at load.
    """
    component_hashes = dict(sorted(bundle.component_hashes.items()))
    toolchain = bundle.manifest.get("toolchain", {})
    toolchain_version = toolchain.get("astro_mine_worlds") if isinstance(toolchain, dict) else None
    return PluginManifest(
        name=name,
        version=version,
        kind=PluginKind.WORLD_PROVIDER,
        core_interfaces={WORLD_PROVIDER_INTERFACE: WORLD_PROVIDER_INTERFACE_VERSION},
        license="Apache-2.0",
        description=bundle.spec.description or f"Astro-Mine world bundle: {bundle.world_id}",
        provenance=Provenance(
            digest=bundle.world_hash,
            input_hashes=sorted(component_hashes.values()),
            source_content_hashes=component_hashes,
            toolchain_version=str(toolchain_version) if toolchain_version else None,
        ),
        attributes={
            "world_id": bundle.world_id,
            "bundle_media_type": BUNDLE_LAYER_MEDIA_TYPE,
        },
    )


def publish_world_bundle(
    bundle: WorldBundle,
    # The `RegistryClient` *protocol*, not the local `Registry` class: `worlds publish` resolves
    # its target through `open_registry`, which returns the local OCI-layout store for a path and
    # the remote OCI Distribution client for a registry URL. Annotating the concrete class made
    # the remote path a type error (worlds#50).
    registry: RegistryClient,
    *,
    private_key_pem: bytes,
    name: str,
    version: str,
) -> PublishedArtifact:
    """Publish ``bundle`` to a local OCI-layout Hub ``registry`` as a signed ``world`` artifact.

    Builds the Core manifest (config) and the single deterministic-tar payload layer, then stores +
    signs + attests (cosign ECDSA-P256 signature, SLSA provenance, CycloneDX SBOM) via the Hub
    client. Fully offline: no hosted Hub, no Cloud. Returns the :class:`PublishedArtifact` whose
    ``digest`` a consumer resolves by. ``astro-mine-hub`` is imported here so the base package (and
    the :meth:`~astro_mine.worlds.provider.DemWorldProvider.from_bundle` load path) stays light.
    """
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Blob

    manifest = build_world_manifest(bundle, name=name, version=version)
    layer = Blob(media_type=BUNDLE_LAYER_MEDIA_TYPE, data=deterministic_bundle_tar(bundle.path))
    inputs = sorted(bundle.component_hashes.values())
    components = [
        {"name": f"world.{component}", "version": digest}
        for component, digest in sorted(bundle.component_hashes.items())
    ]
    client = HubClient(registry)
    return client.publish(
        name=name,
        version=version,
        kind=WORLD_ARTIFACT_KIND,
        manifest=manifest,
        layers=[layer],
        private_key_pem=private_key_pem,
        inputs=inputs,
        components=components,
    )
