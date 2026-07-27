"""Materializing a Worlds bundle pulled from Hub by digest (RM-P1-STUDIO-06).

The embedded View renders terrain by fetching a Worlds bundle over **HTTP** —
``<GlobeScene world={{ manifestUrl }}>``. View's tiles proxy lives in the Phase-2 View Gateway,
which does not exist, so at design time something must serve the bundle. Studio does, and the
way it does so is fixed by studio.md §5: Studio "stores references, not the bytes", and pulls
sibling artifacts
"by content hash; never copied authoritatively".

So: resolve the world in Hub, **pull it by digest with the supply chain re-verified client-side**
(signature + SLSA + SBOM, fail-closed — hub.md §2 principle 3), and unpack it into a
**digest-keyed cache** that a static mount serves. The cache is a cache: its key is the artifact's
content address, it is disposable, and Studio's durable state records only the reference. Nothing is
copied authoritatively, and a tampered registry cannot make Studio serve bytes it did not verify.

Studio computes nothing here (studio.md §2 principle 1). It does not open a GeoTIFF, re-derive a
tileset, or georeference anything — it hands View the bytes Worlds published, unchanged.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astro_mine.core.registry import PluginKind, PluginManifest

from .._base import FrozenStudioModel

if TYPE_CHECKING:  # pragma: no cover - the [hub] extra is not in the base wheel
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Descriptor

__all__ = [
    "HubWorldMaterializer",
    "MaterializeError",
    "MaterializedWorld",
    "WorldMaterializer",
]

#: The manifest a Worlds bundle publishes; View fetches this file first.
WORLD_MANIFEST_NAME = "world.json"


class MaterializeError(RuntimeError):
    """A world could not be resolved, verified, or unpacked."""


class MaterializedWorld(FrozenStudioModel):
    """A verified world bundle on local disk, ready to be served to the embedded View.

    ``digest`` is the artifact's content address and the only durable identity Studio keeps;
    ``path`` is a cache location derived from it, not a source of truth.
    """

    reference: str
    digest: str
    world_id: str
    path: Path
    manifest_path: Path


@runtime_checkable
class WorldMaterializer(Protocol):
    """The ←Hub seam for terrain. A deployment binds it to Hub, a test to a temp registry."""

    def materialize(self, reference: str) -> MaterializedWorld: ...


def _extract_bundle(payload: bytes, dest: Path, reference: str) -> None:
    """Unpack the bundle tar into ``dest``.

    ``filter="data"`` is what makes this safe: it refuses absolute paths, ``..`` traversal, links,
    and device nodes. The layer's media type declares this is a tar, and reading a declared standard
    format is not reimplementing a producer — Studio never interprets what is inside it.

    A bundle that tries to escape its cache directory is a supply-chain event, not a parse error, so
    it surfaces as :class:`MaterializeError` rather than a bare ``tarfile`` exception.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            archive.extractall(dest, filter="data")
    except tarfile.FilterError as exc:
        raise MaterializeError(
            f"world {reference} contains an unsafe path and was not unpacked: {exc}"
        ) from exc
    except tarfile.TarError as exc:
        raise MaterializeError(f"world {reference} is not a readable bundle tar: {exc}") from exc


class HubWorldMaterializer:
    """Pull a Worlds bundle by digest, verify it, and unpack it into a content-addressed cache."""

    def __init__(self, client: HubClient, *, cache_dir: str | Path) -> None:
        self._client = client
        self._cache_dir = Path(cache_dir)

    def materialize(self, reference: str) -> MaterializedWorld:
        registry = self._client.registry
        try:
            digest = registry.resolve(reference).digest
        except Exception as exc:
            raise MaterializeError(f"no world {reference!r} in Hub: {exc}") from exc

        # Re-verify client-side before a single byte is trusted (verify twice, hub.md §2.3).
        try:
            config = self._client.pull(digest, verify=True)
        except Exception as exc:
            raise MaterializeError(f"refusing world {reference}: {exc}") from exc

        manifest = PluginManifest.model_validate_json(config)
        if manifest.kind is not PluginKind.WORLD_PROVIDER:
            raise MaterializeError(
                f"{reference} is a {manifest.kind.value} artifact, not a world_provider"
            )

        world_id = str(manifest.attributes.get("world_id", manifest.name))
        # The digest keys the cache, so a re-pull of the same world is free and two worlds never
        # collide. `:` is not portable in a path segment.
        dest = self._cache_dir / digest.replace(":", "-")
        manifest_path = dest / WORLD_MANIFEST_NAME
        if manifest_path.is_file():
            return MaterializedWorld(
                reference=reference,
                digest=digest,
                world_id=world_id,
                path=dest,
                manifest_path=manifest_path,
            )

        media_type = manifest.attributes.get("bundle_media_type")
        # Select off the verified manifest, then fetch exactly one layer: `payload_descriptors`
        # reads the layer list without pulling bytes, and `pull_layer` re-hashes the selected layer
        # against the digest that manifest commits to (hub.md §2.3; conventions.md §9). Selecting
        # before fetching is load-bearing — a world bundle is multi-GB, so pulling every layer's
        # bytes to find one would read the whole artifact.
        try:
            descriptors = self._client.payload_descriptors(digest, verify=True)
            layer = _bundle_descriptor(descriptors, media_type, reference)
            payload = self._client.pull_layer(digest, layer.digest, verify=True)
        except MaterializeError:
            raise
        except Exception as exc:
            raise MaterializeError(
                f"world {reference} has no readable bundle layer: {exc}"
            ) from exc

        _extract_bundle(payload, dest, reference)
        if not manifest_path.is_file():
            raise MaterializeError(
                f"world {reference} unpacked without a {WORLD_MANIFEST_NAME}; the embedded viewer "
                "has nothing to fetch"
            )

        return MaterializedWorld(
            reference=reference,
            digest=digest,
            world_id=world_id,
            path=dest,
            manifest_path=manifest_path,
        )


def _bundle_descriptor(
    layers: tuple[Descriptor, ...], media_type: object, reference: str
) -> Descriptor:
    """Pick the bundle layer out of the **verified** manifest's descriptors — no bytes fetched yet.

    Hub hands back ``()`` for an artifact that declares no layers, so the refusal is Studio's to
    make: a world_provider artifact carrying nothing to unpack is a publishing error, not an empty
    world.
    """
    if not layers:
        raise MaterializeError(f"world {reference} declares no layers")
    if isinstance(media_type, str):
        for layer in layers:
            if layer.media_type == media_type:
                return layer
        raise MaterializeError(f"world {reference} carries no {media_type} layer")
    # A bundle that names no `bundle_media_type` predates the attribute; it has exactly one layer.
    return layers[0]
