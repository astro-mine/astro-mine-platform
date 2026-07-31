"""The ``astro-mine-hub`` client SDK — resolve / verify / pull / cache / publish (RM-P1-HUB-06).

The tier-1 path that MUST always work with **no hosted Hub** (hub.md principle 7, §7): a
:class:`HubClient` resolves, verifies, and pulls artifacts against any
:class:`~astro_mine.hub.registry.RegistryClient` — a local OCI-layout
:class:`~astro_mine.hub.registry.Registry` (offline) or a
:class:`~astro_mine.hub.registry.RemoteRegistry` (ghcr.io / Zot / Harbor over the OCI Distribution
Spec) — through one interface. Its defining guarantee is **re-verify at pull, before Core loads the
plugin** (hub.md §2.3; ``LUNAR-SR-002``): :meth:`pull` re-runs the full supply-chain check
client-side, so a compromised registry cannot make a consumer accept tampered bytes — a tampered
pull **fails closed**. Pulls are served from a **content-addressed cache** keyed on digest. Bench
resolves a submission **by digest** through it.

**Config *and* payload.** An artifact's OCI config blob is its Core plugin manifest; its **layers**
carry what the artifact exists to distribute (the ONNX policy, the SADF bundle, the Zarr/COG world —
hub.md §3, §5). :meth:`pull`/:meth:`load` return the manifest; :meth:`pull_payload`,
:meth:`pull_layer`, and :meth:`materialize` return the **payload layers**, each layer's bytes
re-hashed against its content address *inside* the client before it is returned or written
(conventions.md §5, §9). A consumer never has to reach past the client to
``registry.pull_blob(...)`` for raw, unverified bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from astro_mine.core.registry import PluginManifest, PluginRegistry
from astro_mine.hub._content import content_hash
from astro_mine.hub.index import Catalog, InMemoryCatalog, ingest
from astro_mine.hub.registry import (
    ArtifactNotFound,
    Blob,
    Descriptor,
    IntegrityError,
    PublishedArtifact,
    RegistryClient,
)
from astro_mine.hub.resolve import Resolution, ResolutionRequest, resolve
from astro_mine.hub.supply_chain import DEFAULT_REQUIRED, admit, attest
from astro_mine.hub.supply_chain import verify as supply_verify

__all__ = ["HubClient", "PayloadLayer", "catalog_from_registry"]


@dataclass(frozen=True)
class PayloadLayer:
    """One **verified** payload layer: its OCI descriptor and its bytes."""

    descriptor: Descriptor
    data: bytes

    @property
    def media_type(self) -> str:
        return self.descriptor.media_type

    @property
    def digest(self) -> str:
        """The layer's content address — the bytes in :attr:`data` hash to exactly this."""
        return self.descriptor.digest

    @property
    def size(self) -> int:
        return self.descriptor.size


def catalog_from_registry(
    registry: RegistryClient, *, catalog: Catalog | None = None, publisher: str = "local"
) -> Catalog:
    """Rebuild a catalog from a registry's stored manifests (stateless CLI / cold start)."""
    result = catalog if catalog is not None else InMemoryCatalog()
    for reference in registry.references():
        digest = registry.resolve(reference).digest
        manifest = PluginManifest.model_validate_json(registry.read_config(digest))
        ingest(result, manifest, digest=digest, publisher=publisher)
    return result


class HubClient:
    """Resolve/verify/pull/cache/publish against a registry — no hosted Hub required."""

    def __init__(
        self,
        registry: RegistryClient,
        *,
        catalog: Catalog | None = None,
        cache_dir: str | Path | None = None,
        trusted_public_key_pem: bytes | None = None,
    ) -> None:
        self.registry = registry
        self.catalog = catalog if catalog is not None else InMemoryCatalog()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.trusted_public_key_pem = trusted_public_key_pem

    def publish(
        self,
        *,
        name: str,
        version: str,
        kind: str,
        manifest: PluginManifest,
        layers: Sequence[Blob] = (),
        private_key_pem: bytes,
        namespace: str = "open",
        publisher: str = "local",
        inputs: Sequence[str] = (),
        components: Sequence[Mapping[str, str]] = (),
    ) -> PublishedArtifact:
        """Store the artifact, sign + attest it, verify at admission (fail-closed), then index it.

        ``private_key_pem`` is **required**: ``hub.md`` §9 defines no namespace tier for unsigned
        content, so admitting an unsigned artifact would index something the trust model cannot
        describe. Keyed ECDSA signing is offline and accountless (``astro-mine hub keygen`` mints a
        key), so the local tier keeps working (CX-LOCAL).

        Nothing is indexed unless admission passes — a half-admitted artifact (bytes present,
        evidence absent, entry queryable) is the state this gate exists to prevent.
        """
        artifact = self.registry.publish(
            name=name,
            version=version,
            kind=kind,
            config=manifest.model_dump(mode="json"),
            layers=layers,
        )
        attest(
            self.registry,
            artifact.digest,
            private_key_pem=private_key_pem,
            name=name,
            version=version,
            inputs=inputs,
            components=components,
        )
        # The publish-side half of verify-twice, through the one gate the service and curation also
        # use (hub.md §2.3) — so a check can never exist on one admission path and be missing from
        # another. Pinned to the client's trusted key when one is set.
        admit(
            self.registry,
            self.catalog,
            manifest,
            digest=artifact.digest,
            publisher=publisher,
            namespace=namespace,
            trusted_public_key_pem=self.trusted_public_key_pem,
        )
        return artifact

    def resolve(self, request: ResolutionRequest) -> Resolution:
        """Resolve a constraint set to a pinned-digest artifact (via the catalog)."""
        return resolve(self.catalog, request)

    def _cache_path(self, digest: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / digest.split(":", 1)[1]

    def verify(self, reference: str, *, require: Sequence[str] = DEFAULT_REQUIRED) -> str:
        """Re-verify a resolved artifact's integrity + attestations; return its digest."""
        digest = self.registry.resolve(reference).digest
        supply_verify(
            self.registry,
            digest,
            trusted_public_key_pem=self.trusted_public_key_pem,
            require=require,
        )
        return digest

    def _verified_digest(self, reference: str, *, verify: bool, require: Sequence[str]) -> str:
        """Resolve ``reference`` and run the supply-chain check (the gate every read passes)."""
        digest = self.registry.resolve(reference).digest
        if verify:
            supply_verify(
                self.registry,
                digest,
                trusted_public_key_pem=self.trusted_public_key_pem,
                require=require,
            )
        return digest

    def pull(
        self, reference: str, *, verify: bool = True, require: Sequence[str] = DEFAULT_REQUIRED
    ) -> bytes:
        """Resolve, **re-verify** (unless disabled), cache-by-digest, and return the config.

        The client re-runs the supply-chain check before returning bytes (defense in depth); a
        tampered artifact raises ``SupplyChainError`` — fail closed.
        """
        digest = self._verified_digest(reference, verify=verify, require=require)
        cached = self._cache_path(digest)
        if cached is not None and cached.exists():
            return cached.read_bytes()
        self.registry.verify(digest)
        config = self.registry.read_config(digest)
        if cached is not None:
            cached.write_bytes(config)
        return config

    def load(self, reference: str, *, require: Sequence[str] = DEFAULT_REQUIRED) -> PluginManifest:
        """Pull + re-verify, then hand the manifest to a **Core** registry — verify-before-load.

        The client verifies the supply chain *first* (:meth:`pull`), then registers the manifest
        through Core's ``PluginRegistry`` (the plugin-load contract), so trust is re-established on
        the client before Core loads anything (hub.md §2.3).
        """
        manifest = PluginManifest.model_validate_json(
            self.pull(reference, verify=True, require=require)
        )
        PluginRegistry(require_signature=False).register(manifest)
        return manifest

    # -- payload layers (the bytes the artifact exists to carry) -------------------------------

    def payload_descriptors(
        self,
        reference: str,
        *,
        media_type: str | None = None,
        verify: bool = True,
        require: Sequence[str] = DEFAULT_REQUIRED,
    ) -> tuple[Descriptor, ...]:
        """The artifact's payload-layer descriptors, in manifest order (supply chain checked first).

        Use this to *select* a layer (by media type or digest) before fetching it with
        :meth:`pull_layer` — a multi-GB world need not be materialized to be inspected.
        """
        digest = self._verified_digest(reference, verify=verify, require=require)
        layers = [
            Descriptor.from_dict(layer) for layer in self.registry.read_manifest(digest)["layers"]
        ]
        return tuple(
            layer for layer in layers if media_type is None or layer.media_type == media_type
        )

    def _verified_bytes(self, descriptor: Descriptor) -> bytes:
        """Fetch one layer and **re-hash it against its content address** before returning it.

        The single choke point for payload bytes: cache hit or registry fetch, local layout or
        remote registry, the bytes handed back always hash to the digest the (verified) manifest
        committed to. A mismatch raises :class:`~astro_mine.hub.registry.IntegrityError` — a caller
        never receives unverified content (hub.md §2.3; conventions.md §9).
        """
        cached = self._cache_path(descriptor.digest)
        data = (
            cached.read_bytes()
            if cached is not None and cached.exists()
            else self.registry.pull_blob(descriptor.digest)
        )
        actual = content_hash(data)
        if actual != descriptor.digest:
            raise IntegrityError(
                f"payload layer {descriptor.digest} content-address mismatch (bytes hash {actual})"
            )
        if cached is not None and not cached.exists():
            cached.write_bytes(data)
        return data

    def pull_payload(
        self,
        reference: str,
        *,
        media_type: str | None = None,
        verify: bool = True,
        require: Sequence[str] = DEFAULT_REQUIRED,
    ) -> tuple[PayloadLayer, ...]:
        """Resolve, re-verify, and return the artifact's **verified payload layers**.

        This is the payload half of :meth:`pull` (which returns the Core manifest): the ONNX policy,
        SADF bundle, or Zarr/COG world the artifact carries. Each layer's bytes are re-hashed
        against
        the digest the verified manifest commits to *before* they are returned, so payload retrieval
        is inside the verify-before-use contract — no caller needs ``registry.pull_blob()``.
        Optionally filtered to one ``media_type``. Fails closed on any tamper.
        """
        descriptors = self.payload_descriptors(
            reference, media_type=media_type, verify=verify, require=require
        )
        return tuple(PayloadLayer(desc, self._verified_bytes(desc)) for desc in descriptors)

    def pull_layer(
        self,
        reference: str,
        digest: str,
        *,
        verify: bool = True,
        require: Sequence[str] = DEFAULT_REQUIRED,
    ) -> bytes:
        """The verified bytes of the single payload layer ``digest`` of ``reference``.

        ``digest`` MUST be a layer of the artifact's **verified** manifest — an arbitrary blob
        digest is refused with :class:`~astro_mine.hub.registry.ArtifactNotFound`, so this cannot be
        used to
        launder an unattested blob out of the registry.
        """
        for descriptor in self.payload_descriptors(reference, verify=verify, require=require):
            if descriptor.digest == digest:
                return self._verified_bytes(descriptor)
        raise ArtifactNotFound(f"{digest} is not a payload layer of {reference}")

    def materialize(
        self,
        reference: str,
        *,
        dest: str | Path | None = None,
        media_type: str | None = None,
        verify: bool = True,
        require: Sequence[str] = DEFAULT_REQUIRED,
    ) -> tuple[Path, ...]:
        """Write the verified payload layers to a **content-addressed** dir; return the paths.

        Each layer lands at ``<dir>/<hex>`` (its digest), so materialization is idempotent and
        de-duplicated across artifacts sharing a layer. ``dest`` defaults to the client's
        ``cache_dir``; a client with neither raises ``ValueError``. Bytes are verified before they
        are written — nothing unverified ever reaches the filesystem.
        """
        target = Path(dest) if dest is not None else self.cache_dir
        if target is None:
            raise ValueError("materialize() needs a dest directory or a client cache_dir")
        target.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for layer in self.pull_payload(
            reference, media_type=media_type, verify=verify, require=require
        ):
            path = target / layer.digest.split(":", 1)[1]
            if not path.exists():
                path.write_bytes(layer.data)
            paths.append(path)
        return tuple(paths)
