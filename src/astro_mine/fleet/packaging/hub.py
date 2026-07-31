"""Publish and discover SADF asset bundles through Hub (RM-P1-FLEET-10).

Upgrades the P0 pre-Hub local/object-store OCI path (:mod:`astro_mine.fleet.packaging.oci`,
RM-P0-FLEET-06) to **signed publish + discovery through** [Hub](https://github.com/astro-mine/docs)
(``fleet.md`` §5, §6, §12 Phase 1). Fleet **reuses** the exact Core plugin manifest and
content-addressed layers it already builds
(:func:`~astro_mine.fleet.packaging._content.build_asset_content`)
and **pushes** them to a Hub :class:`~astro_mine.hub.registry.Registry` via
:meth:`~astro_mine.hub.client.HubClient.publish` — it does not redesign packaging. The signed
artifact is content-addressed and re-verified **before trust** on pull (``hub.md`` §2.3), and its
catalog metadata (the Core plugin manifest Hub indexes) is discoverable by name/version.

Two boundaries this module enforces:

- **Export-control gate** (``fleet.md`` §9): :func:`publish_asset` refuses any asset carrying a
  reserved/gated capability tag before it reaches Hub (defense in depth over Core's loader gate).
- **SADF JSON layer contract**: the canonical SADF JSON layer keeps the
  :data:`~astro_mine.fleet.packaging.oci.MEDIA_SADF_JSON` media type, so [Sim](sim.md) rehydrates
  the Asset from it via :func:`astro_mine.core.sadf.load_sadf`.

``astro-mine-hub`` is imported **lazily** (inside each function) so ``import astro_mine.fleet``
stays light — the Hub registry/supply-chain stack loads only when a publish/pull path runs.

Backlog: RM-P1-FLEET-10 -- https://github.com/astro-mine/astro-mine-fleet/issues/21
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from astro_mine.core.sadf import SadfDocument, load_sadf
from astro_mine.fleet.capabilities import assert_open_commons
from astro_mine.fleet.packaging import oci
from astro_mine.fleet.packaging._content import build_asset_content

if TYPE_CHECKING:
    from astro_mine.hub.registry import RegistryClient

__all__ = ["HubError", "HubPublication", "discover_asset", "publish_asset", "pull_asset"]


class HubError(Exception):
    """A Hub publish/pull could not be completed (e.g. astro-mine-hub is not installed)."""


@dataclass(frozen=True)
class HubPublication:
    """A published asset artifact: its Hub reference, content digest, and signing state."""

    reference: str  # name:version tag
    digest: str  # the OCI image-manifest digest -- pull/verify by this content hash
    asset_digest: str  # sha256 of the SADF wire form -- the signed content identity
    namespace: str  # open (self-published) | curated (reviewed)
    signed: bool


def _require_hub() -> None:
    try:
        import astro_mine.hub  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise HubError(
            "publishing to Hub requires astro-mine-hub; install it (it is a Fleet dependency, "
            "git-pinned in pyproject.toml [tool.uv.sources])"
        ) from exc


def publish_asset(
    doc: SadfDocument,
    registry: RegistryClient,
    *,
    sign_key: bytes,
    base_dir: str | Path | None = None,
    namespace: str = "open",
    publisher: str = "local",
    version: str | None = None,
) -> HubPublication:
    """Publish *doc* as a signed, content-addressed asset artifact to a Hub registry.

    Reuses the Core plugin manifest + content-addressed layers from
    :func:`~astro_mine.fleet.packaging._content.build_asset_content` and pushes them via
    :meth:`~astro_mine.hub.client.HubClient.publish` (``kind="asset"``). With *sign_key* (an
    ECDSA P-256 private-key PEM) the artifact is cosign-signed and carries SLSA provenance +
    an SBOM (``hub.md`` §9). Refuses an asset declaring a reserved/gated capability tag
    (:func:`~astro_mine.fleet.capabilities.assert_open_commons`). Returns a
    :class:`HubPublication`; the ``digest`` is the content hash to re-pull/verify by.
    """
    _require_hub()
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Blob

    assert_open_commons(doc.asset.capabilities)  # export-control publish gate (fleet.md §9)
    content = build_asset_content(doc, base_dir)
    identity = doc.asset.identity

    client = HubClient(registry)
    layers = [Blob(blob.media_type, blob.data) for blob in content.layers]
    artifact = client.publish(
        name=identity.id,
        version=version or identity.version,
        kind="asset",
        manifest=content.manifest,
        layers=layers,
        private_key_pem=sign_key,
        namespace=namespace,
        publisher=publisher,
        inputs=[content.asset_digest],
    )
    return HubPublication(
        reference=artifact.reference,
        digest=artifact.digest,
        asset_digest=content.asset_digest,
        namespace=namespace,
        signed=sign_key is not None,
    )


def pull_asset(
    registry: RegistryClient,
    reference: str,
    *,
    trusted_public_key_pem: bytes | None = None,
    verify: bool = True,
    require: Sequence[str] | None = None,
) -> SadfDocument:
    """Pull an asset from a Hub registry **by content hash** and rehydrate the SADF document.

    *reference* is a ``name:version`` tag or a ``sha256:…`` content digest. Unless *verify* is
    ``False``, the full supply chain is re-verified client-side **before trust** (``hub.md``
    §2.3) — a tampered artifact fails closed. *require* selects the attestations demanded
    (default: signature + SLSA + SBOM, i.e. a signed publish); pass ``()`` for an unsigned one.
    The Asset is rebuilt from the SADF JSON layer via :func:`astro_mine.core.sadf.load_sadf`.
    """
    _require_hub()
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.supply_chain import DEFAULT_REQUIRED

    client = HubClient(registry, trusted_public_key_pem=trusted_public_key_pem)
    demanded = tuple(DEFAULT_REQUIRED) if require is None else tuple(require)
    digest = (
        client.verify(reference, require=demanded) if verify else registry.resolve(reference).digest
    )

    manifest = registry.read_manifest(digest)
    for layer in manifest["layers"]:
        if layer["mediaType"] == oci.MEDIA_SADF_JSON:
            return load_sadf(registry.pull_blob(layer["digest"]))
    raise HubError(f"artifact {reference} has no SADF JSON layer to rehydrate")


def discover_asset(
    registry: RegistryClient, name: str, *, version_spec: str = ""
) -> tuple[str, str]:
    """Discover an asset in a Hub registry's catalog by *name*; return ``(reference, digest)``.

    Rebuilds the catalog from the registry's stored Core plugin manifests and resolves the
    highest version satisfying *version_spec* (a PEP 440 specifier; ``""`` = any) to its
    immutable digest (``hub.md`` §3, §5). Raises
    :class:`~astro_mine.hub.resolve.ResolutionError` if nothing matches.
    """
    _require_hub()
    from astro_mine.hub.client import HubClient, catalog_from_registry
    from astro_mine.hub.resolve import ResolutionRequest

    client = HubClient(registry, catalog=catalog_from_registry(registry))
    primary = client.resolve(ResolutionRequest(name=name, version_spec=version_spec)).primary
    return primary.reference, primary.digest
