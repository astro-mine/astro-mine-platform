# SPDX-License-Identifier: Apache-2.0
"""The ``astro-mine-hub`` binding of the :class:`~astro_mine.sim.runtime.content.BundleStore` seam.

Sim's resolver depends on the narrow :class:`BundleStore` Protocol, not on Hub; this adapter is the
one place that touches ``astro_mine.hub`` and it does so **lazily** (the import lives inside
:func:`open_bundle_store`), so the base ``astro-mine-sim`` wheel stays Hub-free — install the
``[hub]`` extra to resolve content from a real registry. The store reads a local OCI-layout
registry (``files/hub-registry/`` by convention), verifying the supply chain **and** each payload
layer's content address fail-closed on pull; no hosted Hub or Cloud is required (hub.md principle
7).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import RegistryClient

__all__ = ["HubBundleStore", "open_bundle_store"]


class HubBundleStore:
    """Adapt a :class:`astro_mine.hub.client.HubClient` to Sim's ``BundleStore`` Protocol.

    **Every byte Sim consumes comes off the client's verified path, never the raw registry.**
    ``pull_manifest`` goes through :meth:`HubClient.pull` (re-verifying signature/SLSA/SBOM before
    returning the config bytes) and ``pull_layers`` through :meth:`HubClient.pull_payload`, whose
    descriptors are read off the *verified* manifest and whose bytes are re-hashed against their
    content address before return — a tampered or corrupted layer raises
    :class:`~astro_mine.hub.registry.IntegrityError` rather than being decoded into a SADF asset or
    handed to a provider factory. That content-address check holds even under ``verify=False``
    (which relaxes only the supply-chain re-check), so no unverified byte reaches Sim on any path
    (hub.md §2.3; conventions.md §9).

    Only ``resolve_digest`` still touches the :class:`~astro_mine.hub.registry.RegistryClient` — a
    reference→digest lookup, not content — and it is the Protocol, not the concrete local-layout
    ``Registry``, so a hosted client satisfies this seam too.
    """

    def __init__(self, client: HubClient) -> None:
        self._client = client
        self._registry: RegistryClient = client.registry

    def resolve_digest(self, reference: str) -> str:
        return self._registry.resolve(reference).digest

    def pull_manifest(self, reference: str, *, verify: bool = True) -> bytes:
        return self._client.pull(reference, verify=verify)

    def pull_layers(self, reference: str, *, verify: bool = True) -> dict[str, bytes]:
        # ``pull_payload`` yields the verified layers in manifest order, so a bundle that repeats a
        # media type keeps the same last-wins collapse the raw-registry read had.
        return {
            layer.media_type: layer.data
            for layer in self._client.pull_payload(reference, verify=verify)
        }


def open_bundle_store(
    path: str | Path,
    *,
    trusted_public_key_pem: bytes | None = None,
    cache_dir: str | Path | None = None,
) -> HubBundleStore:
    """Open a :class:`HubBundleStore` over the local OCI-layout registry at ``path``.

    Requires the ``[hub]`` extra (``astro-mine-hub``). Pass ``trusted_public_key_pem`` to pin the
    publisher key, and ``cache_dir`` for a content-addressed pull cache.
    """
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Registry

    client = HubClient(
        Registry(str(path)),
        cache_dir=cache_dir,
        trusted_public_key_pem=trusted_public_key_pem,
    )
    return HubBundleStore(client)
