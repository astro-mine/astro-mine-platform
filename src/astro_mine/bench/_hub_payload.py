# SPDX-License-Identifier: Apache-2.0
"""Verified retrieval of an artifact's payload bytes — Bench's one door onto them (bench#44).

Bench **resolves and verifies artifacts by digest but consumes no payload bytes itself**: both Hub
intakes — the leaderboard's policy intake (:mod:`astro_mine.bench.leaderboard._hub`) and metric-
plugin discovery (:mod:`astro_mine.bench.metrics._plugin`) — record the layer digests off the
*verified* manifest and reach the policy/metric through its ``entrypoint`` attribute. The byte
consumer is the **deployment-injected loader** on the other side of each seam
(:class:`~astro_mine.bench.leaderboard.PolicyLoader`,
:class:`~astro_mine.bench.metrics.MetricPluginLoader`): the one that materializes an ONNX policy, or
a metric that lives in a payload layer rather than in an importable module.

:func:`pull_verified_layer` is the door those loaders get, and it is the only one Bench opens. It
opens onto Hub's **verified** retrieval path (:meth:`astro_mine.hub.client.HubClient.pull_payload`),
which is verify-before-use end to end (hub.md §2.3; conventions.md §9):

- every layer's bytes are **re-hashed** against the digest the *verified* manifest commits to before
  a byte is returned — a fetch is a check, not a read;
- a digest that is not a layer of that manifest is **refused**, so the route cannot launder an
  unattested blob out of the registry;
- the artifact's supply-chain evidence (cosign signature, SLSA provenance, SBOM) is re-checked on
  the way through — the same gate the leaderboard applies at admission (bench.md §9).

A raw registry blob read has none of those properties: it hands back bytes that no manifest vouched
for, on the strength of a digest nobody re-derived. Bench offers no such call.

The registry is typed structurally (:class:`ContentRegistry`), as everywhere else in Bench's Hub
seam, so the module stays import-light and free of a private Hub schema (bench.md §2.2); the
concrete :class:`~astro_mine.hub.client.HubClient` is imported lazily, behind the ``[leaderboard]``
extra, so the base package imports without the Hub client.

Backlog: bench#44 — astro-mine-bench#44
"""

from __future__ import annotations

from typing import Any, Protocol, cast

__all__ = [
    "ContentRegistry",
    "PayloadRetrievalError",
    "ResolvedArtifact",
    "pull_verified_layer",
]


class ContentRegistry(Protocol):
    """The registry surface verified retrieval needs — met by both intakes' ``HubRegistry``."""

    def resolve(self, reference: str) -> Any:
        """Resolve a ``name:version`` tag or a digest to its manifest descriptor (``.digest``)."""
        ...

    def verify(self, digest: str) -> None:
        """Assert the stored blob at ``digest`` hashes to it — content-addressing on read."""
        ...

    def read_manifest(self, digest: str) -> dict[str, Any]:
        """The parsed OCI image manifest at ``digest`` (``config`` + ``layers`` descriptors)."""
        ...

    def read_config(self, manifest_digest: str) -> bytes:
        """The artifact's config blob (its Core plugin manifest) given the image-manifest digest."""
        ...


class ResolvedArtifact(Protocol):
    """A verified Hub artifact — met by ``ResolvedSubmission`` and ``ResolvedMetricPlugin``.

    Read-only: the two concrete types are frozen dataclasses, and the only thing verified retrieval
    needs of them is *which* artifact was authenticated, by digest.
    """

    @property
    def reference(self) -> str:
        """The Hub reference the artifact was resolved from (a tag or a ``sha256:`` digest)."""
        ...

    @property
    def manifest_digest(self) -> str:
        """The image-manifest digest the intake verified — the identity the bytes hang off."""
        ...


class PayloadRetrievalError(Exception):
    """Raised when an artifact carries no payload layer the loader asked for."""


def pull_verified_layer(
    registry: ContentRegistry,
    resolved: ResolvedArtifact,
    *,
    media_type: str | None = None,
) -> bytes:
    """The verified bytes of ``resolved``'s payload layer — a loader's route to an artifact's data.

    Retrieval is by **manifest digest**, never by the reference: the bytes come from exactly the
    artifact whose manifest the intake authenticated, so a tag that is re-pointed between
    resolution and load cannot substitute a different payload. Hub re-hashes the layer against the
    digest that verified manifest commits to before returning it (hub.md §2.3; conventions.md §9).

    ``media_type`` selects one layer of a multi-layer artifact (e.g. the ONNX model beside its
    tokenizer). Raises :class:`PayloadRetrievalError` when the artifact carries no such layer, or
    when it carries several and none was named — an ambiguous pull is a caller error, not a coin
    flip. Requires the ``[leaderboard]`` extra (bench.md §2.2).
    """
    from astro_mine.hub.client import HubClient

    # Hub types its client on the concrete Registry; Bench holds the structural protocol (bench.md
    # §2.2 — Bench never reaches into a private Hub schema), so the cast is the seam.
    client = HubClient(cast(Any, registry))
    # Hub's IntegrityError / SupplyChainError propagate unwrapped: they are the fail-closed signal
    # that these bytes are not the bytes the manifest attests to, and dressing them as a Bench error
    # would blunt exactly the alarm a caller must not miss.
    layers = client.pull_payload(resolved.manifest_digest, media_type=media_type)
    if not layers:
        raise PayloadRetrievalError(
            f"artifact {resolved.reference!r} carries no payload layer"
            + (f" of media type {media_type!r}" if media_type is not None else "")
        )
    if len(layers) > 1 and media_type is None:
        raise PayloadRetrievalError(
            f"artifact {resolved.reference!r} carries {len(layers)} payload layers "
            f"({', '.join(layer.descriptor.media_type for layer in layers)}); "
            "name the media_type to pull"
        )
    return layers[0].data
