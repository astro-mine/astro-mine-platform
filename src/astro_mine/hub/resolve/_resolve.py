# SPDX-License-Identifier: Apache-2.0
"""SemVer + Core-interface-range dependency/compat resolution (RM-P1-HUB-04).

Given a constraint set — name, version range, required Core interface versions, and required
capability tags — the resolver returns a **compatible, pinned-digest** artifact (hub.md §3, §11).
It **refuses incompatible / unknown Core interfaces** by consuming Core's rule
(:func:`astro_mine.core.compat.assert_core_compatible` — never reimplemented) and pins to an
**immutable digest**, so a downstream Bench result reproduces exactly (hub.md §2.1, §5). Resolution
is **deterministic**: the same catalog + request always yield the same closure (highest satisfying
version, ties broken by reference).

While Core's ``CORE_INTERFACE_VERSIONS`` is frozen at 0.1.0 (VERSIONING.md), the range check
resolves everything as compatible **but still rejects a misspelled interface name**. Cross-artifact
(shape/CRS) compatibility beyond Core interface majors is an open research dependency (hub.md §11),
so the returned closure is the resolved artifact; the structure holds a set for when Core models
inter-artifact dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from astro_mine.core.compat import (
    CORE_INTERFACE_VERSIONS,
    IncompatibleCoreInterface,
    assert_core_compatible,
)
from astro_mine.hub.index import Catalog, CatalogEntry

__all__ = ["Resolution", "ResolutionError", "ResolutionRequest", "ResolvedArtifact", "resolve"]


class ResolutionError(Exception):
    """No compatible artifact resolves the request, or the request is malformed."""


@dataclass(frozen=True)
class ResolutionRequest:
    """A resolution constraint set (hub.md §3)."""

    name: str
    version_spec: str = ""  # a PEP 440 specifier ("" = any); e.g. ">=1.0.0,<2.0.0"
    interfaces: Mapping[str, str] | None = None  # Core interfaces the artifact must implement
    capability_tags: Sequence[str] = ()
    provided_interfaces: Mapping[str, str] | None = (
        None  # runtime Core (default: CORE_INTERFACE_VERSIONS)
    )
    include_prereleases: bool = False


@dataclass(frozen=True)
class ResolvedArtifact:
    """A pinned artifact: its ``name:version`` reference, immutable digest, and version."""

    reference: str
    digest: str
    version: str


@dataclass(frozen=True)
class Resolution:
    """The pinned-digest closure for a request (deterministic order)."""

    request: ResolutionRequest
    artifacts: tuple[ResolvedArtifact, ...]

    @property
    def primary(self) -> ResolvedArtifact:
        """The resolved artifact (the closure root)."""
        return self.artifacts[0]


def _provided(request: ResolutionRequest) -> Mapping[str, str]:
    if request.provided_interfaces is not None:
        return request.provided_interfaces
    return CORE_INTERFACE_VERSIONS


def _specifier(version_spec: str) -> SpecifierSet:
    try:
        return SpecifierSet(version_spec)
    except InvalidSpecifier as exc:
        raise ResolutionError(f"invalid version specifier {version_spec!r}: {exc}") from exc


def _version(entry: CatalogEntry) -> Version | None:
    try:
        return Version(entry.version)
    except InvalidVersion:
        return None  # a non-PEP440 version can't participate in range resolution


def resolve(catalog: Catalog, request: ResolutionRequest) -> Resolution:
    """Resolve ``request`` against ``catalog`` to a pinned-digest :class:`Resolution`.

    Rejects a request naming an **unknown/misspelled** Core interface (or one incompatible with the
    runtime); then selects catalog entries for ``request.name`` whose version satisfies
    ``version_spec``, that are **runnable on the runtime Core** (refusing incompatible interface
    majors), that **satisfy** the requested interfaces + capability tags, and that are **not
    yanked** — pinning the **highest** such version to its digest. Raises :class:`ResolutionError`
    if nothing resolves.
    """
    provided = _provided(request)
    if request.interfaces:
        try:
            assert_core_compatible(request.interfaces, provided=provided)
        except IncompatibleCoreInterface as exc:
            raise ResolutionError(f"unresolvable request: {exc}") from exc

    specifier = _specifier(request.version_spec)
    tags = list(request.capability_tags) or None
    matches: list[tuple[Version, CatalogEntry]] = []
    for entry in catalog.all():
        if entry.name != request.name or entry.yanked:
            continue
        version = _version(entry)
        if version is None or not specifier.contains(
            version, prereleases=request.include_prereleases
        ):
            continue
        try:
            assert_core_compatible(entry.manifest.core_interfaces, provided=provided)
        except IncompatibleCoreInterface:
            continue  # built against an incompatible Core major — refuse
        if not entry.satisfies(interfaces=request.interfaces, capability_tags=tags):
            continue
        matches.append((version, entry))

    if not matches:
        raise ResolutionError(
            f"no artifact resolves {request.name!r} {request.version_spec or '(any version)'} "
            f"with the requested interfaces/tags on the current Core"
        )
    matches.sort(key=lambda pair: (pair[0], pair[1].reference), reverse=True)
    best = matches[0][1]
    resolved = ResolvedArtifact(reference=best.reference, digest=best.digest, version=best.version)
    return Resolution(request=request, artifacts=(resolved,))
