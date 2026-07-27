"""SemVer + Core-interface-range dependency & compatibility resolution (RM-P1-HUB-04).

Resolve a constraint set (name/version range + Core interface range + capability tags) to a
**compatible, pinned-digest** artifact, refusing incompatible/unknown Core interface majors and
pinning to immutable digests so a downstream result reproduces exactly (hub.md §3, §11). Consumes
Core's ``compat`` rule; never reimplements version negotiation.

Backlog: RM-P1-HUB-04 — https://github.com/astro-mine/astro-mine-hub/issues/4
"""

from __future__ import annotations

from astro_mine.hub.resolve._resolve import (
    Resolution,
    ResolutionError,
    ResolutionRequest,
    ResolvedArtifact,
    resolve,
)

__all__ = [
    "Resolution",
    "ResolutionError",
    "ResolutionRequest",
    "ResolvedArtifact",
    "resolve",
]
