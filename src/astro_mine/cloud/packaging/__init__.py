"""Container-first, cluster-ready packaging.

Packages every Phase-0 workload (Sim, Bench) as **digest-pinned** OCI images with
**reproducible** builds on **pinned bases**, so they scale out in Phase 1 without rework.
:class:`ImageRef` is the digest-pinned reference every run consumes; :class:`BuildSpec` +
:func:`render_dockerfile` + :func:`build_image` produce one reproducibly.

Backlog: RM-P0-CLOUD-01 -- https://github.com/astro-mine/astro-mine-cloud/issues/1
"""

from __future__ import annotations

from astro_mine.cloud.packaging.build import (
    BuildSpec,
    DockerBuildxBuilder,
    ImageBuilder,
    build_image,
    render_dockerfile,
)
from astro_mine.cloud.packaging.image import ImageRef

__all__ = [
    "BuildSpec",
    "DockerBuildxBuilder",
    "ImageBuilder",
    "ImageRef",
    "build_image",
    "render_dockerfile",
]
