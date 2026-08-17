# SPDX-License-Identifier: Apache-2.0
"""Reproducible, digest-pinned image builds.

Renders a reproducible Dockerfile for a Phase-0 workload (Sim, Bench) on a **pinned base
image** (by digest) and, given a builder, produces a **digest-pinned**
:class:`~astro_mine.cloud.packaging.image.ImageRef`. Determinism is the point: a pinned
base, a fixed ``SOURCE_DATE_EPOCH``, and no build-time network mean two builds of the
same source yield the same image (``cloud.md`` §4 principle 4, §7; ``conventions.md`` §7).

The build *recipe* (:func:`render_dockerfile`) and the plumbing (:func:`build_image`) are
pure and unit-tested by injecting a builder, mirroring how
:class:`~astro_mine.cloud.artifacts.s3.S3ArtifactStore` takes an injectable client. The
default :class:`DockerBuildxBuilder` shells out to ``docker buildx`` and so runs only in
a Docker-present environment -- never in the dependency-free local tier.

Backlog: RM-P0-CLOUD-01 -- astro-mine-cloud#1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.cloud.packaging.image import ImageRef

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "BuildSpec",
    "DockerBuildxBuilder",
    "ImageBuilder",
    "build_image",
    "render_dockerfile",
]


class BuildSpec(BaseModel):
    """A reproducible workload-image build recipe.

    ``base`` is the digest-pinned base image; ``repository`` is the target image name;
    ``entrypoint`` is the container entrypoint (argv). ``source_date_epoch`` fixes the
    build clock, ``revision`` records the source commit, and ``labels`` supplies extra
    OCI annotations -- all folded into a deterministic Dockerfile.
    """

    model_config = ConfigDict(extra="forbid")

    base: ImageRef
    repository: str
    version: str
    entrypoint: list[str]
    source_date_epoch: int = 0
    revision: str | None = None
    title: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class ImageBuilder(Protocol):
    """Builds a rendered Dockerfile and returns the resulting ``sha256:<hex>`` digest."""

    def build(
        self,
        *,
        context: str,
        dockerfile: str,
        tags: Sequence[str],
        build_args: Mapping[str, str],
    ) -> str: ...


def _oci_labels(spec: BuildSpec) -> dict[str, str]:
    """Assemble the OCI image annotations, with caller ``labels`` taking precedence."""
    labels = {
        "org.opencontainers.image.title": spec.title or spec.repository,
        "org.opencontainers.image.version": spec.version,
        "org.opencontainers.image.base.name": spec.base.repository,
        "org.opencontainers.image.base.digest": spec.base.digest,
    }
    if spec.revision is not None:
        labels["org.opencontainers.image.revision"] = spec.revision
    labels.update(spec.labels)
    return labels


def render_dockerfile(spec: BuildSpec) -> str:
    """Render a reproducible Dockerfile for *spec* (deterministic; labels sorted)."""
    label_block = " \\\n      ".join(
        f'{key}="{value}"' for key, value in sorted(_oci_labels(spec).items())
    )
    return _TEMPLATE.format(
        base=spec.base.reference,
        epoch=spec.source_date_epoch,
        labels=label_block,
        entrypoint=json.dumps(list(spec.entrypoint)),
    )


def build_image(
    spec: BuildSpec,
    *,
    context: str = ".",
    tags: Sequence[str] = (),
    builder: ImageBuilder | None = None,
) -> ImageRef:
    """Build *spec* and return the resulting digest-pinned :class:`ImageRef`.

    ``SOURCE_DATE_EPOCH`` is passed as a build arg so the image metadata is reproducible.
    With no *builder*, the Docker-backed :class:`DockerBuildxBuilder` is used.
    """
    builder = builder or DockerBuildxBuilder()
    digest = builder.build(
        context=context,
        dockerfile=render_dockerfile(spec),
        tags=list(tags),
        build_args={"SOURCE_DATE_EPOCH": str(spec.source_date_epoch)},
    )
    return ImageRef(repository=spec.repository, digest=digest, tag=spec.version)


class DockerBuildxBuilder:
    """Builds via ``docker buildx`` and returns the built image's digest.

    Shells out to Docker, so it runs only where a Docker daemon is present -- exercised by
    the opt-in ``docker``-marked tests, never in the dependency-free default path.
    """

    def build(  # pragma: no cover - requires a Docker daemon
        self,
        *,
        context: str,
        dockerfile: str,
        tags: Sequence[str],
        build_args: Mapping[str, str],
    ) -> str:
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dockerfile_path = Path(tmp) / "Dockerfile"
            dockerfile_path.write_text(dockerfile)
            metadata_path = Path(tmp) / "metadata.json"
            argv = [
                "docker",
                "buildx",
                "build",
                "--file",
                str(dockerfile_path),
                "--metadata-file",
                str(metadata_path),
                "--provenance=false",
            ]
            for key, value in build_args.items():
                argv += ["--build-arg", f"{key}={value}"]
            for tag in tags:
                argv += ["--tag", tag]
            argv.append(context)
            subprocess.run(argv, check=True)
            metadata = json.loads(metadata_path.read_text())
        return str(metadata["containerimage.digest"])


_TEMPLATE = """\
# syntax=docker/dockerfile:1
#
# Reproducible workload image -- generated by
# astro_mine.cloud.packaging.render_dockerfile (RM-P0-CLOUD-01). A digest-pinned base, a
# fixed SOURCE_DATE_EPOCH, and no build-time network make the build byte-stable, so a
# laptop image and a cluster image are identical (cloud.md §4, §7; conventions.md §7).
# Do not hand-edit -- regenerate it from the BuildSpec.
#
FROM {base}

# Deterministic build clock -- reproducible timestamps in the image metadata.
ARG SOURCE_DATE_EPOCH={epoch}
ENV SOURCE_DATE_EPOCH=${{SOURCE_DATE_EPOCH}}

# OCI image annotations (sorted for a stable layer).
LABEL {labels}

# Unprivileged by default: a numeric uid:gid so no package manager / distro is assumed
# (distroless-compatible). Cloud never runs workloads as root in shared tenancy
# (cloud.md §9).
USER 65532:65532
WORKDIR /work

ENTRYPOINT {entrypoint}
"""
