"""Digest-pinned OCI image references.

An :class:`ImageRef` is the packaging discipline's unit of currency: a workload image
pinned by content **digest** (never a floating tag), so a run starts from the same bytes
on a laptop as on a future cluster (``cloud.md`` §4 principle 4; ``conventions.md`` §7).
The digest is validated through the platform's shared ``sha256:<hex>`` convention in
:mod:`astro_mine.cloud.artifacts.addressing`, so an image digest and an artifact address
speak one hash language.

Boundary enforcement lives in :meth:`ImageRef.parse`: a ``repository:tag`` string with
no ``@sha256:<hex>`` is rejected outright -- an unpinned image can never enter a
:class:`~astro_mine.cloud.submission.jobspec.JobSpec`.

Backlog: RM-P0-CLOUD-01 -- https://github.com/astro-mine/astro-mine-cloud/issues/1
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from astro_mine.cloud.artifacts.addressing import format_address, parse_address

__all__ = ["ImageRef"]


class ImageRef(BaseModel):
    """A digest-pinned OCI image reference (``repository@sha256:<hex>``).

    ``repository`` is the image name (registry/namespace/name, without a tag or digest);
    ``digest`` is the ``sha256:<hex>`` content digest that pins the exact image; ``tag``
    is optional and *informational only* -- reproducibility rides on the digest, never on
    the tag.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str
    digest: str
    tag: str | None = None

    @field_validator("repository")
    @classmethod
    def _clean_repository(cls, value: str) -> str:
        if not value or value != value.strip() or " " in value:
            raise ValueError("repository must be a non-empty image name without whitespace")
        if "@" in value:
            raise ValueError("repository must not contain a digest; pass it as `digest`")
        return value

    @field_validator("digest")
    @classmethod
    def _canonical_digest(cls, value: str) -> str:
        # parse_address validates sha256:<64hex> (and accepts Sim's bare-hex form);
        # re-format to the canonical `sha256:<hex>` string so equal images compare equal.
        _algorithm, hexdigest = parse_address(value)
        return format_address(hexdigest)

    @property
    def reference(self) -> str:
        """The pinned pull reference ``repository@sha256:<hex>``."""
        return f"{self.repository}@{self.digest}"

    @classmethod
    def parse(cls, reference: str, *, tag: str | None = None) -> ImageRef:
        """Parse a ``repository@sha256:<hex>`` reference; reject anything not digest-pinned."""
        repository, separator, digest = reference.partition("@")
        if not separator:
            raise ValueError(
                f"unpinned image reference {reference!r}: expected 'repository@sha256:<hex>'"
            )
        return cls(repository=repository, digest=digest, tag=tag)

    def __str__(self) -> str:
        return self.reference
