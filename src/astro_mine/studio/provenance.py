"""Reproducibility-by-construction provenance (studio.md §2 principle 6, §5).

Every produced artifact records the inputs it was built from (content hashes), the
Core interface versions it was built against, the sibling engine versions that
computed it, the seed, and the environment lockfile — so re-running a study with the
same inputs reproduces the same result (the foundation RM-P1-STUDIO-07 later gates in
CI). The envelope is **deterministic**: it carries no wall-clock, so it can itself be
content-addressed. Observational, non-reproducible facts (who/when) live in the
workspace audit log, not here.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version

from pydantic import Field

from astro_mine.core.compat import CORE_INTERFACE_VERSIONS

from ._base import FrozenStudioModel
from .hashing import content_hash_json

# The resolved-environment fingerprint hashes these versions (conventions.md §5).
_FINGERPRINT_PACKAGES = ("astro-mine-studio", "astro-mine-core", "numpy", "scipy", "pydantic")


def _studio_version() -> str:
    try:
        return version("astro-mine-platform")
    except PackageNotFoundError:  # pragma: no cover - source tree without metadata
        return "0.0.0"


def environment_fingerprint() -> str:
    """A deterministic ``sha256`` over the resolved versions of the key dependencies — the
    runtime-observable stand-in for the environment lockfile (the library can't read
    ``uv.lock`` at run time). Stable within one environment, so it never perturbs
    within-run determinism; it changes when a dependency version changes, which is exactly
    the reproducibility signal (studio.md §5)."""
    versions: dict[str, str] = {}
    for name in _FINGERPRINT_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - all pinned deps are installed
            continue
    return content_hash_json(versions)


class ArtifactProvenance(FrozenStudioModel):
    """The reproducibility envelope stamped on every produced Studio artifact."""

    input_hashes: list[str] = Field(default_factory=list)
    core_interface_versions: dict[str, str] = Field(default_factory=dict)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    seed: int | None = None
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None


def capture_provenance(
    *,
    input_hashes: Sequence[str],
    seed: int | None = None,
    engine_versions: Mapping[str, str] | None = None,
    env_lockfile: str | None = None,
) -> ArtifactProvenance:
    """Build a provenance envelope, snapshotting the Core interface versions, the
    Studio/toolchain versions, and the environment fingerprint of the running process.
    ``env_lockfile`` defaults to :func:`environment_fingerprint`; pass an explicit lockfile
    hash to override."""
    return ArtifactProvenance(
        input_hashes=list(input_hashes),
        core_interface_versions=dict(CORE_INTERFACE_VERSIONS),
        engine_versions=dict(engine_versions or {}),
        seed=seed,
        code_version=_studio_version(),
        toolchain_version=f"python{sys.version_info.major}.{sys.version_info.minor}",
        env_lockfile=env_lockfile if env_lockfile is not None else environment_fingerprint(),
    )
