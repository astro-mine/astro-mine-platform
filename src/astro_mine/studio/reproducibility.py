# SPDX-License-Identifier: Apache-2.0
"""Reproducibility-by-construction — the enforced contract (RM-P1-STUDIO-07).

The guarantee that makes a Studio design trustworthy: **every** produced artifact
(``ObjectiveSpec`` / ``EvaluatedCandidate`` / ``TradeStudy`` / ``Campaign``) records its
input content hashes, the Core interface versions, the sibling engine versions, the seed,
and the environment fingerprint — so re-running a study with the same inputs reproduces
the same result. Reproducibility is a property of the content-addressed artifact chain,
not of any Studio-side engine (studio.md §2 principle 6, §5); this module is the check,
and the property tests assert it holds on the whole chain.
"""

from __future__ import annotations

from .provenance import ArtifactProvenance


class ReproducibilityError(ValueError):
    """A produced artifact is missing provenance required to reproduce it."""


def missing_provenance_fields(
    provenance: ArtifactProvenance, *, require_seed: bool = False
) -> list[str]:
    """The reproducibility fields absent from ``provenance`` — empty when it is complete.
    ``require_seed`` is set for computational artifacts (a candidate evaluation, a trade
    study); authored artifacts like an ``ObjectiveSpec`` carry no seed."""
    missing: list[str] = []
    if not provenance.input_hashes:
        missing.append("input_hashes")
    if not provenance.core_interface_versions:
        missing.append("core_interface_versions")
    if provenance.code_version is None:
        missing.append("code_version")
    if provenance.toolchain_version is None:
        missing.append("toolchain_version")
    if provenance.env_lockfile is None:
        missing.append("env_lockfile")
    if require_seed and provenance.seed is None:
        missing.append("seed")
    return missing


def is_reproducible(provenance: ArtifactProvenance, *, require_seed: bool = False) -> bool:
    """Whether ``provenance`` carries everything needed to reproduce its artifact."""
    return not missing_provenance_fields(provenance, require_seed=require_seed)


def assert_reproducible(provenance: ArtifactProvenance, *, require_seed: bool = False) -> None:
    """Raise :class:`ReproducibilityError` if any reproducibility field is missing."""
    missing = missing_provenance_fields(provenance, require_seed=require_seed)
    if missing:
        raise ReproducibilityError(f"provenance is missing reproducibility fields: {missing}")
