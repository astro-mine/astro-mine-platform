# SPDX-License-Identifier: Apache-2.0
"""Provenance bundle + re-execution integrity (RM-P1-BENCH-10; bench.md §5, §9).

Every hosted leaderboard entry must carry the lineage that makes it **byte-for-byte reproducible**
and must survive a **re-execution from that lineage** — the two halves of the public integrity
posture (bench.md §9). This module is both:

- :class:`ProvenanceBundle` — the content-addressed record binding a scored submission to its
  full lineage: the ScenarioSpec hash, the Core interface versions, the pinned content hashes, the
  submission source (a Hub digest for a community submission, or a ``policy_ref`` for the local
  tier), the code version + dependency lockfile, the held-out seeds, and the per-seed scored
  values. :attr:`~ProvenanceBundle.bundle_hash` digests only the reproducible fields (the machine
  stamp is recorded but excluded), mirroring the harness :class:`~astro_mine.bench.harness.Result`.
- :func:`resample_from_bundle` — the determinism-enforcement audit: re-run a **sampled fraction**
  of the bundle's seeds and compare per-seed values against the recorded bundle; any mismatch
  returns ``"flagged"`` (non-determinism or tampering — bench.md §9). This is "sampled re-execution
  *from the provenance bundle*", as opposed to the in-line sample the P0 leaderboard runs.

The bundle is stored in the object store (:mod:`._objects`) keyed by its own digest, so a dispute
is auditable from the stored lineage alone.

Backlog: RM-P1-BENCH-10 — astro-mine-bench#18
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.bench.harness import EnvironmentStamp, environment_stamp, lockfile_digest
from astro_mine.bench.leaderboard._models import Integrity
from astro_mine.bench.metrics import Scorecard
from astro_mine.bench.sandbox import PolicyScorer
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.scenario._hash import content_hash
from astro_mine.core import SCHEMA_DIGEST

__all__ = [
    "ProvenanceBundle",
    "SeedRecord",
    "build_provenance_bundle",
    "resample_from_bundle",
]


class SeedRecord(BaseModel):
    """One held-out seed's scored values — the unit the re-execution audit compares."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    metrics: dict[str, float | None]


class ProvenanceBundle(BaseModel):
    """Full lineage for a leaderboard entry, content-addressed and re-executable (bench.md §5).

    :attr:`bundle_hash` is a deterministic ``sha256:`` over the reproducible fields only — the
    :class:`~astro_mine.bench.harness.EnvironmentStamp` is recorded for audit but excluded, so the
    same inputs reproduce the identical hash on any machine.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    scenario_spec_hash: str
    core_interface_version: dict[str, str]
    #: The Core schema digest the submission validated under — so an entry can be audited against
    #: the exact contract it ran against, not just the frozen interface version (VERSIONING.md
    #: §4.1).
    core_schema_digest: str
    content_hashes: dict[str, str]
    #: How the policy was supplied: a Hub digest (community submission) or a ``policy_ref``.
    source: str = Field(min_length=1)
    #: The resolved Hub image-manifest digest, when the submission came from Hub (else ``None``).
    source_digest: str | None = None
    code_version: str
    environment_lockfile: str
    scorecard_hash: str
    seeds: tuple[int, ...] = Field(min_length=1)
    per_seed: tuple[SeedRecord, ...] = Field(min_length=1)
    environment: EnvironmentStamp

    @property
    def bundle_hash(self) -> str:
        """A deterministic ``sha256:`` over the reproducible lineage (excludes the environment)."""
        return content_hash(
            {
                "code_version": self.code_version,
                "content_hashes": self.content_hashes,
                "core_interface_version": self.core_interface_version,
                "core_schema_digest": self.core_schema_digest,
                "environment_lockfile": self.environment_lockfile,
                "per_seed": [
                    {"seed": r.seed, "metrics": r.metrics}
                    for r in sorted(self.per_seed, key=lambda r: r.seed)
                ],
                "scenario_id": self.scenario_id,
                "scenario_spec_hash": self.scenario_spec_hash,
                "scorecard_hash": self.scorecard_hash,
                "seeds": list(self.seeds),
                "source": self.source,
                "source_digest": self.source_digest,
            }
        )


def _values_by_seed(card: Scorecard) -> dict[int, dict[str, float | None]]:
    """Invert a scorecard into ``{seed: {metric: value}}`` — the per-seed comparison view."""
    by_seed: dict[int, dict[str, float | None]] = {}
    for aggregate in card.metrics:
        for seed, value in zip(aggregate.seeds, aggregate.per_seed, strict=True):
            by_seed.setdefault(seed, {})[aggregate.metric] = value
    return by_seed


def build_provenance_bundle(
    spec: ScenarioSpec,
    card: Scorecard,
    *,
    source: str,
    source_digest: str | None = None,
    code_version: str,
    lockfile: Path | str | None = None,
) -> ProvenanceBundle:
    """Bind a scored run to a content-addressed :class:`ProvenanceBundle` (bench.md §5).

    ``source`` is the submission's provenance handle (a Hub digest or a ``policy_ref``);
    ``code_version`` and the dependency ``lockfile`` complete the reproducibility key. Records every
    held-out seed's per-metric values so the bundle can later be re-executed against
    (:func:`resample_from_bundle`).
    """
    by_seed = _values_by_seed(card)
    records = tuple(SeedRecord(seed=seed, metrics=by_seed[seed]) for seed in sorted(by_seed))
    lock = lockfile_digest(lockfile)
    return ProvenanceBundle(
        scenario_id=spec.scenario_id,
        scenario_spec_hash=spec.spec_hash,
        core_interface_version=dict(spec.core_interface),
        # The contract the submission validated under. When the spec pins a digest, resolve_scenario
        # has already proven the two are equal — a mismatch never reaches a scored run.
        core_schema_digest=SCHEMA_DIGEST,
        content_hashes={ref.id: ref.content_hash for ref in spec.content_refs()},
        source=source,
        source_digest=source_digest,
        code_version=code_version,
        environment_lockfile=lock,
        scorecard_hash=card.content_hash,
        seeds=tuple(sorted(by_seed)),
        per_seed=records,
        environment=environment_stamp(),
    )


def resample_from_bundle(
    bundle: ProvenanceBundle,
    spec: ScenarioSpec,
    policy_ref: str,
    *,
    scorer: PolicyScorer,
    fraction: float = 0.25,
) -> Integrity:
    """Re-execute a sampled fraction of the bundle's seeds and verify they reproduce (bench.md §9).

    Re-runs ``max(1, ceil(fraction * len(seeds)))`` of the bundle's held-out seeds via ``scorer``
    and compares each re-executed seed's per-metric values against the recorded bundle. Returns
    ``"verified"`` when every sampled seed matches, else ``"flagged"`` (non-determinism or
    tampering). The sampled seeds are the deterministic lowest-id prefix, so the audit is itself
    reproducible.

    ``scorer`` is the same sandboxed execution seam the original scoring used (bench#30) — the
    integrity audit re-runs untrusted code, so it must be no less contained than the run it audits.
    """
    import math

    n = max(1, math.ceil(fraction * len(bundle.seeds)))
    sample = tuple(sorted(bundle.seeds))[:n]
    resampled = _values_by_seed(scorer(spec, policy_ref, seeds=sample))
    recorded = {r.seed: r.metrics for r in bundle.per_seed}
    for seed in sample:
        if resampled.get(seed) != recorded.get(seed):
            return "flagged"
    return "verified"
