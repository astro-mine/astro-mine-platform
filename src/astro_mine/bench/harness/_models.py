# SPDX-License-Identifier: Apache-2.0
"""Content-addressed Result + provenance models for the reproducibility harness (bench.md §5).

A :class:`Result` binds a run to the lineage a leaderboard entry needs to be *byte-for-byte
reproducible*: the ScenarioSpec and resolved-scenario hashes, the Core interface versions, the
pinned content hashes, the runner identity, the code version and dependency lockfile, and the
per-seed + aggregate scores. :attr:`Result.result_hash` digests only the deterministic fields — the
:class:`EnvironmentStamp` (interpreter, platform) is recorded for audit but excluded, so the same
inputs reproduce the identical hash on any machine (mirrors Sim's provenance split, SIM-09).

Backlog: RM-P0-BENCH-04 — astro-mine-bench#4
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.bench.scenario._hash import content_hash

__all__ = ["EnvironmentStamp", "ReproductionReport", "Result", "SeedResult"]


class _Model(BaseModel):
    """Frozen base: reject unknown fields loudly, immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvironmentStamp(_Model):
    """The machine fingerprint — recorded for audit, kept OUT of the result hash.

    Excluded from :attr:`Result.result_hash` so a Result reproduces byte-for-byte across machines
    and interpreters (mirrors Sim's environment fingerprint, ``recording/__init__.py``).
    """

    python: str
    platform: str


class SeedResult(_Model):
    """A single seed's scored outcome within a :class:`Result`."""

    seed: int
    determinism_key: str
    metrics: dict[str, float]


class Result(_Model):
    """A content-addressed benchmark result binding a run to its provenance (bench.md §5).

    :attr:`result_hash` is a deterministic digest over the reproducible fields only — two runs with
    the same pinned inputs produce the identical hash regardless of the recorded environment.
    """

    scenario_id: str
    scenario_spec_hash: str
    scenario_hash: str
    core_interface_version: dict[str, str]
    #: The Core schema digest the run resolved against — the contract the result was produced under
    #: (VERSIONING.md §4.1). The interface versions above are frozen, so this is the field that
    #: actually distinguishes one Core schema set from another when auditing a submission.
    core_schema_digest: str
    content_hashes: dict[str, str]
    runner: str
    code_version: str
    environment_lockfile: str
    environment: EnvironmentStamp
    per_seed: tuple[SeedResult, ...] = Field(min_length=1)
    aggregate: dict[str, float]

    @property
    def result_hash(self) -> str:
        """A deterministic ``sha256:`` over the reproducible fields (excludes the environment)."""
        return content_hash(
            {
                "aggregate": self.aggregate,
                "code_version": self.code_version,
                "content_hashes": self.content_hashes,
                "core_interface_version": self.core_interface_version,
                "core_schema_digest": self.core_schema_digest,
                "environment_lockfile": self.environment_lockfile,
                "per_seed": [
                    {"seed": s.seed, "determinism_key": s.determinism_key, "metrics": s.metrics}
                    for s in self.per_seed
                ],
                "runner": self.runner,
                "scenario_hash": self.scenario_hash,
                "scenario_id": self.scenario_id,
                "scenario_spec_hash": self.scenario_spec_hash,
            }
        )


class ReproductionReport(_Model):
    """The outcome of running a scenario ``runs`` times and comparing determinism.

    ``reproducible`` is True iff every run produced the identical result hash. ``result`` is the
    canonical (first) run; ``result_hashes`` is the per-run digest list, for auditing a drift.
    """

    scenario_id: str
    reproducible: bool
    runs: int
    result_hash: str
    result_hashes: tuple[str, ...]
    result: Result
