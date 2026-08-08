"""The RunContext provenance envelope.

Records the minimum a run needs to reproduce byte-for-byte -- inputs by content hash,
producing code version, environment lockfile, and seed (``conventions.md`` §5) -- plus
fields reserved for the execution substrate (image digest, Core interface version,
MLflow run id, outputs) that downstream Cloud work populates *without a schema bump*.

This is a **Cloud-local** envelope: Cloud "defines no new Core schema" (``cloud.md``
§6). Field names are kept aligned with Sim's MCAP provenance manifest (SIM-09) and
Bench so a future, RFC-gated shared Core schema can absorb it. It is a **sibling** of
the Sim manifest, not a superset -- it references a run by content hash and never
carries physics fields (engine tiers, error budgets); Cloud has no physics
(``cloud.md`` §1).

Backlog: RM-P0-CLOUD-03 -- astro-mine-cloud#3
"""

from __future__ import annotations

import sys
from importlib import metadata
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astro_mine.cloud._compat import validate_core_interface_version
from astro_mine.cloud.artifacts import addressing

if TYPE_CHECKING:
    from astro_mine.core.artifacts import ArtifactStore

__all__ = ["EnvironmentFingerprint", "RunContext", "code_version"]


def code_version(distribution: str = "astro-mine-cloud") -> str:
    """Return the installed version of *distribution* (the producing code version).

    Matches Sim's convention -- the producing *package* version via
    ``importlib.metadata``, not a git SHA -- and falls back to ``"0.0.0"`` when the
    distribution is not installed.
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "0.0.0"


class _Model(BaseModel):
    """Base model: reject unknown/typo'd fields loudly (house style)."""

    model_config = ConfigDict(extra="forbid")


class EnvironmentFingerprint(_Model):
    """Observational environment stamp -- recorded, but *outside* the determinism set."""

    code_version: str = Field(default_factory=code_version)
    python: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=lambda: sys.platform)


class RunContext(_Model):
    """The reproducibility envelope attached to a run.

    ``source_content_hashes``, ``code_version``, ``env_lockfile`` and ``seed`` are the
    ``conventions.md`` §5 minimum; the remaining fields are reserved for downstream
    Cloud work and default to empty/``None`` so populating them needs no schema bump.
    """

    schema_version: Literal["0.1"] = "0.1"
    # conventions.md §5 reproducibility minimum --------------------------------
    source_content_hashes: dict[str, str] = Field(default_factory=dict)
    code_version: str = Field(default_factory=code_version)
    env_lockfile: str | None = None
    seed: int | None = None
    # produced outputs, content-addressed --------------------------------------
    outputs: dict[str, str] = Field(default_factory=dict)
    # reproducibility-relevant, populated by downstream Cloud work -------------
    image_digest: str | None = None  # pinned OCI image (RM-P0-CLOUD-01)
    core_interface_version: str | None = None  # Core admission (RM-P0-CLOUD-02 / CORE-05)
    # observational bookkeeping, excluded from content_address() ----------------
    # ``run_id`` is the MLflow id (RM-P1-CLOUD-05); it is keyed *by* the pin, not part of it.
    run_id: str | None = None
    environment: EnvironmentFingerprint = Field(default_factory=EnvironmentFingerprint)

    @field_validator("core_interface_version")
    @classmethod
    def _check_core_interface_version(cls, v: str | None) -> str | None:
        """Admit only a Core interface version this Core can satisfy (``_compat``)."""
        return validate_core_interface_version(v)

    def content_address(self) -> str:
        """Return the content address of this envelope's *deterministic* core.

        Excludes the observational ``environment`` stamp and the ``run_id`` (an MLflow id
        assigned *from* the pin, not part of it), so the address depends only on
        reproducibility-relevant fields -- mirroring how Sim splits the hashed ``run`` record
        from the unhashed ``environment`` (SIM-09). Use it to pin a run: two executions of the
        same job reproduce to the same address even under different MLflow run ids.
        """
        return addressing.content_address(
            self.model_dump(mode="json", exclude={"environment", "run_id"})
        )

    def run_pin(self) -> str:
        """A run identity **stable across the whole lifecycle** -- the input pin without outputs.

        :meth:`content_address` folds ``outputs`` in, so it changes once a run produces them; a
        lifecycle event stream needs one identity shared by ``submitted`` (no outputs yet) through
        ``completed`` (outputs present). ``run_pin`` is that identity: the content address of the
        reproducibility core with ``outputs`` (and ``environment``/``run_id``) excluded, so the
        pre-run context and the final context of the *same* job pin identically (RM-P1-CLOUD-06).
        """
        return addressing.content_address(
            self.model_dump(mode="json", exclude={"environment", "run_id", "outputs"})
        )

    def to_json(self) -> bytes:
        """Serialize the full envelope (including ``environment``) to UTF-8 JSON bytes."""
        return self.model_dump_json().encode()

    @classmethod
    def from_json(cls, data: bytes | str) -> RunContext:
        """Parse a RunContext from JSON bytes/str, rejecting unknown fields."""
        return cls.model_validate_json(data)

    def store(self, store: ArtifactStore) -> str:
        """Write the full envelope to *store*; return the content address of its bytes.

        Note this addresses the serialized bytes (with ``environment``); use
        :meth:`content_address` for the environment-independent run pin.
        """
        return store.put(self.to_json())

    @classmethod
    def load(cls, store: ArtifactStore, address: str) -> RunContext:
        """Read a RunContext back from *store* by content address."""
        return cls.from_json(store.get(address))
