# SPDX-License-Identifier: Apache-2.0
"""SafetyVerdict v0.1 — the auditable per-tick output of the Guard arbiter (RM-P1-GUARD-06).

The Core-catalogued, Guard-**owned** record of what the trusted Rust safety core
(:mod:`astro_mine.guard._core`) decided each tick: the certified action, whether/why an
intervention occurred, the invoked spec clause(s), the active assurance layer, the
barrier-margin certificate, plus the reproducibility provenance a safety claim needs — the
source ``SafetySpec`` content hash, the compiled-model content hash, the Guard code version,
and the content hash of the inputs the core certified against (guard.md §5, §6).

Two invariants shape this model:

- **It only records; it never decides.** Every field mirrors the trusted core's
  :class:`~astro_mine.guard._core.Verdict` (the fail-safe decision was already made inside the
  TCB, guard.md §9.1) — a logging fault can never change the certified action.
- **It reports Guard's own certificate, never the wrapped policy's internals** (guard.md §6,
  §9.1). The shielding-cost signal is carried as the derived scalar :attr:`action_divergence`
  (``‖certified - proposed‖``), not the policy's proposed vector.

Closed vocabularies (``layer`` / ``intervention`` / ``reason`` / ``backup_kind``) are carried
as plain strings — exactly as the core's typed stub exposes them — so an *additive* core
outcome never fails an existing consumer (an audit record must degrade loudly, not crash).

The canonical schema is ``schema/safety_verdict.schema.json`` (shipped in-package); this model
mirrors it, guarded by ``scripts/check_model_drift.py``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.hashing import content_hash_json

__all__ = ["VERDICT_VERSION", "SafetyVerdict", "load_schema"]

VERDICT_VERSION: Literal["0.1"] = "0.1"

#: The record field carrying the (non-deterministic, wall-clock) per-tick latency — excluded
#: from the reproducibility digest so the determinism gate stays portable across machines.
_LATENCY_FIELD = "shield_latency_us"


class SafetyVerdict(BaseModel):
    """One tick's certified action + intervention record + reproducibility provenance.

    Field-for-field the trusted core's :class:`~astro_mine.guard._core.Verdict`
    (``certified_action``, ``layer``, ``intervention``, ``reason``, ``constraint_ids`` = the
    core's ``fired``, ``min_barrier_margin``) plus the provenance that makes a safety claim
    reproducible (spec/compiled content hashes, Guard code version, input content hash) and the
    Bench-facing derived metrics (``action_divergence``, ``shield_latency_us``).
    """

    model_config = ConfigDict(extra="forbid")

    verdict_version: Literal["0.1"]
    agent_id: str
    tick: int = Field(ge=0)
    sim_time_s: float
    spec_id: str
    spec_content_hash: str
    compiled_content_hash: str
    guard_code_version: str
    layer: str
    intervention: str
    reason: str
    # Nullable scalar: default ``None`` + ``exclude_none`` on the wire, mirroring the compiled-IR
    # nullable-scalar pattern (astro_mine.guard.spec.ir.KeepOutTerm.radius/offset) so an unset
    # optional round-trips through Protobuf presence exactly.
    backup_kind: str | None = None
    constraint_ids: list[str] = Field(default_factory=list)
    certified_action: list[float] = Field(default_factory=list)
    min_barrier_margin: float
    action_divergence: float = Field(ge=0.0)
    inputs_content_hash: str
    shield_latency_us: float = Field(ge=0.0)

    def provenance(self) -> dict[str, Any]:
        """The deterministic projection of this verdict — every field **except** the wall-clock
        latency — the reproducibility view a golden/determinism gate pins (guard.md §6).

        ``shield_latency_us`` is environment-dependent, so a re-run reproduces the same
        ``provenance()`` (and the same :meth:`content_hash`) even though its measured latency
        differs — the same "hash the deterministic part, record the environment alongside"
        discipline Sim uses for its run envelope (conventions.md §5)."""
        return self.model_dump(mode="json", exclude={_LATENCY_FIELD})

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this verdict's deterministic provenance.

        Over the canonical JSON of :meth:`provenance` (the platform's one content-address
        primitive, :func:`astro_mine.core.hashing.content_hash_json`) so the same seeded run
        reproduces the same verdict provenance hash across machines (RM-P1-GUARD-06 golden gate)."""
        return content_hash_json(self.provenance())


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical SafetyVerdict JSON Schema (shipped in-package)."""
    text = (
        resources.files("astro_mine.guard.audit")
        .joinpath("schema/safety_verdict.schema.json")
        .read_text(encoding="utf-8")
    )
    schema: dict[str, Any] = json.loads(text)
    return schema
