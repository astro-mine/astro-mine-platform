"""Run-time provenance v0.1 — typed Pydantic models (issue #18; conventions.md §5).

The **run-time** counterpart to the build-time artifact ``Provenance`` schemas in
:mod:`~astro_mine.core.sadf`, :mod:`~astro_mine.core.objective`, and
:mod:`~astro_mine.core.registry`. Those describe how an *artifact* was produced; this
describes how a *run* executed — the fields every reproducible run must record:

- ``input_hashes`` — the resolved content hashes of every pinned input (world, fleet,
  prospect field, scenario, config), so a run's inputs are exactly identifiable;
- ``engine_versions`` / ``fidelity_tiers`` — which engine ran each domain and at what
  fidelity tier (the sim scheduler's per-domain choice, sim.md §11);
- ``seed`` / ``seeds`` — the master episode seed and any named RNG sub-streams (the
  Sim seed/RNG manager, RM-P0-SIM-01) — determinism's root inputs;
- ``error_budget_outcomes`` — the per-tier error-budget verdicts a multi-fidelity run
  produces (sim.md §9), the thing ``env/model.py`` flagged Core had no schema for.

Consumed by Sim (RM-P0-SIM-09 MCAP provenance stamping), composed by Cloud's
``RunContext`` (RM-P0-CLOUD-03 — image digest + resolved inputs + run id; Cloud owns
that envelope but mirrors *this* vocabulary so Sim/Bench/Cloud round-trip cleanly), and
recorded in the Worlds/Bench content bundles. Schema only — Core neither runs anything
nor computes these values; producers fill them (the input-hash helper is issue #19).

These models mirror the canonical ``schema/run_provenance.schema.json`` (shipped
in-package); a consistency test (``tests/test_provenance_consistency.py``) plus the
drift guard (``scripts/check_model_drift.py``) keep the two aligned.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ErrorBudgetOutcome",
    "RunProvenance",
    "RunProvenanceDocument",
]

RUN_PROVENANCE_VERSION: Literal["0.1"] = "0.1"


class _Model(BaseModel):
    """Base for every run-provenance model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


class ErrorBudgetOutcome(_Model):
    """One error-budget verdict recorded by a run (sim.md §9, §11).

    A multi-fidelity run checks each tracked error (orbital position drift, coupling
    residual, terramechanics residual, …) against its budget and records the outcome.
    ``name`` identifies the budget, ``within_budget`` is the producer's verdict, and the
    optional ``tier``/``metric``/``value``/``tolerance`` carry the context. Core records
    the outcome; it does not decide it (mechanism, not policy — core.md §2)."""

    name: str
    within_budget: bool
    tier: str | None = None
    metric: str | None = None
    value: float | None = None
    tolerance: float | None = None


class RunProvenance(_Model):
    """Run-time provenance: what a run pinned, ran, and produced, so it reproduces.

    Every field is optional so a record can be populated progressively as a run
    proceeds (parity with the build-time ``Provenance`` schemas, which are likewise
    all-optional). ``engine_versions``/``fidelity_tiers``/``seeds`` are keyed by a
    free-form domain/engine/stream name — the closed asset-fidelity vocabulary lives in
    :mod:`astro_mine.core.sadf.enums`, not here."""

    run_id: str | None = None
    input_hashes: list[str] = Field(default_factory=list)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    fidelity_tiers: dict[str, str] = Field(default_factory=dict)
    seed: int | None = None
    seeds: dict[str, int] = Field(default_factory=dict)
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None
    error_budget_outcomes: list[ErrorBudgetOutcome] = Field(default_factory=list)


class RunProvenanceDocument(_Model):
    """Top-level run-provenance document. ``run_provenance_version`` pins the schema minor."""

    run_provenance_version: Literal["0.1"]
    run_provenance: RunProvenance
