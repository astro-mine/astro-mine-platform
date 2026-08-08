"""Run-time provenance — the shared execution-provenance vocabulary (issue #18).

The run-time counterpart to the build-time artifact ``Provenance`` schemas (SADF,
ObjectiveSpec, plugin manifest). :class:`RunProvenance` is the vocabulary every
reproducible run records — resolved input content hashes, engine/toolchain versions,
per-domain fidelity tiers, seeds, and per-tier error-budget outcomes — so that Sim's
MCAP provenance stamping (RM-P0-SIM-09), Cloud's ``RunContext`` (RM-P0-CLOUD-03), and
the Worlds/Bench bundles all share **one** shape instead of re-deriving it per package
(conventions.md §5; core.md §5). Schema only — Core records provenance, it does not run
anything or compute these values (the input-hash helper is issue #19).

The canonical schema is ``schema/run_provenance.schema.json`` (shipped in-package); the
typed models live in :mod:`astro_mine.core.provenance.model`. No Protobuf wire form yet
— run provenance is a metadata record, not a per-tick message (mirrors the manifest);
a wire form is additive if a consumer needs one.

Public API:

- the document — :class:`RunProvenance`, :class:`RunProvenanceDocument`,
  :class:`ErrorBudgetOutcome`;
- load + validate — :func:`load_run_provenance` / :func:`validate_run_provenance` /
  :func:`load_schema`, with :class:`RunProvenanceError` /
  :class:`RunProvenanceValidationError`.

Backlog: issue #18 — astro-mine-core#18
"""

from __future__ import annotations

from astro_mine.core.provenance import enums, loader, model
from astro_mine.core.provenance.loader import (
    RunProvenanceError,
    RunProvenanceValidationError,
    load_run_provenance,
    load_schema,
    validate_run_provenance,
)
from astro_mine.core.provenance.model import (
    ErrorBudgetOutcome,
    RunProvenance,
    RunProvenanceDocument,
)

__all__ = [
    "ErrorBudgetOutcome",
    "RunProvenance",
    "RunProvenanceDocument",
    "RunProvenanceError",
    "RunProvenanceValidationError",
    "enums",
    "load_run_provenance",
    "load_schema",
    "loader",
    "model",
    "validate_run_provenance",
]
