"""ObjectiveSpec + the objective->metric binding — the shared objective contract.

A first-class Core schema: an objective plus its binding to Bench metrics and (later)
the Ledger value model. Authored by Studio, measured by Bench, valued by Ledger, and
tracked by Ops/View. Schema only — optimization and evaluation live above Core
(core.md §3 "objective contract"; LUNAR-FR-008).

The canonical schema is ``schema/objective.schema.json`` (shipped in-package); the
typed models live in :mod:`astro_mine.core.objective.model`, the closed vocabularies
in :mod:`astro_mine.core.objective.enums`.

Public API:

- :func:`load_objective` / :func:`validate_objective` — parse + validate (structural +
  semantic);
- :func:`to_wire` / :func:`from_wire` — byte-stable Protobuf round-trip;
- :func:`to_proto` / :func:`from_proto` — message-level conversion;
- :func:`load_schema` — the canonical JSON Schema as a dict.

Backlog: RM-P0-CORE-04 — https://github.com/astro-mine/astro-mine-core/issues/4
"""

from __future__ import annotations

from astro_mine.core.objective import conformance, enums, model
from astro_mine.core.objective.conformance import ObjectiveContractError, check_objective
from astro_mine.core.objective.enums import MetricAggregation, MetricDirection, WindowKind
from astro_mine.core.objective.loader import (
    ObjectiveError,
    ObjectiveValidationError,
    load_objective,
    load_schema,
    validate_objective,
)
from astro_mine.core.objective.model import (
    EvaluationWindow,
    MetricBinding,
    ObjectiveDocument,
    ObjectiveSpec,
    SuccessCriterion,
)
from astro_mine.core.objective.wire import from_proto, from_wire, to_proto, to_wire

__all__ = [
    "EvaluationWindow",
    "MetricAggregation",
    "MetricBinding",
    "MetricDirection",
    "ObjectiveContractError",
    "ObjectiveDocument",
    "ObjectiveError",
    "ObjectiveSpec",
    "ObjectiveValidationError",
    "SuccessCriterion",
    "WindowKind",
    "check_objective",
    "conformance",
    "enums",
    "from_proto",
    "from_wire",
    "load_objective",
    "load_schema",
    "model",
    "to_proto",
    "to_wire",
    "validate_objective",
]
