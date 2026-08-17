# SPDX-License-Identifier: Apache-2.0
"""ObjectiveSpec v0.1 — typed Pydantic models (the shared objective contract).

A first-class Core schema: an objective plus its **binding** to Bench metrics, by
which Studio states a goal, Bench measures it, and Ops/View track progress against
it in both design and operations (core.md §3 "objective contract"; LUNAR-FR-008).
Schema only — the optimization (Studio's trade-study engine) and the evaluation
(Bench/Ops) live above Core.

The **canonical** schema is the hand-authored JSON Schema in
``schema/objective.schema.json`` (shipped in-package); these models mirror it and a
consistency test (``tests/test_objective_consistency.py``) asserts the two agree
until RM-P0-CORE-07 generates one from the other. All quantities are SI; the
binding's ``unit`` is explicit (conventions.md §5).

Deferred (out of scope, RM-P0-CORE-04): the Ledger value-model binding (P3) — a
``MetricBinding`` carries no Ledger field yet; and the ``MissionSpec``/objective
linkage (RFC-0001, P1).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.objective.enums import MetricAggregation, MetricDirection, WindowKind

__all__ = [
    "EvaluationWindow",
    "MetricBinding",
    "ObjectiveDocument",
    "ObjectiveSpec",
    "Provenance",
    "SuccessCriterion",
]

OBJECTIVE_VERSION = "0.1"


class _Model(BaseModel):
    """Base for every objective model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


class Provenance(_Model):
    """Reproducibility provenance (conventions.md §5). Objectives are content-addressed
    so a design-time score and an operational reading reproduce (LUNAR-TR-006)."""

    input_hashes: list[str] = Field(default_factory=list)
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None
    seed: int | None = None


class EvaluationWindow(_Model):
    """How a metric binding is evaluated over time. Without one, a binding is evaluated
    cumulatively over the whole episode. A ``rolling`` window expresses rate/sustained
    objectives ("10 t per lunar day") and requires ``duration_s``; ``cumulative`` and
    ``per_phase`` forbid it (enforced in the loader). Time-windowing semantics are the
    metric's job (Bench); this declares the intent."""

    kind: WindowKind
    duration_s: float | None = Field(default=None, gt=0.0)


class MetricBinding(_Model):
    """The objective->metric binding — the load-bearing contract.

    Binds one success criterion to a Bench metric with an explicit, quantitative
    target and tolerance (acceptance: "binds each success criterion to a Bench metric
    with target + tolerance"). ``metric`` is the Bench metric key (resolved by Bench,
    not Core — Core owns the binding shape, not the metric registry)."""

    metric: str
    unit: str
    direction: MetricDirection
    target: float
    tolerance: float = Field(ge=0.0)
    aggregation: MetricAggregation = MetricAggregation.MEAN
    # Optional hard pass/fail threshold, distinct from the (soft) target+tolerance.
    threshold: float | None = None
    # Optional temporal evaluation window (rate / sustained objectives); cumulative by default.
    evaluation_window: EvaluationWindow | None = None
    # NOTE: the Ledger value-model binding (a scalar economic valuation of this metric)
    # is deferred to P3 (RM-P0-CORE-04 out-of-scope) and is intentionally absent here.


class SuccessCriterion(_Model):
    """One measurable criterion of an objective, bound to a metric.

    ``required`` distinguishes a must-meet criterion from a soft/stretch goal;
    ``weight`` supports scalarization in Studio's multi-objective trade-study engine
    (scenario §8.3, LUNAR-FR-010) — optimization lives above Core, this only carries
    the weight. ``deadline_s`` is an optional achieve-by deadline in SI seconds of
    episode/sim time (``sim_time_s`` elsewhere); mission-epoch-relative deadlines are
    the Mission/Phase model's job (RFC-0001, reserved P1)."""

    id: str
    description: str | None = None
    binding: MetricBinding
    required: bool = True
    weight: float | None = Field(default=None, ge=0.0)
    deadline_s: float | None = Field(default=None, gt=0.0)


class ObjectiveSpec(_Model):
    """An objective: an identity plus its measurable success criteria.

    Authored by Studio (optionally via human-reviewed LLM intent capture) and
    consumed by Bench/Ledger/Ops/View (core.md §3). ``scenario_ref`` is an optional
    content reference to the ScenarioSpec the objective is stated against."""

    id: str
    name: str
    description: str | None = None
    scenario_ref: str | None = None
    success_criteria: list[SuccessCriterion] = Field(min_length=1)
    labels: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance | None = None

    def metric_keys(self) -> list[str]:
        """The distinct Bench metric keys this objective binds, in first-seen order.

        The single source of the objective's metric set: Bench resolves each key against
        its metric registry to assemble the scenario's scoring set, and Studio validates
        each against the declared metric vocabulary — both reading *this* binding, never a
        divergent copy (LUNAR-FR-008/009). The metric registry itself lives in Bench;
        Core owns only the binding."""
        seen: dict[str, None] = {}
        for criterion in self.success_criteria:
            seen.setdefault(criterion.binding.metric, None)
        return list(seen)


class ObjectiveDocument(_Model):
    """Top-level objective document. ``objective_version`` pins the schema minor."""

    objective_version: Literal["0.1"]
    objective: ObjectiveSpec
