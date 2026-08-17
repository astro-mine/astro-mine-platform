# SPDX-License-Identifier: Apache-2.0
"""Studio-owned artifacts (studio.md §3 "Key abstractions exposed").

``DesignCandidate``/``TradeStudy``/``Campaign`` are **not** Core types — Core owns
``ObjectiveSpec`` (the cross-component contract) and Studio composes the rest locally,
referencing Fleet SADF assets and Hub policies by content hash rather than importing
siblings (studio.md §2 principle 4). ``IntentDraft`` is the deterministic-forms input
(STUDIO-01) that projects into a Core ``ObjectiveSpec``; ``DesignCandidate`` is the
input to the design loop (STUDIO-03) that STUDIO-02 will later *produce*.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from astro_mine.core.objective.enums import MetricDirection
from astro_mine.core.units.model import PlanetaryCRS

from ._base import FrozenStudioModel, StudioModel
from .hashing import content_hash_json
from .provenance import ArtifactProvenance

# --------------------------------------------------------------------------- #
# STUDIO-01 — intent (forms) input
# --------------------------------------------------------------------------- #


class GeoRegion(StudioModel):
    """A body/region an objective is stated against, carrying an **explicit planetary
    CRS** (conventions.md §5) — no implicit Earth/WGS84 assumption. ``crs`` is the Core
    ``PlanetaryCRS`` type; the deterministic-forms path rejects a missing/invalid CRS
    at the boundary (see ``intent.forms``)."""

    name: str
    crs: PlanetaryCRS


class TargetProduct(StudioModel):
    """A "produce X of product at rate R" objective → a soft, weighted success
    criterion bound to a Bench metric with target + tolerance. ``rate_window_s`` set
    means a sustained-rate objective ("10 t per lunar day") → a rolling window."""

    criterion_id: str
    metric: str
    unit: str
    direction: MetricDirection = MetricDirection.HIGHER_BETTER
    target: float
    tolerance: float = Field(ge=0.0)
    rate_window_s: float | None = Field(default=None, gt=0.0)
    weight: float | None = Field(default=None, ge=0.0)


class HardConstraint(StudioModel):
    """A power/thermal/comms/safety hard constraint → a *required* success criterion
    with a pass/fail ``threshold`` (studio.md §3)."""

    criterion_id: str
    metric: str
    unit: str
    direction: MetricDirection = MetricDirection.LOWER_BETTER
    threshold: float


class AssetSelection(StudioModel):
    """A Fleet SADF asset chosen for a swarm, by content reference + count. ``sadf_ref``
    is a content hash / catalog id resolved from Hub, never an imported object."""

    sadf_ref: str
    count: int = Field(ge=1)


class IntentDraft(StudioModel):
    """The full richness captured by the no-LLM forms — a superset of what a Core
    ``ObjectiveSpec`` can hold. The optimization-relevant part (criteria + metric
    bindings) projects into an ``ObjectiveSpec``; the design-space inputs (asset
    inventory-or-budget, region) are Studio workspace state consumed by the trade-study
    engine (STUDIO-02)."""

    id: str
    name: str
    author: str
    region: GeoRegion
    description: str | None = None
    scenario_ref: str | None = None
    products: list[TargetProduct] = Field(default_factory=list)
    constraints: list[HardConstraint] = Field(default_factory=list)
    inventory: list[AssetSelection] = Field(default_factory=list)
    budget: float | None = Field(default=None, ge=0.0)
    labels: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# STUDIO-03 — design loop artifacts
# --------------------------------------------------------------------------- #


class DesignCandidate(FrozenStudioModel):
    """A proposed solution: an SADF swarm composition + infrastructure + a policy stack
    (drawn from Hub by ref), plus the decision-variable vector that produced it
    (STUDIO-02 fills the vector; STUDIO-03 only consumes the candidate)."""

    id: str
    swarm: list[AssetSelection]
    infrastructure: list[str] = Field(default_factory=list)
    policy_refs: dict[str, str] = Field(default_factory=dict)
    decision_vector: dict[str, float] = Field(default_factory=dict)

    def digest(self) -> str:
        return content_hash_json(self.model_dump(mode="json"))


class CandidateScore(FrozenStudioModel):
    """The Bench score of a candidate against an objective. Uncertainty is **shown, not
    hidden** (studio.md §2 principle 7): surrogate/low-fidelity estimates carry an
    explicit per-metric bound alongside the point estimate."""

    objective_hash: str
    metric_scores: dict[str, float]
    metric_uncertainty: dict[str, float] = Field(default_factory=dict)
    aggregate: float
    passed: bool


class EvaluatedCandidate(FrozenStudioModel):
    """A candidate that has been fanned through the design loop and scored, with the
    provenance needed to reproduce it (studio.md §6)."""

    candidate: DesignCandidate
    score: CandidateScore
    seed: int
    world_ref: str
    provenance: ArtifactProvenance

    def digest(self) -> str:
        return content_hash_json(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# STUDIO-02 — design-space exploration artifacts
# --------------------------------------------------------------------------- #


class AssetChoice(StudioModel):
    """One heterogeneous swarm dimension: an asset kind whose count is a decision
    variable bounded by ``[min_count, max_count]``. Variable-cardinality swarms fall out
    of letting each kind's count (including zero) vary independently."""

    sadf_ref: str
    max_count: int = Field(ge=1)
    min_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> AssetChoice:
        if self.min_count > self.max_count:
            raise ValueError(f"min_count {self.min_count} exceeds max_count {self.max_count}")
        return self


class DecisionSpace(StudioModel):
    """The trade-study's design space: the asset kinds whose counts are searched, plus the
    fixed infrastructure and policy stack every candidate carries. The decision vector is
    one integer count per asset choice — the ``designspace.encode`` codec is lossless
    against it (studio.md §11 "design-space encoding")."""

    assets: list[AssetChoice] = Field(min_length=1)
    infrastructure: list[str] = Field(default_factory=list)
    policies: dict[str, str] = Field(default_factory=dict)

    def dimension(self) -> int:
        return len(self.assets)

    def bounds(self) -> list[tuple[int, int]]:
        return [(choice.min_count, choice.max_count) for choice in self.assets]


class TradeStudy(FrozenStudioModel):
    """A reproducible design-space exploration: the evaluated candidates and the
    Pareto-ranked front they yield, plus the provenance to reproduce the front exactly
    (studio.md §3 ``TradeStudy``)."""

    id: str
    objective_hash: str
    backend: str
    #: Who produced the metric values — ``"stand-in/0.1.0"`` for the local deterministic stub, a
    #: sibling's own id once real physics evaluates the candidates. Required and folded into
    #: ``digest()``, so a stand-in-scored study and a physics-scored one are different artifacts and
    #: a reviewer holding either can tell which they have. The parallel is Bench's
    #: ``Scorecard.runner`` (bench.md §2.1): the numbers alone never reveal what made them.
    evaluator: str
    seeds: list[int]
    evaluated: list[EvaluatedCandidate]
    pareto_front: list[str]
    provenance: ArtifactProvenance

    @property
    def front_is_degenerate(self) -> bool:
        """True when every evaluated candidate is on the Pareto front.

        Not a finding: it means no candidate dominates any other, which is what the stand-in
        evaluator produces by construction (every metric is a positive multiple of swarm size). A
        surface must say so rather than draw it as a result."""
        return bool(self.evaluated) and len(self.pareto_front) == len(
            {candidate.candidate.id for candidate in self.evaluated}
        )

    def digest(self) -> str:
        return content_hash_json(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# STUDIO-04 — campaign + Ops hand-off artifacts
# --------------------------------------------------------------------------- #


class CampaignPhase(StudioModel):
    """One phase of a campaign timeline (studio.md §3). ``objective_ref`` optionally binds
    a phase to a per-phase objective by content hash; single-phase campaigns are the
    common lunar case (a single-``surface``-phase Mission is exactly a campaign, RFC-0001)."""

    id: str
    name: str
    objective_ref: str | None = None
    duration_s: float | None = Field(default=None, gt=0.0)


class ContingencyBranch(StudioModel):
    """An alternate phase sequence taken when ``trigger`` fires (studio.md §3)."""

    id: str
    trigger: str
    phases: list[CampaignPhase] = Field(min_length=1)


class Campaign(FrozenStudioModel):
    """A chosen design authored into a timeline with contingency branches — the Core-shaped
    artifact handed to Ops **unchanged** in Phase 2 (studio.md §3, §9 "hand-off, don't
    fork"). Studio-owned (Core has no campaign schema in the pinned interface); frozen and
    content-addressed once handed off."""

    id: str
    name: str
    objective_hash: str
    chosen: EvaluatedCandidate
    phases: list[CampaignPhase] = Field(min_length=1)
    contingencies: list[ContingencyBranch] = Field(default_factory=list)
    trade_study_ref: str | None = None
    #: The evaluator identity of the study this design came from, carried forward so a reviewer
    #: pulling the campaign **by digest alone** can tell what justified it — without also having to
    #: fetch the trade study. ``None`` only when the campaign was authored without one.
    evaluator: str | None = None
    #: The world bundle the chosen design was inspected against, when one was resolved. Records
    #: *which* terrain a reviewer was looking at; ``None`` when the design was never inspected on a
    #: world, which is a legitimate state and is stated rather than implied.
    world_ref: str | None = None
    provenance: ArtifactProvenance

    def digest(self) -> str:
        return content_hash_json(self.model_dump(mode="json"))
