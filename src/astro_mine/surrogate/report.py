"""``ErrorReport`` — the machine-readable, calibrated error a surrogate carries.

*The error is the product* (surrogate.md §2, principle 1): a surrogate ships with a
calibrated bound or it does not ship. ``ErrorReport`` is that bound made first-class —
the artifact [Sim](sim.md)'s multi-fidelity scheduler *consumes* to admit or reject a
learned fidelity tier for a given task tolerance (surrogate.md §3, §6; ``LUNAR-TR-002``).
Every field is a machine-consumable number, never prose (surrogate.md §6: "bounds must
be machine-consumable and calibrated, not prose").

Pydantic v2 + JSON Schema (:mod:`astro_mine.surrogate._schema`) for authoring/validation,
plus a canonical Protobuf wire form (:mod:`astro_mine.surrogate.wire`) so a cross-language
scheduler can read it (surrogate.md §5). Models are frozen and ``extra="forbid"`` — an
``ErrorReport`` is an immutable, content-addressed artifact (surrogate.md §5: "never
overwritten — a retrain produces a new version"), and an unknown field fails loudly at the
boundary (core.md principle 7).

The type is **domain-generic**: the per-channel error is typed continuous *or*
categorical, the trust region is a declared box over any named input domain (an
excavation state space *or* an illumination field-query space), the oracle is any
content-addressed high-fidelity producer (Sim *or* Worlds), and the autoregressive
``rollout`` error is an optional facet a single-shot field surrogate simply omits.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.core.hashing import content_hash_json
from astro_mine.surrogate.enums import ChannelKind, PhysicsDomain

__all__ = [
    "Bound",
    "CategoricalMetrics",
    "ChannelError",
    "ContinuousMetrics",
    "CoveragePoint",
    "ErrorReport",
    "OracleRef",
    "RolloutError",
    "SubstitutionPolicy",
    "TailBehavior",
    "TrustRegion",
]


class _Model(BaseModel):
    """Base for every error-report model: immutable, and reject unknown fields loudly.

    ``frozen=True`` makes an ``ErrorReport`` a content-addressable value object (the
    sibling-spec idiom — Prospect/Worlds spec models); ``extra="forbid"`` rejects a
    typo'd or unknown field at the boundary rather than silently dropping it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class CoveragePoint(_Model):
    """One point of a calibration/reliability curve: nominal vs. empirical coverage.

    Calibration operationalized (surrogate.md §10): a ``nominal`` 0.9 interval whose
    ``empirical`` coverage on held-out ground truth is ~0.9 is calibrated; an
    over-confident surrogate (empirical << nominal) fails the promotion gate. For a
    categorical channel this is a confidence-reliability point (predicted probability vs.
    observed frequency).
    """

    nominal: float = Field(ge=0.0, le=1.0)
    empirical: float = Field(ge=0.0, le=1.0)


class TailBehavior(_Model):
    """The tail of a continuous channel's absolute-error distribution.

    Not just mean/RMSE: a surrogate that under-reports its tail is the epistemic-safety
    failure mode (surrogate.md §9). Absolute error in the channel's unit.
    """

    p95_abs_error: float = Field(ge=0.0)
    p99_abs_error: float = Field(ge=0.0)
    max_abs_error: float = Field(ge=0.0)


class ContinuousMetrics(_Model):
    """Per-channel error for a continuous output channel, in the channel's physical unit.

    ``unit`` (SI, per core.md principle 8) makes ``rmse`` and the tail comparable to a
    Sim ``FidelityPolicy`` per-channel error budget. ``coverage`` is the calibration
    curve; ``tail`` bounds the extremes.
    """

    unit: str
    rmse: float = Field(ge=0.0)
    coverage: list[CoveragePoint] = Field(min_length=1)
    tail: TailBehavior


class CategoricalMetrics(_Model):
    """Per-channel error for a categorical output channel (e.g. ``lit|penumbra|shadow``).

    ``accuracy`` on held-out ground truth plus a ``reliability`` curve (predicted
    confidence vs. observed correctness) — the categorical analogue of interval coverage,
    so an over-confident classifier fails the same calibration gate.
    """

    classes: list[str] = Field(min_length=2)
    accuracy: float = Field(ge=0.0, le=1.0)
    reliability: list[CoveragePoint] = Field(min_length=1)


class ChannelError(_Model):
    """One output channel's error, typed continuous *or* categorical.

    ``kind`` selects which metrics block is present; the validator enforces exactly one,
    so a channel can never be both or neither. Carried as an explicit ``kind`` +
    optional blocks (rather than a bare union) so the Protobuf wire form round-trips
    through ``json_format`` unambiguously.
    """

    channel: str
    kind: ChannelKind
    continuous: ContinuousMetrics | None = None
    categorical: CategoricalMetrics | None = None

    @model_validator(mode="after")
    def _exactly_one_metrics_block(self) -> Self:
        if self.kind is ChannelKind.CONTINUOUS:
            if self.continuous is None or self.categorical is not None:
                raise ValueError(
                    f"channel {self.channel!r} is continuous but must carry exactly a "
                    "`continuous` metrics block"
                )
        elif self.categorical is None or self.continuous is not None:
            raise ValueError(
                f"channel {self.channel!r} is categorical but must carry exactly a "
                "`categorical` metrics block"
            )
        return self


class Bound(_Model):
    """A closed interval ``[low, high]`` on one named input dimension."""

    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.high < self.low:
            raise ValueError(f"bound high ({self.high}) is below low ({self.low})")
        return self


class TrustRegion(_Model):
    """The declared, validated input domain a surrogate's bounds hold over.

    A box over named input dimensions — an excavation state space (soil cohesion, tool
    depth, …) *or* a field-query space (northing, easting, epoch_s, sun_elevation_deg).
    ``in_domain``/OOD are defined against this box: a query outside it must raise the OOD
    flag, never a confident extrapolation (surrogate.md principle 3: "Out-of-distribution
    silence is forbidden").
    """

    bounds: dict[str, Bound] = Field(min_length=1)


class OracleRef(_Model):
    """The content-addressed high-fidelity producer the error was measured against.

    Generalizes surrogate.md's "source Sim solver" so the *same* report serves a
    Sim-oracle excavation surrogate and a Worlds-oracle illumination surrogate: quality
    is always measured against a named high-fidelity producer, never against the
    surrogate itself (surrogate.md principle 4). ``config_hash`` pins the solver/model
    configuration for reproducibility.
    """

    producer: str
    producer_version: str
    config_hash: str | None = None


class SubstitutionPolicy(_Model):
    """The surrogate's *recommended* admission policy — advisory input to Sim's scheduler.

    The surrogate proposes; Sim decides (surrogate.md §1: "Surrogate produces a fidelity
    tier; Sim decides when to use it"). ``recommended_error_budget`` maps each output
    channel to the tolerance within which substituting this tier is recommended;
    ``escalate_on_ood`` asks the scheduler to fall back to high fidelity on an
    out-of-trust-region query.

    **What the budget is a bound on, and why saying so matters.** It bounds the **maximum
    absolute deviation across the predicted field at one step** — the same statistic the
    consumer enforces (Sim re-validates with ``abs(surrogate - reference).max()`` over the
    bed's particles). That was left unstated once, and the ambiguity was the whole of
    surrogate#21: the excavation tier declared a budget derived from an **RMSE** while Sim
    checked a **max**, so the number could never be satisfied — an RMSE bounds nothing, half
    the field exceeds it by construction, and the max of a 90-particle bed sits ~2.5-3x
    higher. The tier escalated on its first re-validation and the benchmark it was built for
    could not produce a claim.

    A producer that calibrates this against a mean-like statistic is not declaring a bound.
    """

    recommended_error_budget: dict[str, float] = Field(min_length=1)
    escalate_on_ood: bool = True
    #: The autoregressive rollout horizon (in steps) the budget was calibrated to hold at.
    #:
    #: A step surrogate feeds its own output back as its next input, so its error compounds over a
    #: rollout — a bound that holds for one step need not hold for four. The budget is meaningless
    #: without the horizon it was measured at, and a consumer that grades the tier over a *longer*
    #: rollout than this checks a bound the producer never made (surrogate#23). Sim honours this
    #: by re-anchoring to ground truth at least this often, and refusing a coarser cadence. ``1`` is
    #: the degenerate single-step case (a field surrogate, or a step tier graded every tick).
    budget_horizon_steps: int = Field(default=1, ge=1)


class RolloutError(_Model):
    """Autoregressive long-horizon error — an *optional* facet (step surrogates only).

    A dynamical-step surrogate accumulates error over an autoregressive rollout, so it
    reports per-horizon RMSE (surrogate.md §8). A single-shot field surrogate has no
    rollout and omits this entirely.
    """

    horizon_steps: int = Field(ge=1)
    rmse_by_horizon: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _horizon_length_matches(self) -> Self:
        if len(self.rmse_by_horizon) != self.horizon_steps:
            raise ValueError(
                f"rmse_by_horizon has {len(self.rmse_by_horizon)} entries but "
                f"horizon_steps is {self.horizon_steps}"
            )
        return self


class ErrorReport(_Model):
    """The calibrated, per-channel error a surrogate carries — Sim's admission artifact.

    Identity (``surrogate_name``/``surrogate_version``) is what Sim stamps into its run
    provenance and per-tier Parquet report when it uses the tier (sim.md §5). The
    ``channels`` are the per-channel error distribution (typed continuous/categorical);
    ``trust_region`` bounds where those numbers hold; ``validation_dataset_hash`` +
    ``oracle`` pin the held-out ground truth they were measured against;
    ``substitution_policy`` is the surrogate's admission recommendation; ``rollout`` is
    the optional long-horizon facet.
    """

    surrogate_name: str
    surrogate_version: str
    domain: PhysicsDomain
    channels: list[ChannelError] = Field(min_length=1)
    trust_region: TrustRegion
    validation_dataset_hash: str
    oracle: OracleRef
    substitution_policy: SubstitutionPolicy
    rollout: RolloutError | None = None

    @model_validator(mode="after")
    def _channels_unique(self) -> Self:
        names = [c.channel for c in self.channels]
        if len(names) != len(set(names)):
            raise ValueError("duplicate channel name in ErrorReport.channels")
        return self

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this report (its immutable identity).

        Over the canonical JSON of the model — the platform's one content-address
        primitive (:func:`astro_mine.core.hashing.content_hash_json`) — so the manifest
        can reference the report by hash and two identical reports hash identically
        across machines.
        """
        return content_hash_json(self.model_dump(mode="json"))
