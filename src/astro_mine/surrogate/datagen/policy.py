"""``SamplingPolicy`` — the declarative spec for how ``datagen`` samples the oracle (RM-P1-SURR-03).

The offline **build** loop's experiment design, made a first-class, content-addressed artifact
(surrogate.md §3 "Key abstractions exposed"; §8 "active learning ... Sobol/Latin-hypercube
design"). A policy declares the excavation-parameter box (``parameter_bounds``), the space-filling
``design`` for the initial sweep, and the active-learning ``acquisition`` that resamples where the
surrogate's residual uncertainty is highest — so a surrogate can record the exact sampling policy
(by hash) it was generated under (§5 "Provenance ... hyperparameters").

Frozen + ``extra="forbid"`` (the sibling-spec idiom of
:class:`~astro_mine.surrogate.report.ErrorReport`): a policy is an immutable value object whose
:meth:`content_hash` pins it in a surrogate's provenance. This module imports only Core + Pydantic —
never numpy/scipy — so the policy stays in the contract
layer; the numpy design generation lives in :mod:`astro_mine.surrogate.datagen.design`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.core.hashing import content_hash_json
from astro_mine.surrogate.report import Bound

__all__ = ["AcquisitionKind", "DesignKind", "SamplingPolicy"]


class DesignKind(StrEnum):
    """The space-filling design for the initial sweep — append-only (the platform enum idiom).

    ``SOBOL``/``LHS`` are low-discrepancy/stratified space-filling designs (surrogate.md §8);
    ``GRID`` is a full-factorial lattice for a small, exhaustive sweep. Members grow only by
    addition so a persisted :class:`SamplingPolicy`'s ``string`` value stays valid across versions.
    """

    SOBOL = "sobol"
    LHS = "lhs"
    GRID = "grid"


class AcquisitionKind(StrEnum):
    """The active-learning acquisition that ranks a candidate pool for the next round.

    ``MAX_UNCERTAINTY`` samples where the surrogate's residual/ensemble uncertainty is highest
    (surrogate.md §8 "sample where surrogate residual uncertainty is highest, not uniformly");
    ``RANDOM`` is the uniform baseline the acquisition is measured against. Append-only.
    """

    MAX_UNCERTAINTY = "max_uncertainty"
    RANDOM = "random"


class SamplingPolicy(BaseModel):
    """Declarative spec for a ``datagen`` sweep: the box, the design, and the acquisition.

    ``parameter_bounds`` is the named excavation-parameter box (reusing
    :class:`~astro_mine.surrogate.report.Bound`, which already enforces ``low <= high``) the sweep
    samples over and that a surrogate later declares as its trust region. ``design`` +
    ``n_initial`` fix the space-filling initial sweep; ``acquisition`` + ``n_rounds`` +
    ``n_per_round`` + ``pool_size`` fix the active-learning loop that follows (each round scores a
    ``pool_size`` candidate pool and labels the top ``n_per_round``). ``seed`` makes the whole
    design reproducible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter_bounds: dict[str, Bound] = Field(min_length=1)
    design: DesignKind = DesignKind.SOBOL
    n_initial: int = Field(ge=1)
    acquisition: AcquisitionKind = AcquisitionKind.MAX_UNCERTAINTY
    n_rounds: int = Field(default=0, ge=0)
    n_per_round: int = Field(default=1, ge=1)
    pool_size: int = Field(ge=1)
    seed: int = 0

    @model_validator(mode="after")
    def _pool_covers_a_round(self) -> Self:
        # The acquisition selects ``n_per_round`` from a ``pool_size`` candidate pool, so the pool
        # must be at least a round wide — a positive-counts sanity check.
        if self.pool_size < self.n_per_round:
            raise ValueError(
                f"pool_size ({self.pool_size}) must be >= n_per_round ({self.n_per_round})"
            )
        return self

    @property
    def param_names(self) -> tuple[str, ...]:
        """The ordered parameter names — the column order of a generated design ``(C, P)``."""
        return tuple(self.parameter_bounds)

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this policy (its immutable identity).

        Over the platform's one content-address primitive
        (:func:`astro_mine.core.hashing.content_hash_json`), so the sampling policy a surrogate was
        generated under is pinned by hash in its provenance (surrogate.md §5).
        """
        return content_hash_json(self.model_dump(mode="json"))
