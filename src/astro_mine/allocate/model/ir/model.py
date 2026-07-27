"""The Allocation IR — the solver-neutral problem representation (RM-P1-ALLOC-01).

The stable internal contract that lets solver backends be true plugins ("model/solver
separation", allocate.md §2 principle 3; §3). A compiled :class:`AllocationIR` is a flat,
content-addressed bag of **decision variables**, **linear constraints**, and **objective
terms** — no physics, no solver assumptions — that CP-SAT (RM-P1-ALLOC-02), a MILP
backend, or an auction lowers without changing problem semantics.

The models mirror the canonical Protobuf wire form (``allocation_ir.proto``,
:mod:`astro_mine.allocate.model.ir.wire`) and the exported JSON Schema
(:mod:`astro_mine.allocate._schema`). Every model is frozen and ``extra="forbid"`` — the
IR is an immutable, content-addressed artifact (allocate.md §5: reproducibility &
versioning), and an unknown field fails loudly at the boundary (core.md principle 7). The
compiler emits every collection in **stable sorted order** (by ``id``) so the wire form
and the content hash are byte-stable (allocate.md §8: deterministic and reproducible).

``ir_version`` is an *independent, append-only* interface version pinned here as a
``Literal`` — versioned **with the package** but decoupled from the package's release
version (allocate.md §5; conventions.md §3), exactly as Core versions its interfaces
apart from the ``astro-mine-core`` package version.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.allocate.enums import (
    ConstraintKind,
    ConstraintSense,
    ObjectiveSense,
    VariableKind,
    VariableSemantic,
)
from astro_mine.core.hashing import content_hash_json

__all__ = [
    "IR_VERSION",
    "AllocationIR",
    "Constraint",
    "ConstraintTerm",
    "DecisionVariable",
    "ObjectiveTerm",
]

#: The Allocation IR schema version — append-only, versioned with the package but
#: decoupled from its release version (allocate.md §5). Bump only on an additive IR change.
IR_VERSION: Literal["0.1.0"] = "0.1.0"


class _IRModel(BaseModel):
    """Base for every IR model: immutable and reject unknown/typo'd fields loudly.

    ``frozen=True`` makes an IR a content-addressable value object; ``extra="forbid"``
    rejects a typo'd or unknown field at the boundary rather than silently dropping it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionVariable(_IRModel):
    """One decision variable of the optimization model.

    ``kind`` is its domain type (binary/integer/continuous/interval); ``lower``/``upper``
    are optional closed bounds (a continuous start-time variable's window envelope, a
    binary variable's ``[0, 1]``). ``semantic`` plus the optional ``task_ref``/``asset_ref``
    are the *allocation-domain* meaning of the variable — the seam by which a returned plan
    is mapped back onto variable values for independent feasibility verification
    (:func:`~astro_mine.allocate.verify_feasible`, the Guard-recheckable oracle). They carry
    no physics and do not bind a backend to any particular encoding.
    """

    id: str
    kind: VariableKind
    lower: float | None = None
    upper: float | None = None
    semantic: VariableSemantic
    task_ref: str | None = None
    asset_ref: str | None = None


class ConstraintTerm(_IRModel):
    """One ``coefficient * variable`` term of a linear constraint's left-hand side."""

    var_ref: str
    coefficient: float


class Constraint(_IRModel):
    """A linear constraint ``sum(coefficient * variable) <sense> rhs``.

    Every constraint family — assignment cover, precedence, time window, and (from
    RM-P1-ALLOC-03) the physics-derived power/comms/terrain budgets — is carried in this
    one structural form so a solver backend lowers it without knowing what it *means*.
    ``kind`` labels the family for explanation/IIS reporting (allocate.md §10) and for a
    backend that has a native global-constraint encoding (CP-SAT no-overlap/cumulative).
    """

    id: str
    kind: ConstraintKind
    terms: list[ConstraintTerm] = Field(default_factory=list)
    sense: ConstraintSense
    rhs: float


class ObjectiveTerm(_IRModel):
    """One ``coefficient * variable`` term of the (linear) objective.

    The realized objective of a plan is exactly ``sum(coefficient * variable_value)`` over
    these terms — the identity :func:`~astro_mine.allocate.verify_feasible` re-derives to
    check a reported objective is honest.
    """

    id: str
    var_ref: str
    coefficient: float


class AllocationIR(_IRModel):
    """The compiled, solver-neutral allocation problem.

    A flat set of ``variables``, ``constraints``, and ``objective_terms`` plus the
    ``objective_sense`` and free-form ``metadata`` (e.g. the originating ``request_id``).
    Emitted by :func:`~astro_mine.allocate.compile_request` in stable sorted order so the
    wire form and :meth:`content_hash` are byte-stable for a fixed request (the
    determinism prerequisite for the seeded golden-plan gate, RM-P1-ALLOC-07).
    """

    ir_version: Literal["0.1.0"] = IR_VERSION
    variables: list[DecisionVariable] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    objective_terms: list[ObjectiveTerm] = Field(default_factory=list)
    objective_sense: ObjectiveSense
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _referential_integrity(self) -> Self:
        var_ids = {v.id for v in self.variables}
        if len(var_ids) != len(self.variables):
            raise ValueError("duplicate variable id in AllocationIR.variables")
        if len({c.id for c in self.constraints}) != len(self.constraints):
            raise ValueError("duplicate constraint id in AllocationIR.constraints")
        if len({o.id for o in self.objective_terms}) != len(self.objective_terms):
            raise ValueError("duplicate objective-term id in AllocationIR.objective_terms")
        for c in self.constraints:
            for t in c.terms:
                if t.var_ref not in var_ids:
                    raise ValueError(
                        f"constraint {c.id!r} references unknown variable {t.var_ref!r}"
                    )
        for o in self.objective_terms:
            if o.var_ref not in var_ids:
                raise ValueError(
                    f"objective term {o.id!r} references unknown variable {o.var_ref!r}"
                )
        return self

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this IR (its immutable identity).

        Over the canonical JSON of the model — the platform's one content-address
        primitive (:func:`astro_mine.core.hashing.content_hash_json`) — so a plan can pin
        the exact problem it solved and two identical IRs hash identically across machines.
        """
        return content_hash_json(self.model_dump(mode="json"))
