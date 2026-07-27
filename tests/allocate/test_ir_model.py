"""AllocationIR referential integrity + content addressing (RM-P1-ALLOC-01).

The IR is an immutable, content-addressed artifact: it rejects duplicate ids and dangling
variable references at construction (so any wire-parsed or hand-built IR is well-formed) and
hashes by canonical content.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.allocate import (
    AllocationIR,
    Constraint,
    ConstraintKind,
    ConstraintSense,
    ConstraintTerm,
    DecisionVariable,
    ObjectiveSense,
    ObjectiveTerm,
    VariableKind,
    VariableSemantic,
)


def _var(vid: str) -> DecisionVariable:
    return DecisionVariable(id=vid, kind=VariableKind.BINARY, semantic=VariableSemantic.ASSIGNMENT)


def _linear(cid: str, var_ref: str) -> Constraint:
    return Constraint(
        id=cid,
        kind=ConstraintKind.LINEAR,
        terms=[ConstraintTerm(var_ref=var_ref, coefficient=1.0)],
        sense=ConstraintSense.LE,
        rhs=0.0,
    )


def test_rejects_duplicate_variable_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate variable id"):
        AllocationIR(objective_sense=ObjectiveSense.MAXIMIZE, variables=[_var("x"), _var("x")])


def test_rejects_duplicate_constraint_ids() -> None:
    con = _linear("c", "x")
    with pytest.raises(ValidationError, match="duplicate constraint id"):
        AllocationIR(
            objective_sense=ObjectiveSense.MAXIMIZE, variables=[_var("x")], constraints=[con, con]
        )


def test_rejects_duplicate_objective_term_ids() -> None:
    term = ObjectiveTerm(id="o", var_ref="x", coefficient=1.0)
    with pytest.raises(ValidationError, match="duplicate objective-term id"):
        AllocationIR(
            objective_sense=ObjectiveSense.MAXIMIZE,
            variables=[_var("x")],
            objective_terms=[term, term],
        )


def test_rejects_constraint_referencing_unknown_variable() -> None:
    with pytest.raises(ValidationError, match="references unknown variable"):
        AllocationIR(
            objective_sense=ObjectiveSense.MAXIMIZE,
            variables=[_var("x")],
            constraints=[_linear("c", "ghost")],
        )


def test_rejects_objective_term_referencing_unknown_variable() -> None:
    with pytest.raises(ValidationError, match="references unknown variable"):
        AllocationIR(
            objective_sense=ObjectiveSense.MAXIMIZE,
            variables=[_var("x")],
            objective_terms=[ObjectiveTerm(id="o", var_ref="ghost", coefficient=1.0)],
        )


def test_well_formed_ir_hashes_by_content() -> None:
    ir = AllocationIR(
        objective_sense=ObjectiveSense.MAXIMIZE,
        variables=[_var("x")],
        objective_terms=[ObjectiveTerm(id="o", var_ref="x", coefficient=2.0)],
    )
    assert ir.content_hash().startswith("sha256:")
    assert ir.content_hash() == ir.model_copy(deep=True).content_hash()


def test_ir_is_frozen() -> None:
    ir = AllocationIR(objective_sense=ObjectiveSense.MAXIMIZE)
    with pytest.raises(ValidationError):
        ir.objective_sense = ObjectiveSense.MINIMIZE
