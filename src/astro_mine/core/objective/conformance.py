# SPDX-License-Identifier: Apache-2.0
"""ObjectiveSpec contract-test utility (RM-P1-CORE-03).

The consumer-driven check Studio runs at its authoring boundary and Bench runs before
it resolves an objective's metrics — analogous to
:func:`astro_mine.core.policy.check_policy` and
:func:`astro_mine.core.env.check_environment`. It asserts an objective validates
(structural + semantic) and survives a **byte-stable Protobuf round-trip unchanged**,
which is the guarantee Studio depends on: "every produced spec serializes and
re-validates; malformed input is rejected at the boundary, never persisted as valid."

Scope is deliberately the *binding*, not its evaluation: whether a measured value meets
a criterion's ``target`` ± ``tolerance`` is Bench's to decide (optimization/evaluation
live above Core, core.md §3). What this guarantees is that a design-time author and an
operational reader parse the **same** binding from the frozen artifact — the basis for
comparing a Bench-over-Sim score with an Ops-over-View reading (LUNAR-FR-009,
LUNAR-TR-006).
"""

from __future__ import annotations

from astro_mine.core.objective.loader import (
    ObjectiveValidationError,
    load_objective,
    validate_objective,
)
from astro_mine.core.objective.model import ObjectiveDocument
from astro_mine.core.objective.wire import from_wire, to_wire

__all__ = ["ObjectiveContractError", "check_objective"]


class ObjectiveContractError(AssertionError):
    """Raised when an objective violates the ObjectiveSpec contract."""


def check_objective(document: ObjectiveDocument | str | bytes) -> ObjectiveDocument:
    """Assert ``document`` honors the ObjectiveSpec contract; return the validated doc.

    Accepts a typed :class:`~astro_mine.core.objective.model.ObjectiveDocument` or raw
    YAML/JSON text/bytes. Validates it (structural + semantic), then asserts it survives a
    byte-stable Protobuf round-trip unchanged (author → validate → serialize →
    re-validate). Raises :class:`ObjectiveContractError` on any failure.
    """
    try:
        if isinstance(document, ObjectiveDocument):
            validate_objective(document)
            doc = document
        else:
            doc = load_objective(document)
    except ObjectiveValidationError as exc:
        raise ObjectiveContractError(f"objective failed validation: {exc}") from exc

    restored = from_wire(to_wire(doc))
    if restored != doc:
        raise ObjectiveContractError(
            "objective did not survive a Protobuf wire round-trip unchanged"
        )
    return doc
