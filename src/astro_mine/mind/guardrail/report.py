"""The shield-report seam — structured Guard-intervention provenance (RM-P1-MIND-05).

Principle 7 makes Guard the single output path; RM-P1-MIND-05 makes *what the shield did*
observable in the decision trace without Mind ever importing Guard. The Core Policy contract
fixes a shield's return type (``decide(obs, ctx) -> ActionBatch``), so a shield cannot hand
back its verdict inline. This module is the neutral side-channel: a Mind-owned
:class:`ShieldReport` value and a :class:`ReportingShield` protocol a shield MAY implement to
expose the report for its most recent ``decide``.

The executive reads the report — if the bound shield is a :class:`ReportingShield` — after
:func:`~astro_mine.mind.guardrail.shield.shield_egress`, and folds its clause/certificate
provenance into the trace's :class:`~astro_mine.mind.trace.model.ShieldRecord`. Guard's real
``PolicyShield`` (RM-P1-GUARD-03) is made a :class:`ReportingShield` by the companion
``astro-mine-guard`` entry-point shim, which reads Guard's ``SafetyVerdict`` stream
(RM-P1-GUARD-06) — so the binding is a registry swap, not a Mind edit, and the ``clauses``
are the invoked ``SafetySpec`` constraint ids (RFC-0004). A shield that is not a
:class:`ReportingShield` still records faithfully: the executive falls back to
change-detection (did the emitted batch differ from the proposed one).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

__all__ = ["InterventionKind", "ReportingShield", "ShieldReport"]


class InterventionKind(StrEnum):
    """How a shield intervened this tick (Guard's three enforcement modes; guard.md §2,
    RFC-0004). ``shield_edit`` — the proposed action was projected onto the safe set
    (CBF-QP); ``monitor_breach`` — an STL/MTL monitor flagged a specification breach;
    ``backup_activation`` — the verified simplex backup took control. Append-only."""

    SHIELD_EDIT = "shield_edit"
    MONITOR_BREACH = "monitor_breach"
    BACKUP_ACTIVATION = "backup_activation"


@dataclass(frozen=True, slots=True)
class ShieldReport:
    """A shield's structured account of what it did on the most recent ``decide``.

    ``intervened`` is the fact; ``kind`` is which enforcement mode fired (``None`` when the
    action passed unchanged); ``clauses`` are the invoked ``SafetySpec`` constraint ids
    (RFC-0004) so a reviewer can trace an edit back to the reviewed contract; ``certificate``
    is an opaque handle to the safety certificate / ``SafetyVerdict`` record (RM-P1-GUARD-06)
    the audit stream carries. A pass-through report is ``ShieldReport(intervened=False)``.
    """

    intervened: bool
    kind: InterventionKind | None = None
    clauses: tuple[str, ...] = field(default_factory=tuple)
    certificate: str | None = None


@runtime_checkable
class ReportingShield(Protocol):
    """A shield that can surface its most-recent :class:`ShieldReport`.

    Optional over the Core :class:`~astro_mine.core.policy.protocol.Policy` contract: the
    executive checks for it at run time (``isinstance``) and, when present, records the
    report's provenance instead of inferring intervention from a batch diff. Stateful by
    construction — ``report`` returns the report for the last ``decide`` — so a reporting
    shield is instantiated per run (as every plugin is).
    """

    def report(self) -> ShieldReport | None:
        """The report for the most recent ``decide``, or ``None`` if it has not decided."""
        ...
