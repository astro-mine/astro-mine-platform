"""The guardrail — Mind's mandatory Guard-shield binding (the only output path).

See :mod:`astro_mine.mind.guardrail.shield`: :func:`shield_egress` is the single mechanism
the executive routes every emitted action through (principle 7). RM-P1-MIND-05 adds the
:class:`~astro_mine.mind.guardrail.report.ReportingShield` seam so a shield's intervention
provenance (kind, invoked ``SafetySpec`` clauses, certificate handle) reaches the decision
trace without Mind importing Guard.
"""

from __future__ import annotations

from astro_mine.mind.guardrail.report import InterventionKind, ReportingShield, ShieldReport
from astro_mine.mind.guardrail.shield import Shield, shield_egress

__all__ = [
    "InterventionKind",
    "ReportingShield",
    "Shield",
    "ShieldReport",
    "shield_egress",
]
