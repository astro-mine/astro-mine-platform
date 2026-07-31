"""The guardrail — Mind's mandatory Guard-shield binding (the only output path).

See :mod:`astro_mine.mind.guardrail.shield`: :func:`shield_egress` is the single mechanism
the executive routes every emitted action through (principle 7). RM-P1-MIND-05 adds the
:class:`~astro_mine.core.policy.guardrail.ReportingShield` seam so a shield's intervention
provenance (kind, invoked ``SafetySpec`` clauses, certificate handle) reaches the decision
trace without Mind importing Guard.

The report vocabulary itself — :class:`~astro_mine.core.policy.guardrail.ShieldReport`,
:class:`~astro_mine.core.policy.guardrail.InterventionKind`, and the
:class:`~astro_mine.core.policy.guardrail.ReportingShield` protocol — is Core's, because Guard
implements it and Mind consumes it (conventions.md §3.3). Import it from
:mod:`astro_mine.core.policy`; this package is the *mechanism*, not the contract.
"""

from __future__ import annotations

from astro_mine.mind.guardrail.shield import Shield, shield_egress

__all__ = [
    "Shield",
    "shield_egress",
]
