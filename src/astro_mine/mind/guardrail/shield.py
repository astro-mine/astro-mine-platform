"""The mandatory Guard-shield binding — Mind's single output path (RM-P1-MIND-01).

Principle 7: *Guard-wrapped output is the only output.* This module owns the one egress
mechanism the executive routes **every** candidate action through before it becomes an
Environment API action. Because a shielded policy *is* a policy (Guard's ``PolicyShield``
implements the Core Policy/Planner contract; RFC-0004), the shield is applied as the
outermost composition stage: the proposed :class:`ActionBatch` is threaded into the
shield's ``DecisionContext.upstream`` and the shield's output is what leaves Mind.

The architectural guarantee lives in the executive: the tiers are pure
``decide(obs, ctx) → batch`` policies and never receive the ``Environment`` handle, so no
tier can emit directly — the executive is the only stepper and :func:`shield_egress` is the
only path from a proposed batch to an emitted one. RM-P1-MIND-05 swaps the reference
pass-through shield for Guard's real ``PolicyShield`` (via the registry) and enriches the
:class:`~astro_mine.mind.trace.model.ShieldRecord` with clause/certificate provenance; the
egress mechanism here is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol, runtime_checkable

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.guardrail import ReportingShield
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.policy.protocol import Policy
from astro_mine.mind.trace.model import ShieldRecord

__all__ = ["Shield", "shield_egress"]


@runtime_checkable
class Shield(Policy, Protocol):
    """A safety shield — a Core :class:`Policy` that wraps another policy's output and
    returns a certified action. A nominal marker (like Core's tier sub-interfaces): it adds
    no methods over ``decide``, documenting the role Guard's ``PolicyShield`` fills. A shield
    that also implements :class:`~astro_mine.core.policy.guardrail.ReportingShield` surfaces
    structured intervention provenance into the trace (RM-P1-MIND-05)."""


def shield_egress(
    shield: Policy,
    observations: Mapping[AgentId, Observation],
    context: DecisionContext,
    proposed: ActionBatch,
    *,
    shield_name: str,
) -> tuple[ActionBatch, ShieldRecord]:
    """Pass ``proposed`` through ``shield`` and return the emitted batch + its record.

    The shield decides against the same observations/context with ``proposed`` as its
    ``upstream`` (the composition seam), so a shield edits, clamps, or replaces the proposed
    actions exactly as it would any upstream tier's output — this is the one path from a
    proposed batch to an emitted one (RM-P1-MIND-05).

    Intervention provenance: if ``shield`` is a
    :class:`~astro_mine.core.policy.guardrail.ReportingShield`, its
    :class:`~astro_mine.core.policy.guardrail.ShieldReport` supplies ``intervened`` and the
    invoked clause/certificate detail (Guard's ``PolicyShield`` reports its ``SafetyVerdict``
    this way). Otherwise the record falls back to change-detection — whether the emitted
    batch differs from the proposed one."""
    emitted = shield.decide(observations, replace(context, upstream=proposed))
    if isinstance(shield, ReportingShield):
        report = shield.report()
        if report is not None:
            return emitted, ShieldRecord(
                plugin=shield_name,
                intervened=report.intervened,
                kind=report.kind.value if report.kind is not None else None,
                clauses=report.clauses,
                certificate=report.certificate,
            )
    return emitted, ShieldRecord(plugin=shield_name, intervened=emitted != proposed)
