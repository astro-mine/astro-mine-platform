# SPDX-License-Identifier: Apache-2.0
"""Decision-trace record model (RM-P1-MIND-01).

The structured record of what a Mind stack decided, tick by tick — the substrate for the
determinism gate (seed + pinned plugins + fixed inputs ⇒ identical trace) and, later, for
plan explanation. Deliberately split from its **serialization**: these frozen dataclasses
are the neutral in-memory shape, and :mod:`astro_mine.mind.trace.canonical` renders them
to canonical JSON for the golden-trace gate. RM-P1-MIND-07 adds an MCAP serializer over
the *same* records and deepens provenance (input content hashes) — a drop-in, not a
rewrite. RM-P1-MIND-05 populates :class:`ShieldRecord` intervention detail.

``to_dict`` produces JSON-ready primitives (the Core :class:`ActionBatch` via its own
``model_dump``); it is the one projection every serializer builds on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from astro_mine.core.messages.model import ActionBatch

__all__ = [
    "DecisionProvenance",
    "DecisionTrace",
    "ShieldRecord",
    "TickRecord",
    "TierDecisionRecord",
]


@dataclass(frozen=True, slots=True)
class TierDecisionRecord:
    """What one tier did this tick: whether it re-decided (``replanned``) and why
    (``trigger`` — the replan-trigger kind that fired, or ``"initial"`` on the first
    decision; ``None`` when the tier acted on its still-valid cached decision), and whether
    its configured fallback took over (``fallback_used``). ``note`` annotates a
    degrade-not-collapse event (RM-P1-MIND-06): ``comms_stale_hold`` (held a stale plan under
    comms loss — act-while-stale), ``comms_recovered`` (reconciled on recovery), or
    ``coord_yield`` (yielded a task to a neighbor in decentralized conflict resolution)."""

    role: str
    replanned: bool
    trigger: str | None
    fallback_used: bool
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "replanned": self.replanned,
            "trigger": self.trigger,
            "fallback_used": self.fallback_used,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class ShieldRecord:
    """The mandatory shield stage's record for this tick (RM-P1-MIND-05): which shield
    plugin ran, whether it ``intervened``, and — when the shield is a
    :class:`~astro_mine.core.policy.guardrail.ReportingShield` — the intervention
    provenance: ``kind`` (shield-edit / monitor-breach / backup-activation), the invoked
    ``SafetySpec`` ``clauses`` (RFC-0004), and an opaque ``certificate`` handle into the
    ``SafetyVerdict`` audit stream (RM-P1-GUARD-06). The reference pass-through shield never
    intervenes and reports no clauses; Guard's real ``PolicyShield`` populates them."""

    plugin: str
    intervened: bool
    kind: str | None = None
    clauses: tuple[str, ...] = ()
    certificate: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin": self.plugin,
            "intervened": self.intervened,
            "kind": self.kind,
            "clauses": list(self.clauses),
            "certificate": self.certificate,
        }


@dataclass(frozen=True, slots=True)
class TickRecord:
    """One executive tick: the per-tier decisions, the shield stage, and the emitted
    (post-shield) :class:`ActionBatch` — the action that actually became the Environment
    API input, so the trace is a faithful record of the single output path."""

    tick: int
    sim_time_s: float
    seed: int | None
    tiers: tuple[TierDecisionRecord, ...]
    shield: ShieldRecord
    action_batch: ActionBatch
    comms_denied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "sim_time_s": self.sim_time_s,
            "seed": self.seed,
            "comms_denied": self.comms_denied,
            "tiers": [t.to_dict() for t in self.tiers],
            "shield": self.shield.to_dict(),
            "action_batch": self.action_batch.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class DecisionProvenance:
    """The reproducibility provenance of a decision trace (conventions.md §5): the pinned
    plugin set (name → version), the Core interface versions the stack composed against, the
    run seed, and the **input content hashes** (RM-P1-MIND-07) — the content-addressed
    identities of the stack's inputs (stack spec, SADF, belief snapshot, comms model, ONNX
    policy artifacts), keyed by name. Every field is stable given a pinned stack + fixed
    inputs, so the trace is byte-comparable against a golden (the determinism gate)."""

    plugin_versions: Mapping[str, str]
    core_interface_versions: Mapping[str, str]
    seed: int | None
    input_hashes: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_versions": dict(self.plugin_versions),
            "core_interface_versions": dict(self.core_interface_versions),
            "seed": self.seed,
            "input_hashes": dict(self.input_hashes),
        }


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """A full episode's decision trace: the composed stack's id, its provenance, and the
    per-tick records — the artifact the determinism gate compares against a golden."""

    stack_id: str
    provenance: DecisionProvenance
    ticks: tuple[TickRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack_id": self.stack_id,
            "provenance": self.provenance.to_dict(),
            "ticks": [t.to_dict() for t in self.ticks],
        }
