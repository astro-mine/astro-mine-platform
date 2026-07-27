"""SafetySpec v0.1 — closed vocabularies (Guard-owned safety contract).

Small, closed enums for the declarative safety contract. The ``SafetySpec`` is a
*safety artifact*: like the Core SADF/objective vocabularies (``astro_mine.core.*.enums``)
these grow **only by RFC** and only **additively** — members are append-only and never
removed or repurposed (conventions.md §3; guard.md §5 "additive, RFC-gated"). The
breaking-change CI (``buf breaking`` + ``tests/test_schema_compat.py``) enforces that.

Fail-safe is baked into the vocabulary itself: :class:`OnUncertain` has **no**
``passthrough`` member, so "let the policy's action through unchecked" is not even
expressible — the absence of a positive safety certificate can only resolve to a
verified safe action (guard.md §2 principle 4, §9.1).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ConstraintKind",
    "GeometryKind",
    "OnUncertain",
    "PredicateOp",
    "SignalSource",
    "TemporalOp",
]


class ConstraintKind(StrEnum):
    """The kind of hard constraint a :class:`~astro_mine.guard.spec.model.Constraint`
    declares — the discriminant of the constraint tagged union (guard.md §3, §5).

    Covers the geometric, budget, kinematic, and temporal families the safety core
    enforces. New kinds are an additive, RFC-gated schema change (the spec is a safety
    contract). ``coord``/latency-aware and per-regime kinds are deferred (RM-P1-GUARD-01
    out of scope)."""

    KEEP_OUT = "keep_out"
    POWER_FLOOR = "power_floor"
    ENERGY_FLOOR = "energy_floor"
    THERMAL_CEILING = "thermal_ceiling"
    THERMAL_FLOOR = "thermal_floor"
    TORQUE_CEILING = "torque_ceiling"
    KINEMATIC_LIMIT = "kinematic_limit"
    TEMPORAL = "temporal"


class PredicateOp(StrEnum):
    """The comparison operator of an atomic predicate ``signal <op> threshold``.

    A predicate's *robustness* (the margin to violation the STL/MTL monitor tracks) is
    ``threshold - signal`` or ``signal - threshold`` depending on the operator; the
    compiler lowers each into a signed half-space term so the safe side is unambiguous.
    Equality/inequality are deliberately excluded — a hard safety constraint is a
    one-sided bound, never an exact match."""

    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class TemporalOp(StrEnum):
    """The node kind of an :class:`~astro_mine.guard.spec.model.STLFormula` AST node —
    the discriminant of the STL/MTL formula tree (guard.md §4, §9.2).

    A **structured AST**, not a string DSL: reviewable and analyzable with no parser
    dependency. ``predicate`` is the atomic leaf (``signal <op> threshold``); ``not`` /
    ``and`` / ``or`` are boolean combinators; ``always`` / ``eventually`` / ``until`` are
    the bounded temporal operators. Every temporal operator carries a **finite** interval
    ``[lo, hi]`` seconds — an unbounded operator has no statically-bounded history window
    and is rejected by the loader (fail-safe; guard.md §2 principle 6)."""

    PREDICATE = "predicate"
    NOT = "not"
    AND = "and"
    OR = "or"
    ALWAYS = "always"
    EVENTUALLY = "eventually"
    UNTIL = "until"


class OnUncertain(StrEnum):
    """The safe action a constraint resolves to when it cannot be positively certified —
    the fail-safe selector (guard.md §2 principle 4, §9.1).

    Every value is a *verified safe action*; there is deliberately **no** ``passthrough``
    member, so "fail open" is not representable. Defaults to :attr:`FALLBACK` (hand control
    to the simplex backup controller). :attr:`HOLD` freezes/brakes in place; :attr:`SAFE_STATE`
    retreats to a named safe state (e.g. a charging pose). The runtime enforcement of these is
    RM-P1-GUARD-02; the schema only records the intent, and the compiler guarantees the
    resolved behavior is never passthrough."""

    FALLBACK = "fallback"
    HOLD = "hold"
    SAFE_STATE = "safe_state"


class SignalSource(StrEnum):
    """Where a :class:`~astro_mine.guard.spec.model.SignalRef` is resolved from — the
    *abstract* constraint source, by key/path, never a sibling import (guard.md §5, §6).

    Actual value resolution (a Worlds keep-out raster, a Fleet SADF budget) is deferred to
    RM-P1-GUARD-04; here a signal only names its origin so the compiler and the future
    resolver agree on provenance. ``observation`` is a Core Environment observation channel;
    ``sadf`` is a Fleet SADF budget path (e.g. ``power.floor_w``); ``worlds`` is a Worlds
    keep-out / terrain field; ``derived`` is computed from other signals."""

    OBSERVATION = "observation"
    SADF = "sadf"
    WORLDS = "worlds"
    DERIVED = "derived"


class GeometryKind(StrEnum):
    """The shape of a :class:`~astro_mine.guard.spec.model.KeepOutVolume` — the discriminant
    of the keep-out geometry union (guard.md §3 keep-out volumes).

    ``box`` is anchored on the Core :class:`~astro_mine.core.messages.model.Volume`
    (axis-aligned, frame-explicit); ``sphere`` and ``half_space`` are Guard-local barrier
    primitives the compiler lowers to precomputed barrier terms."""

    BOX = "box"
    SPHERE = "sphere"
    HALF_SPACE = "half_space"
