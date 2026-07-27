"""The zero-violation oracle: the safe-set predicate the falsification search tries to break.

Given a rollout (:mod:`astro_mine.guard.falsify.rollout`) the oracle decides, per tick, whether the
shield upheld the safety contract (guard.md §9.5; issue #5 acceptance: "zero hard-constraint
violations under adversarial test on the anchor"). It is **independent of the trusted core** — it
re-derives the safe set straight from the ``CompiledSafetyModel``'s keep-out terms and scalar
predicate table, so it is a genuine external check, not a restatement of what the core computed.

Three invariants (a shielded step passes only if all hold):

1. **Safe-set containment (controllable state).** For every keep-out term the plant position stays
   in the safe set: ``barrier(position) ≥ -tol``, where ``tol = u_max·dt²`` is the forward-Euler
   one-step overshoot bound of the double-integrator plant (a discretization allowance of the *test
   harness*, not of the continuous CBF, which holds ``barrier ≥ 0`` exactly). The core's own
   ``min_barrier_margin`` certificate is cross-checked against the same bound.
2. **Never fail open (uncontrollable signals).** A commanded acceleration cannot change the battery
   / thermal / torque / speed signals, so the shield's guarantee is that it never *certifies
   the proposed action while a scalar hard constraint is violated* — it must fall back to a verified
   backup. So: whenever any scalar bound is violated (or a signal is unresolvable / ``NaN``) the
   verdict layer must be ``backup``.
3. **Bounded, finite certified action.** Every certified component is finite and within ``u_max``.

The **unshielded control** is scored by physical breach alone (:func:`control_violations`): with no
shield, the raw proposals drive the position into a keep-out region — the search must find that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from astro_mine.guard.falsify.rollout import RolloutStep
from astro_mine.guard.spec.enums import GeometryKind, PredicateOp
from astro_mine.guard.spec.ir import CompiledSafetyModel, KeepOutTerm

__all__ = [
    "Violation",
    "control_violations",
    "keepout_barrier",
    "scalar_violations",
    "shielded_violations",
]

_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class Violation:
    """One falsification finding: which tick, what kind, and a human-readable detail."""

    index: int
    kind: str
    detail: str


def keepout_barrier(term: KeepOutTerm, position: Sequence[float]) -> float:
    """The keep-out barrier ``h(position)`` for one term — safe iff ``h ≥ margin`` is maintained,
    i.e. this returns *clearance minus margin* so ``h ≥ 0`` is the safe set.

    - **sphere**: ``|p - center| - radius - margin``;
    - **box** (axis-aligned): ``dist_to_box_surface(p) - margin`` (0 inside);
    - **half-space** (safe set ``normal·x + offset ≥ margin``): ``normal·p + offset - margin``.
    """
    p = [float(x) for x in position]
    margin = term.margin_m
    if term.shape == GeometryKind.SPHERE:
        assert term.radius is not None
        return math.dist(p[: len(term.center)], term.center) - term.radius - margin
    if term.shape == GeometryKind.BOX:
        c, e = term.center, term.half_extents
        outside = [max(abs(p[i] - c[i]) - e[i], 0.0) for i in range(len(c))]
        return math.sqrt(sum(d * d for d in outside)) - margin
    # half-space
    assert term.offset is not None
    dot = sum(term.normal[i] * p[i] for i in range(len(term.normal)))
    return dot + term.offset - margin


def scalar_violations(compiled: CompiledSafetyModel, signals: dict[str, float]) -> list[str]:
    """The constraint ids of every scalar bound the current ``signals`` violate (or cannot resolve).

    Evaluates each :class:`~astro_mine.guard.spec.ir.ScalarBound` against its predicate atom
    ``signal[i] <op> threshold`` — a ``GE`` bound is satisfied when ``value ≥ threshold``, a ``LE``
    bound when ``value ≤ threshold``. A missing or ``NaN`` signal counts as violated (unresolvable ⇒
    the core fails the tick closed)."""
    keys = compiled.predicate_table.signals
    atoms = compiled.predicate_table.atoms
    violated: list[str] = []
    for bound in compiled.scalar_bounds:
        atom = atoms[bound.atom_index]
        value = signals.get(keys[atom.signal_index], math.nan)
        if not math.isfinite(value):
            violated.append(bound.constraint_id)
            continue
        ok = value >= atom.threshold if atom.op == PredicateOp.GE else value <= atom.threshold
        if not ok:
            violated.append(bound.constraint_id)
    return violated


def _keepout_breaches(
    compiled: CompiledSafetyModel, position: Sequence[float], tol: float
) -> list[tuple[str, float]]:
    """Keep-out terms whose barrier dips below ``-tol`` at ``position`` (id, barrier)."""
    return [
        (term.constraint_id, keepout_barrier(term, position))
        for term in compiled.keep_out_terms
        if keepout_barrier(term, position) < -tol
    ]


def _overshoot_tol(u_max: float, dt: float) -> float:
    """The forward-Euler one-step position-overshoot bound of the double integrator."""
    return u_max * dt * dt


def shielded_violations(
    steps: Sequence[RolloutStep],
    compiled: CompiledSafetyModel,
    *,
    u_max: float,
    dt: float,
) -> list[Violation]:
    """Every safety violation the shielded rollout exhibits — **empty means the shield held**.

    Enforces the three invariants (see the module docstring): safe-set containment, never-fail-open,
    and a bounded/finite certified action. A step whose verdict is missing (no sink) fails invariant
    2 vacuously-safe (the containment + finiteness checks still apply)."""
    tol = _overshoot_tol(u_max, dt)
    out: list[Violation] = []
    for step in steps:
        pos = step.state.position
        # (1) safe-set containment on the controllable position
        for cid, barrier in _keepout_breaches(compiled, pos, tol):
            out.append(
                Violation(step.index, "keep_out", f"{cid}: barrier {barrier:.4f} < -{tol:.4f}")
            )
        verdict = step.verdict
        if verdict is not None:
            # cross-check: the core's own certificate never dips below the same bound
            if verdict.min_barrier_margin < -tol:
                out.append(
                    Violation(
                        step.index,
                        "certificate",
                        f"min_barrier_margin {verdict.min_barrier_margin:.4f} < -{tol:.4f}",
                    )
                )
            # (2) never fail open: an unsafe scalar/temporal state must not certify the proposal
            unsafe = scalar_violations(compiled, dict(step.state.signals))
            if unsafe and verdict.layer != "backup":
                out.append(
                    Violation(
                        step.index,
                        "fail_open",
                        f"scalar constraint(s) {unsafe} violated but layer={verdict.layer!r} "
                        f"reason={verdict.reason!r} (expected a backup fallback)",
                    )
                )
        # (3) bounded, finite certified action
        for i, a in enumerate(step.certified_action):
            if not math.isfinite(a) or abs(a) > u_max + _EPS:
                out.append(
                    Violation(step.index, "action", f"certified[{i}]={a} exceeds u_max={u_max}")
                )
    return out


def control_violations(
    steps: Sequence[RolloutStep],
    compiled: CompiledSafetyModel,
    *,
    u_max: float,
    dt: float,
) -> list[Violation]:
    """Physical safe-set breaches of the **unshielded control** — keep-out region entry or a scalar
    bound crossing. This must be non-empty: the uncorrected adversary reaches a violation, proving
    the search is not vacuous."""
    tol = _overshoot_tol(u_max, dt)
    out: list[Violation] = []
    for step in steps:
        for cid, barrier in _keepout_breaches(compiled, step.state.position, tol):
            out.append(
                Violation(step.index, "keep_out", f"{cid}: barrier {barrier:.4f} (inside keep-out)")
            )
        for cid in scalar_violations(compiled, dict(step.state.signals)):
            out.append(Violation(step.index, "scalar", f"{cid}: scalar bound crossed"))
    return out
