# SPDX-License-Identifier: Apache-2.0
"""The constraint compiler — lower a validated SafetySpec into the compiled IR.

``compile_spec`` is the deterministic lowering (RM-P1-GUARD-01): a validated
:class:`~astro_mine.guard.spec.model.SafetyDocument` in, a
:class:`~astro_mine.guard.spec.ir.CompiledSafetyModel` out. "The property enforced is
exactly the property reviewed" (guard.md §9.3): the compiler is a pure function — every
list is sorted, every string signal key is resolved to an integer index, and every float
coefficient is precomputed — so two lowerings of the same spec are byte-for-byte identical
(the golden/determinism gate).

Two disciplines are baked in:

- **Fail-safe re-check.** ``compile_spec`` re-validates its input (so an invalid spec can
  never be lowered) and, during lowering, asserts every temporal operator carries a finite
  interval. Absence of a bound is a hard error, never a silent pass (guard.md §2 principle 4).
- **Static-bounds analysis.** The lowering computes the :class:`ResourceBounds`
  — the worst-case predicate-slot / monitor / term / history-window counts — and rejects any
  construct it cannot statically bound. Everything a pre-allocating safety core needs is known
  at compile time, so "no hot-path allocation" is a compile-time property (RM-P1-GUARD-01).
"""

from __future__ import annotations

import math

from astro_mine.guard.spec.enums import (
    ConstraintKind,
    GeometryKind,
    PredicateOp,
    TemporalOp,
)
from astro_mine.guard.spec.ir import (
    COMPILED_VERSION,
    ActionLimits,
    CompiledAdmissibleDirectives,
    CompiledNode,
    CompiledSafePose,
    CompiledSafetyModel,
    KeepOutTerm,
    MonitorAutomaton,
    PredicateAtom,
    PredicateTable,
    ResourceBounds,
    ScalarBound,
)
from astro_mine.guard.spec.loader import SafetySpecError, validate_safety_spec
from astro_mine.guard.spec.model import Constraint, SafetyDocument, SafetySpec, STLFormula

__all__ = ["DEFAULT_SAMPLE_PERIOD_S", "CompileError", "compile_spec"]

#: The default tick period (SI seconds) the monitor history windows are sized against.
DEFAULT_SAMPLE_PERIOD_S = 1.0

#: An atom before the signal key is resolved: (operator, signal_key, threshold).
_UnresolvedAtom = tuple[PredicateOp, str, float]
#: An atom after resolution: (operator, signal_index, threshold) — the predicate-table key.
_RawAtom = tuple[PredicateOp, int, float]


class CompileError(SafetySpecError):
    """Raised when a validated spec cannot be lowered to a statically-bounded IR."""


def _scalar_atoms(c: Constraint) -> list[_UnresolvedAtom]:
    """The (op, signal_key, threshold) predicate atoms a scalar constraint lowers to.

    A floor is ``signal >= threshold`` (``ge``); a ceiling is ``signal <= threshold``
    (``le``). A kinematic limit contributes one atom per bound that is set."""
    kind = c.kind
    if kind == ConstraintKind.POWER_FLOOR:
        assert c.power_floor is not None
        return [(PredicateOp.GE, c.power_floor.signal, c.power_floor.floor_w)]
    if kind == ConstraintKind.ENERGY_FLOOR:
        assert c.energy_floor is not None
        return [(PredicateOp.GE, c.energy_floor.signal, c.energy_floor.floor_j)]
    if kind == ConstraintKind.THERMAL_CEILING:
        assert c.thermal_ceiling is not None
        return [(PredicateOp.LE, c.thermal_ceiling.signal, c.thermal_ceiling.limit_k)]
    if kind == ConstraintKind.THERMAL_FLOOR:
        assert c.thermal_floor is not None
        return [(PredicateOp.GE, c.thermal_floor.signal, c.thermal_floor.limit_k)]
    if kind == ConstraintKind.TORQUE_CEILING:
        assert c.torque_ceiling is not None
        return [(PredicateOp.LE, c.torque_ceiling.signal, c.torque_ceiling.max_nm)]
    if kind == ConstraintKind.KINEMATIC_LIMIT:
        assert c.kinematic_limit is not None
        km = c.kinematic_limit
        atoms: list[_UnresolvedAtom] = []
        if km.max_velocity_mps is not None:
            atoms.append((PredicateOp.LE, km.signal, km.max_velocity_mps))
        if km.max_accel_mps2 is not None:
            atoms.append((PredicateOp.LE, km.signal, km.max_accel_mps2))
        return atoms
    return []  # pragma: no cover - only called for scalar-signal kinds


def _action_limits(spec: SafetySpec) -> ActionLimits:
    """The reviewed kinematic envelope on the **commanded** action — the tightest bound over every
    ``kinematic_limit`` constraint (RM-P1-GUARD-03).

    The same reviewed constraint lowers *twice*, deliberately: to a :class:`ScalarBound` that
    policies the **measured** signal (detect), and — here — to the envelope the shield projects the
    **commanded** setpoint onto (correct). Both readings come from one reviewed clause, so the limit
    a ``VELOCITY``/``POSITION`` command is clamped to is the one the safety engineer signed off, not
    a runtime knob. The *tightest* bound wins, so adding a constraint can only ever narrow the
    envelope (fail-safe under composition).

    **A zero bound is not a command envelope.** The schema admits ``max_velocity_mps: 0`` /
    ``max_accel_mps2: 0`` ("this signal must be zero"), which is a perfectly meaningful bound on a
    *measured* signal — and it keeps its full force as a :class:`ScalarBound`, firing the verified
    backup the instant the measurement moves. But lowering it into the *commanded* envelope would
    collapse the control box to ``{0}``, which disables the recover layer itself: a brake-to-stop
    needs acceleration authority to brake with, and a zero envelope would leave the plant coasting.
    A degenerate bound is therefore **not** promoted to the command envelope (the configured ceiling
    stands, and the shield's keep-out projection and the detect layer both keep their teeth) — the
    strictly safer reading of a contradictory clause."""
    velocities = [
        c.kinematic_limit.max_velocity_mps
        for c in spec.constraints
        if c.kind == ConstraintKind.KINEMATIC_LIMIT
        and c.kinematic_limit is not None
        and c.kinematic_limit.max_velocity_mps is not None
        and c.kinematic_limit.max_velocity_mps > 0.0
    ]
    accels = [
        c.kinematic_limit.max_accel_mps2
        for c in spec.constraints
        if c.kind == ConstraintKind.KINEMATIC_LIMIT
        and c.kinematic_limit is not None
        and c.kinematic_limit.max_accel_mps2 is not None
        and c.kinematic_limit.max_accel_mps2 > 0.0
    ]
    return ActionLimits(
        max_velocity_mps=min(velocities) if velocities else None,
        max_accel_mps2=min(accels) if accels else None,
    )


def _admissible_directives(spec: SafetySpec) -> CompiledAdmissibleDirectives | None:
    """The reviewed ``MODE``/``TASK`` allowlist lowered for the action gate (RFC-0004 Amendment 2).

    A discrete directive carries no numeric command the shield could project, so the trusted core
    certifies it only by **enumeration** — and an admitted directive is re-emitted *untouched*.
    Which is why the grant is lowered from the reviewed, content-addressed contract and not read
    from deployment configuration: the core intersects this with the *configured*
    :class:`~astro_mine.guard.wrap.ActionPolicy` and certifies a directive only if **both** admit
    it.

    **Silence lowers to absence, and absence admits nothing.** Unlike :func:`_action_limits` — where
    an unauthored ceiling leaves the *configured* one standing — an unauthored permission set grants
    **∅**, whatever the configuration says. Both merges are the greatest-lower-bound of (config,
    authored); only the identity of the meet differs (``+∞`` for a ceiling, ``∅`` for a permission
    set). Reading silence as "config stands" here would be fail-open-by-silence.

    Lists are **sorted and deduplicated** so the lowering is byte-for-byte reproducible (the
    golden/determinism gate) and the core's membership scan is over a canonical set."""
    if spec.admissible_directives is None:
        return None
    grant = spec.admissible_directives
    return CompiledAdmissibleDirectives(
        modes=sorted(set(grant.modes)),
        tasks=sorted({t.value for t in grant.tasks}),
    )


def _formula_predicate_atoms(formula: STLFormula) -> list[_UnresolvedAtom]:
    """Every atomic predicate ``(op, signal_key, threshold)`` in a formula tree."""
    if formula.op == TemporalOp.PREDICATE:
        assert (
            formula.cmp is not None and formula.signal is not None and formula.threshold is not None
        )
        return [(formula.cmp, formula.signal, formula.threshold)]
    out: list[_UnresolvedAtom] = []
    for child in formula.args:
        out.extend(_formula_predicate_atoms(child))
    return out


def _keep_out_term(c: Constraint) -> KeepOutTerm:
    """Lower a keep-out constraint to a barrier term with precomputed coefficients."""
    assert c.keep_out is not None
    ko = c.keep_out
    vol = ko.volume
    shape = vol.shape
    if shape == GeometryKind.BOX:
        assert vol.box is not None
        box = vol.box
        return KeepOutTerm(
            constraint_id=c.id,
            on_uncertain=c.on_uncertain,
            shape=shape,
            frame=box.frame,
            frame_ref=box.frame_ref,
            margin_m=ko.margin_m,
            center=[box.center_m.x, box.center_m.y, box.center_m.z],
            half_extents=[
                box.dimensions_m.x / 2.0,
                box.dimensions_m.y / 2.0,
                box.dimensions_m.z / 2.0,
            ],
            collision_pair=ko.collision_pair,
        )
    if shape == GeometryKind.SPHERE:
        assert vol.sphere is not None
        sph = vol.sphere
        return KeepOutTerm(
            constraint_id=c.id,
            on_uncertain=c.on_uncertain,
            shape=shape,
            frame=sph.frame,
            frame_ref=sph.frame_ref,
            margin_m=ko.margin_m,
            center=[sph.center_m.x, sph.center_m.y, sph.center_m.z],
            radius=sph.radius_m,
            collision_pair=ko.collision_pair,
        )
    # half-space: normalize so margin_m is a true metric distance.
    assert vol.half_space is not None
    hs = vol.half_space
    norm = math.sqrt(hs.normal.x**2 + hs.normal.y**2 + hs.normal.z**2)
    if norm == 0.0:  # pragma: no cover - rejected by the loader before compile
        raise CompileError(f"constraint {c.id!r}: half-space normal must be non-zero")
    return KeepOutTerm(
        constraint_id=c.id,
        on_uncertain=c.on_uncertain,
        shape=shape,
        frame=hs.frame,
        frame_ref=hs.frame_ref,
        margin_m=ko.margin_m,
        normal=[hs.normal.x / norm, hs.normal.y / norm, hs.normal.z / norm],
        offset=hs.offset_m / norm,
        collision_pair=ko.collision_pair,
    )


def _compile_node(
    formula: STLFormula,
    signal_index: dict[str, int],
    index_of: dict[_RawAtom, int],
    sample_period_s: float,
) -> tuple[CompiledNode, int, list[int]]:
    """Lower a formula subtree to a resolved, integer-keyed CompiledNode.

    Returns the node, the number of nodes in the subtree (for the node-count bound), and the
    predicate indices it reads."""
    op = formula.op
    if op == TemporalOp.PREDICATE:
        assert (
            formula.cmp is not None and formula.signal is not None and formula.threshold is not None
        )
        idx = index_of[(formula.cmp, signal_index[formula.signal], formula.threshold)]
        return CompiledNode(op=op, predicate_index=idx), 1, [idx]

    child_nodes: list[CompiledNode] = []
    node_count = 1
    predicate_indices: list[int] = []
    for child in formula.args:
        node, count, preds = _compile_node(child, signal_index, index_of, sample_period_s)
        child_nodes.append(node)
        node_count += count
        predicate_indices.extend(preds)

    lo_samples: int | None = None
    hi_samples: int | None = None
    if op in (TemporalOp.ALWAYS, TemporalOp.EVENTUALLY, TemporalOp.UNTIL):
        interval = formula.interval_s
        if interval is None:  # pragma: no cover - rejected by the loader before compile
            raise CompileError(
                f"temporal op {op!r} reached the compiler without a bounded interval — fail-safe"
            )
        lo_samples = math.ceil(interval.lo / sample_period_s)
        hi_samples = math.ceil(interval.hi / sample_period_s)

    node = CompiledNode(
        op=op,
        interval_lo_samples=lo_samples,
        interval_hi_samples=hi_samples,
        args=child_nodes,
    )
    return node, node_count, predicate_indices


def _history_window(node: CompiledNode) -> int:
    """A conservative (never-under-counting) upper bound on the monitor's history length:
    the sum of every bounded temporal horizon in the tree, in samples. Nested temporal
    operators compound along a path, and the total sum bounds any path, so a pre-allocating
    core is never handed too small a buffer."""
    total = node.interval_hi_samples or 0
    for child in node.args:
        total += _history_window(child)
    return total


def compile_spec(
    document: SafetyDocument,
    *,
    sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S,
) -> CompiledSafetyModel:
    """Lower a validated SafetySpec document to its compiled, content-addressed IR.

    Deterministic: the signal table and every atom/term/monitor list is sorted, so two
    compiles of the same spec are byte-identical. Re-validates the input first (fail-safe:
    an invalid spec never reaches lowering) and runs the static-bounds analysis, rejecting
    any construct it cannot statically bound."""
    if sample_period_s <= 0.0:
        raise CompileError(f"sample_period_s must be positive, got {sample_period_s}")
    validate_safety_spec(document)
    spec = document.safety

    # 1. Signal table: all declared signals, sorted -> integer index.
    signal_index = {key: i for i, key in enumerate(sorted(s.key for s in spec.signals))}

    def _resolve(atoms: list[_UnresolvedAtom]) -> list[_RawAtom]:
        return [(op, signal_index[key], threshold) for (op, key, threshold) in atoms]

    # 2. Collect every raw atom (scalar constraints + temporal predicate leaves).
    raw_atoms: set[_RawAtom] = set()
    scalar_by_constraint: dict[str, list[_RawAtom]] = {}
    for c in spec.constraints:
        if c.kind == ConstraintKind.TEMPORAL:
            assert c.temporal is not None
            raw_atoms.update(_resolve(_formula_predicate_atoms(c.temporal.formula)))
        elif c.kind != ConstraintKind.KEEP_OUT:
            resolved = _resolve(_scalar_atoms(c))
            scalar_by_constraint[c.id] = resolved
            raw_atoms.update(resolved)

    # 3. Canonical, sorted, deduplicated predicate table.
    ordered_atoms = sorted(raw_atoms, key=lambda a: (a[1], a[0].value, a[2]))
    index_of: dict[_RawAtom, int] = {atom: i for i, atom in enumerate(ordered_atoms)}
    predicate_table = PredicateTable(
        signals=sorted(signal_index, key=lambda k: signal_index[k]),
        atoms=[
            PredicateAtom(op=op, signal_index=si, threshold=th) for (op, si, th) in ordered_atoms
        ],
    )

    # 4. Scalar bounds (sorted by constraint id then atom index).
    scalar_bounds: list[ScalarBound] = []
    for c in spec.constraints:
        for raw in scalar_by_constraint.get(c.id, []):
            scalar_bounds.append(
                ScalarBound(
                    constraint_id=c.id,
                    on_uncertain=c.on_uncertain,
                    atom_index=index_of[raw],
                )
            )
    scalar_bounds.sort(key=lambda b: (b.constraint_id, b.atom_index))

    # 5. Keep-out terms (sorted by constraint id).
    keep_out_terms = sorted(
        (_keep_out_term(c) for c in spec.constraints if c.kind == ConstraintKind.KEEP_OUT),
        key=lambda t: t.constraint_id,
    )

    # 6. Monitor automata (sorted by constraint id).
    monitors: list[MonitorAutomaton] = []
    for c in spec.constraints:
        if c.kind != ConstraintKind.TEMPORAL:
            continue
        assert c.temporal is not None
        root, node_count, preds = _compile_node(
            c.temporal.formula, signal_index, index_of, sample_period_s
        )
        monitors.append(
            MonitorAutomaton(
                constraint_id=c.id,
                on_uncertain=c.on_uncertain,
                root=root,
                history_window_len=_history_window(root),
                node_count=node_count,
                predicate_indices=sorted(set(preds)),
            )
        )
    monitors.sort(key=lambda m: m.constraint_id)

    # 7. Static resource bounds (the pre-allocation budget).
    resource_bounds = ResourceBounds(
        predicate_slot_count=len(predicate_table.atoms),
        scalar_bound_count=len(scalar_bounds),
        keep_out_term_count=len(keep_out_terms),
        monitor_count=len(monitors),
        max_history_len=max((m.history_window_len for m in monitors), default=0),
        worst_case_term_count=(
            len(predicate_table.atoms) + len(keep_out_terms) + sum(m.node_count for m in monitors)
        ),
    )

    # 8. Safe pose (retreat target): lower the authored Vec3 into a bare position vector in the
    # keep-out frame. Absent when the spec authored none (safe_state then degrades to brake).
    safe_pose = (
        CompiledSafePose(
            frame=spec.safe_pose.frame,
            frame_ref=spec.safe_pose.frame_ref,
            position=[
                spec.safe_pose.position_m.x,
                spec.safe_pose.position_m.y,
                spec.safe_pose.position_m.z,
            ],
        )
        if spec.safe_pose is not None
        else None
    )

    return CompiledSafetyModel(
        compiled_version=COMPILED_VERSION,
        spec_id=spec.id,
        spec_content_hash=document.content_hash(),
        sample_period_s=sample_period_s,
        predicate_table=predicate_table,
        scalar_bounds=scalar_bounds,
        keep_out_terms=keep_out_terms,
        monitors=monitors,
        resource_bounds=resource_bounds,
        # 9. The reviewed kinematic envelope on the commanded action (the shield's projection set).
        action_limits=_action_limits(spec),
        safe_pose=safe_pose,
        # 10. The reviewed MODE/TASK allowlist on the action gate. Absent => the contract certifies
        # NO directive, whatever the deployment configuration grants (RFC-0004 Amendment 2).
        admissible_directives=_admissible_directives(spec),
    )
