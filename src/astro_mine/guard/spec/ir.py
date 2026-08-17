# SPDX-License-Identifier: Apache-2.0
"""CompiledSafetyModel v0.1 — the analyzable, content-addressed compiled IR.

The output of the constraint compiler (:mod:`astro_mine.guard.spec.compiler`): the
serializable, integer-keyed lowering of a validated :class:`~astro_mine.guard.spec.model.SafetySpec`
that the future Rust safety core (RM-P1-GUARD-02, out of scope here) consumes to enforce
the contract. It is itself a **Core-catalogued message** — its own JSON Schema
(``schema/compiled_safety_model.schema.json``), Protobuf wire form
(:mod:`astro_mine.guard.spec.wire`), and content hash.

The IR is deliberately shaped for a **small, deterministic, allocation-free** safety core
(guard.md §2 principles 3, 6; §9.1):

- **Predicate table** — a flat array of atoms ``{op, signal_index, threshold}`` with every
  string signal key resolved to an integer index at compile time. Deduplicated so it is the
  set of predicate *slots* the core pre-allocates.
- **Keep-out / barrier terms** — box / sphere / half-space regions with precomputed
  coefficients + margin, so the runtime evaluates a fixed linear/quadratic form per tick.
- **Monitor automata** — each temporal clause as a bounded-memory online monitor whose
  history-window length (in samples) is **computed at compile time** from the finite interval
  bounds, so the core knows its worst-case ring-buffer size up front.
- **Resource bounds** — the static-bounds analysis result: the worst-case counts (predicate
  slots, monitors, terms, history length) a pre-allocating core needs. The compiler rejects
  any construct it cannot statically bound, so "no hot-path allocation" is a compile-time
  property, not a runtime hope (RM-P1-GUARD-01 acceptance).

The canonical schema is hand-authored; these models mirror it, guarded by the drift check
(``scripts/check_model_drift.py``). All lists are sorted and integer-keyed so the lowering
is byte-for-byte reproducible (the golden/determinism gate).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.hashing import content_hash_json
from astro_mine.core.units import ReferenceFrame
from astro_mine.guard.spec.enums import (
    GeometryKind,
    OnUncertain,
    PredicateOp,
    TemporalOp,
)

__all__ = [
    "COMPILED_VERSION",
    "ActionLimits",
    "CompiledAdmissibleDirectives",
    "CompiledNode",
    "CompiledSafePose",
    "CompiledSafetyModel",
    "KeepOutTerm",
    "MonitorAutomaton",
    "PredicateAtom",
    "PredicateTable",
    "ResourceBounds",
    "ScalarBound",
]

COMPILED_VERSION: Literal["0.1"] = "0.1"


class _Model(BaseModel):
    """Base for every compiled-IR model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


class PredicateAtom(_Model):
    """One atomic predicate ``signal[signal_index] <op> threshold`` — a single evaluatable
    slot the safety core reads each tick. ``signal_index`` indexes
    :attr:`PredicateTable.signals`; string keys are resolved to integers at compile time."""

    op: PredicateOp
    signal_index: int = Field(ge=0)
    threshold: float


class PredicateTable(_Model):
    """The flat, deduplicated predicate table: the sorted signal-key table (index -> key)
    plus the canonical set of :class:`PredicateAtom` slots the core pre-allocates."""

    signals: list[str] = Field(default_factory=list)
    atoms: list[PredicateAtom] = Field(default_factory=list)


class ScalarBound(_Model):
    """A scalar floor/ceiling/torque/kinematic constraint lowered to one predicate slot.

    ``atom_index`` references the canonical :class:`PredicateAtom` in the table; ``constraint_id``
    and ``on_uncertain`` carry the traceability and fail-safe posture through to the core (a
    kinematic constraint with two bounds contributes two :class:`ScalarBound` records sharing
    a ``constraint_id``)."""

    constraint_id: str
    on_uncertain: OnUncertain
    atom_index: int = Field(ge=0)


class KeepOutTerm(_Model):
    """A keep-out region lowered to a barrier term with precomputed coefficients + margin.

    The populated fields depend on ``shape``: ``box`` sets ``center`` (3) + ``half_extents``
    (3); ``sphere`` sets ``center`` (3) + ``radius``; ``half_space`` sets a **unit** ``normal``
    (3) + ``offset`` (the safe set is ``normal · x + offset >= margin_m``). Coefficients are
    precomputed at compile time so the runtime evaluates a fixed form with no allocation."""

    constraint_id: str
    on_uncertain: OnUncertain
    shape: GeometryKind
    frame: str
    margin_m: float
    center: list[float] = Field(default_factory=list)
    half_extents: list[float] = Field(default_factory=list)
    radius: float | None = None
    normal: list[float] = Field(default_factory=list)
    offset: float | None = None
    collision_pair: tuple[str, str] | None = None
    #: Optional typed Core ReferenceFrame sibling of ``frame`` (RFC-0007), carried through the
    #: lowering from the source keep-out geometry. The trusted Rust core validates it
    #: (require_frame) but reads only the geometry coefficients + position for control.
    frame_ref: ReferenceFrame | None = None


class CompiledNode(_Model):
    """One node of a **fully-resolved, integer-keyed, bounded** STL/MTL monitor tree.

    ``op`` is the node kind. A ``predicate`` leaf carries ``predicate_index`` (into
    :attr:`PredicateTable.atoms`); a temporal node (``always``/``eventually``/``until``)
    carries its horizon converted to **sample counts** (``interval_lo_samples`` /
    ``interval_hi_samples``); ``args`` are the child nodes. This is exactly the pre-allocatable
    form the Rust core walks —
    no strings, no unbounded intervals."""

    op: TemporalOp
    predicate_index: int | None = Field(default=None, ge=0)
    interval_lo_samples: int | None = Field(default=None, ge=0)
    interval_hi_samples: int | None = Field(default=None, ge=0)
    args: list[CompiledNode] = Field(default_factory=list)


class MonitorAutomaton(_Model):
    """A temporal clause lowered to a bounded-memory online monitor.

    ``root`` is the resolved formula tree; ``history_window_len`` is the worst-case ring-buffer
    length in samples, **computed at compile time** as the largest temporal horizon in the tree
    — the static memory bound the core pre-allocates. ``predicate_indices`` is the sorted set of
    atom slots the monitor reads; ``node_count`` bounds the per-tick evaluation work."""

    constraint_id: str
    on_uncertain: OnUncertain
    root: CompiledNode
    history_window_len: int = Field(ge=0)
    node_count: int = Field(ge=1)
    predicate_indices: list[int] = Field(default_factory=list)


class ResourceBounds(_Model):
    """The static-bounds analysis result — the worst-case counts a pre-allocating safety core
    needs so nothing on the hot path allocates (guard.md §2 principle 6; RM-P1-GUARD-01
    "no hot-path allocation").

    ``worst_case_term_count`` is the upper bound on distinct things the core evaluates per tick:
    predicate slots + keep-out terms + total monitor nodes."""

    predicate_slot_count: int = Field(ge=0)
    scalar_bound_count: int = Field(ge=0)
    keep_out_term_count: int = Field(ge=0)
    monitor_count: int = Field(ge=0)
    max_history_len: int = Field(ge=0)
    worst_case_term_count: int = Field(ge=0)


class ActionLimits(_Model):
    """The reviewed kinematic envelope on the **commanded** action (RM-P1-GUARD-03).

    The tightest bound over the spec's
    :class:`~astro_mine.guard.spec.model.KinematicLimitConstraint`\\ s. Those constraints already
    lower to :class:`ScalarBound`\\ s that police the *measured* signal (the **detect** layer); this
    is the *same reviewed limit* applied to the **command** (the **correct** layer), so a
    ``POSITION`` / ``VELOCITY`` / ``EFFORT`` setpoint is projected onto the envelope the safety
    engineer signed off — not onto a runtime configuration knob a policy's operator could widen.

    A member is ``None`` when the spec authored no such limit; the trusted core's configured ceiling
    (``u_max`` / ``v_max``) then stands. An authored limit may only ever **tighten** the configured
    one (``min(config, reviewed)``) — configuration cannot loosen the reviewed contract."""

    #: Largest commanded speed ``‖w‖`` (m/s) — the ``VELOCITY`` ball and (via ``·dt``) the
    #: ``POSITION`` step ball.
    max_velocity_mps: float | None = Field(default=None, gt=0.0)
    #: Largest commanded acceleration magnitude (m/s²) — tightens the ``EFFORT`` control box.
    max_accel_mps2: float | None = Field(default=None, gt=0.0)


class CompiledAdmissibleDirectives(_Model):
    """The reviewed ``MODE``/``TASK`` allowlist on the **action gate**, lowered from
    :class:`~astro_mine.guard.spec.model.AdmissibleDirectives` (RFC-0004 Amendment 2).

    A discrete directive carries no numeric command the shield could project, so the trusted core
    certifies it only by **enumeration** — and an admitted directive is re-emitted untouched, which
    is why the grant is part of the reviewed contract rather than of local configuration.

    The core intersects this with the *configured*
    :class:`~astro_mine.guard.wrap.ActionPolicy` **once**, at construction, so the hot path stays
    allocation-free (guard.md §2 principle 6): ``effective = spec ∩ config``. Configuration may only
    ever **narrow** the reviewed grant. **Absent from a compiled model ⇒ the contract admits no
    directive**, whatever the configuration grants — silence is ``∅``, not "config stands" (contrast
    :class:`ActionLimits`, whose absent members leave the configured ceiling standing: the identity
    of a ceiling's meet is ``+∞``, of a permission set's ``∅``).

    ``tasks`` carries Core ``TaskKind`` *values* as plain strings, the IR's convention for every
    closed vocabulary — the vocabulary itself is enforced upstream, at spec-authoring time."""

    #: Pre-certified ``ModeCommand.mode`` names (an open vocabulary: SADF ``loads_by_mode``).
    modes: list[str] = Field(default_factory=list)
    #: Pre-certified ``TaskDirective.task_kind`` values (Core ``TaskKind``, as strings).
    tasks: list[str] = Field(default_factory=list)


class CompiledSafePose(_Model):
    """The authored safe/charging pose lowered to a bare position in the keep-out spatial frame —
    the target the verified retreat (``safe_state``) backup steers toward (RM-P1-GUARD-04).

    ``position`` carries the ``SafePose.position_m`` components resolved in the keep-out ``frame``;
    the trusted core reads only the position, using the first ``spatial_dim`` coordinates. Absent
    from a compiled model whenever the source spec authored no ``safe_pose`` — a ``safe_state``
    fallback then degrades to brake-to-stop (fail-safe, never fail-open).

    ``frame_ref`` is the optional typed Core :class:`ReferenceFrame` sibling of ``frame``
    (RFC-0007)."""

    frame: str
    position: list[float] = Field(default_factory=list)
    frame_ref: ReferenceFrame | None = None


class CompiledSafetyModel(_Model):
    """The compiled, content-addressed safety model the Rust core consumes.

    ``spec_content_hash`` binds this IR to the exact source ``SafetyDocument`` it was lowered
    from (so "the property enforced is exactly the property reviewed"); the IR
    carries its own content hash too. ``sample_period_s`` is the tick period the monitor history
    windows were sized against. ``safe_pose`` is the optional retreat target (RM-P1-GUARD-04);
    ``action_limits`` the reviewed kinematic envelope the shield projects commands onto
    (RM-P1-GUARD-03); ``admissible_directives`` the reviewed ``MODE``/``TASK`` allowlist the action
    gate certifies against (RFC-0004 Amendment 2) — absent ⇒ **no** directive is certifiable."""

    compiled_version: Literal["0.1"]
    spec_id: str
    spec_content_hash: str
    sample_period_s: float = Field(gt=0.0)
    predicate_table: PredicateTable
    scalar_bounds: list[ScalarBound] = Field(default_factory=list)
    keep_out_terms: list[KeepOutTerm] = Field(default_factory=list)
    monitors: list[MonitorAutomaton] = Field(default_factory=list)
    resource_bounds: ResourceBounds
    action_limits: ActionLimits = Field(default_factory=ActionLimits)
    safe_pose: CompiledSafePose | None = None
    admissible_directives: CompiledAdmissibleDirectives | None = None

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this compiled model (its immutable identity).

        Over the canonical JSON of the model (:func:`astro_mine.core.hashing.content_hash_json`) —
        the platform's one content-address primitive — so a golden/determinism gate can pin the
        compiler output and two identical lowerings hash identically across machines."""
        return content_hash_json(self.model_dump(mode="json"))


# Resolve the self-referential CompiledNode.args forward reference.
CompiledNode.model_rebuild()
