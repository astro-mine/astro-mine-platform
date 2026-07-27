"""SafetySpec v0.1 — typed Pydantic models (the Guard-owned safety contract).

The declarative document of *hard* constraints every Guard layer compiles from
(guard.md §3, §5): geometric keep-out volumes and collision pairs, power/energy floors,
thermal and torque ceilings, kinematic limits, and **STL/MTL temporal clauses**. Authored
by a safety engineer, reviewed as a safety artifact, and lowered by the constraint
compiler (:mod:`astro_mine.guard.spec.compiler`) into the analyzable IR the future Rust
safety core (RM-P1-GUARD-02) enforces.

The **canonical** schema is the hand-authored JSON Schema in
``schema/safety_spec.schema.json`` (shipped in-package); these models mirror it and a
drift guard (``scripts/check_model_drift.py`` + ``tests/test_safetyspec_consistency.py``)
asserts the two agree — the Core objective/SADF idiom (RM-P0-CORE-01/04). All quantities
are SI; every spatial value resolves in an explicitly named frame (conventions.md §5).

Design notes:

- **Guard owns this schema** (guard.md §5); it does not edit ``astro_mine.core.messages``.
  It is registered *through* the Core plugin registry (:mod:`astro_mine.guard.spec.catalog`)
  and content-addressed with the one Core hashing primitive.
- **Constraint sources are referenced abstractly** by a :class:`SignalRef` key/path
  (guard.md §5, §6) — a Worlds keep-out, a Fleet SADF budget like ``power.floor_w``, or a
  Core observation channel. Actual value resolution is deferred to RM-P1-GUARD-04.
- **Fail-safe by construction** (guard.md §2 principle 4): every constraint carries an
  :class:`~astro_mine.guard.spec.enums.OnUncertain` that defaults to ``fallback`` and has no
  ``passthrough`` value; temporal operators must carry a **finite** interval, enforced by
  the loader — an unbounded operator has no statically-bounded monitor and is rejected.
- The models are **purely structural**. Cross-reference and fail-safe *semantic* rules
  (unique ids, exactly-one union field set, declared-signal references, bounded intervals)
  exceed JSON Schema's expressiveness and live in :mod:`astro_mine.guard.spec.loader`, so
  the model stays behaviourally identical to the canonical JSON Schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.hashing import content_hash_json
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.messages.model import Vec3, Volume
from astro_mine.core.units import ReferenceFrame
from astro_mine.guard.spec.enums import (
    ConstraintKind,
    GeometryKind,
    OnUncertain,
    PredicateOp,
    SignalSource,
    TemporalOp,
)

__all__ = [
    "SAFETY_VERSION",
    "AdmissibleDirectives",
    "Constraint",
    "EnergyFloorConstraint",
    "Interval",
    "KeepOutConstraint",
    "KeepOutHalfSpace",
    "KeepOutSphere",
    "KeepOutVolume",
    "KinematicLimitConstraint",
    "PowerFloorConstraint",
    "Provenance",
    "ReferenceFrame",
    "STLFormula",
    "SafePose",
    "SafetyDocument",
    "SafetySpec",
    "SignalRef",
    "TemporalConstraint",
    "ThermalCeilingConstraint",
    "ThermalFloorConstraint",
    "TorqueCeilingConstraint",
    "Vec3",
    "Volume",
]

SAFETY_VERSION = "0.1"


class _Model(BaseModel):
    """Base for every SafetySpec model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


# --- signals & value types -------------------------------------------------------


class SignalRef(_Model):
    """A named runtime signal a constraint reads, referenced **abstractly** by key.

    The vocabulary declared once in ``SafetySpec.signals`` and referenced elsewhere by
    ``key`` (a predicate's ``signal``, a floor/ceiling constraint's ``signal``). ``source``
    names where it is resolved from (a Core observation channel, a Fleet SADF budget path
    like ``power.floor_w``, a Worlds keep-out field, or a derived signal); ``unit`` is the
    explicit SI unit the ``threshold``/limit is expressed in (conventions.md §5). Actual
    resolution against Worlds/Fleet is deferred to RM-P1-GUARD-04 — here a signal is only a
    declared, unit-explicit reference the compiler resolves to an integer index."""

    key: str
    unit: str
    source: SignalSource
    description: str | None = None


class Interval(_Model):
    """A closed, **finite** time interval ``[lo, hi]`` in SI seconds.

    The horizon of a bounded temporal operator (``always``/``eventually``/``until``). The
    loader requires ``0 <= lo <= hi`` with both endpoints finite — an unbounded horizon has
    no statically-bounded monitor history window and is rejected (fail-safe; guard.md §2
    principle 6). ``hi`` is what the compiler turns into a bounded history-buffer length."""

    lo: float = Field(ge=0.0)
    hi: float = Field(ge=0.0)


# --- keep-out geometry (a Guard-local union anchored on the Core Volume) ----------


class KeepOutSphere(_Model):
    """A spherical keep-out region: the ball of ``radius_m`` about ``center_m`` in
    ``frame``. The compiler lowers it to a barrier term ``|x - center| - radius``.

    ``frame`` is the frame name (a non-empty, whitespace-free SPICE token; conventions.md §5);
    ``frame_ref`` is its optional typed sibling — the Core :class:`ReferenceFrame` resolved
    against the ``require_frame`` guard (RFC-0007). When set its ``name`` must match ``frame``
    (loader-enforced)."""

    frame: str
    center_m: Vec3
    radius_m: float = Field(gt=0.0)
    frame_ref: ReferenceFrame | None = None


class KeepOutHalfSpace(_Model):
    """A half-space keep-out barrier. The **safe** set is ``{x : normal · x + offset_m >= 0}``
    (the standard control-barrier-function half-space form), so the compiler emits a single
    linear barrier term with the given coefficients. ``normal`` need not be unit length; the
    compiler normalizes it so ``margin_m`` is a true metric distance.

    ``frame_ref`` is the optional typed Core :class:`ReferenceFrame` sibling of ``frame``
    (RFC-0007)."""

    frame: str
    normal: Vec3
    offset_m: float
    frame_ref: ReferenceFrame | None = None


class KeepOutVolume(_Model):
    """A keep-out region — a tagged union over :class:`~astro_mine.guard.spec.enums.GeometryKind`.

    Exactly one typed field matching ``shape`` is set (enforced in the loader). ``box`` is
    the Core :class:`~astro_mine.core.messages.model.Volume` (axis-aligned, frame-explicit);
    ``sphere`` and ``half_space`` are the Guard-local barrier primitives."""

    shape: GeometryKind
    box: Volume | None = None
    sphere: KeepOutSphere | None = None
    half_space: KeepOutHalfSpace | None = None


# --- STL/MTL temporal formula AST ------------------------------------------------


class STLFormula(_Model):
    """One node of an STL/MTL formula tree — a **structured AST**, not a string DSL
    (guard.md §4, §9.2): reviewable and analyzable with no parser dependency.

    ``op`` is the node kind (:class:`~astro_mine.guard.spec.enums.TemporalOp`) and selects
    which fields are meaningful (enforced in the loader):

    - ``predicate`` — an atomic bound ``signal <cmp> threshold``; ``args`` empty.
    - ``not`` — unary boolean; ``args`` has exactly one child.
    - ``and`` / ``or`` — n-ary boolean; ``args`` has at least two children.
    - ``always`` / ``eventually`` — unary temporal; ``args`` has exactly one child and
      ``interval_s`` is a required **finite** horizon.
    - ``until`` — binary temporal; ``args`` has exactly two children (``args[0]`` until
      ``args[1]``) and ``interval_s`` is a required finite horizon.

    A temporal operator without a finite ``interval_s`` is rejected by the loader, so every
    monitor the compiler emits has a statically-bounded history window (fail-safe)."""

    op: TemporalOp
    # atomic predicate (op == predicate)
    signal: str | None = None
    cmp: PredicateOp | None = None
    threshold: float | None = None
    # bounded temporal horizon (op in always/eventually/until)
    interval_s: Interval | None = None
    # recursive operands (unary: 1, binary: 2, n-ary: >= 2)
    args: list[STLFormula] = Field(default_factory=list)


# --- constraint kinds (the tagged-union arms) ------------------------------------


class KeepOutConstraint(_Model):
    """Stay outside ``volume`` by at least ``margin_m`` metres. ``collision_pair`` optionally
    names the two bodies/agents whose separation this enforces (a pairwise keep-out)."""

    volume: KeepOutVolume
    margin_m: float = Field(ge=0.0)
    collision_pair: tuple[str, str] | None = None


class PowerFloorConstraint(_Model):
    """Instantaneous power draw/availability ``signal`` must stay at or above ``floor_w``
    watts — a hard power floor (Fleet SADF ``power.floor_w``; guard.md §3)."""

    signal: str
    floor_w: float


class EnergyFloorConstraint(_Model):
    """Stored energy ``signal`` (e.g. battery state-of-charge in joules) must stay at or
    above ``floor_j`` — the lunar-night survival floor (Fleet SADF ``PowerStorage``)."""

    signal: str
    floor_j: float


class ThermalCeilingConstraint(_Model):
    """Temperature ``signal`` must stay at or below ``limit_k`` kelvin (operating/survival
    ceiling; Fleet SADF ``ThermalBudget``)."""

    signal: str
    limit_k: float


class ThermalFloorConstraint(_Model):
    """Temperature ``signal`` must stay at or above ``limit_k`` kelvin (survival floor for
    lunar night / cruise; Fleet SADF ``ThermalBudget.survival_range_k``)."""

    signal: str
    limit_k: float


class TorqueCeilingConstraint(_Model):
    """Actuator torque ``signal`` must stay at or below ``max_nm`` newton-metres (Fleet SADF
    ``Actuator.torque_nm`` / ``JointLimits.effort_nm``)."""

    signal: str
    max_nm: float = Field(gt=0.0)


class KinematicLimitConstraint(_Model):
    """A kinematic envelope on ``signal``: at least one of ``max_velocity_mps`` /
    ``max_accel_mps2`` bounds the signal's magnitude (Fleet SADF ``JointLimits.velocity_rad_s``,
    ``ContactElement`` limits). The loader requires at least one bound to be set."""

    signal: str
    max_velocity_mps: float | None = Field(default=None, ge=0.0)
    max_accel_mps2: float | None = Field(default=None, ge=0.0)


class TemporalConstraint(_Model):
    """A temporal-logic clause: the ``formula`` (bounded STL/MTL) must hold — e.g.
    "battery SoC >= floor *until* a charging window" or "always: distance-to-keep-out >=
    margin" (guard.md §3)."""

    formula: STLFormula


class Constraint(_Model):
    """One hard constraint — a tagged union over the :class:`~.enums.ConstraintKind` discriminant.

    Exactly one typed field matching ``kind`` is set (enforced in the loader, the
    TaskDirective/Action idiom). Every constraint carries an ``on_uncertain`` that defaults
    to ``fallback`` and can never be ``passthrough`` — the fail-safe guarantee is in the
    schema, not just the runtime (guard.md §2 principle 4, §9.1)."""

    kind: ConstraintKind
    id: str
    on_uncertain: OnUncertain = OnUncertain.FALLBACK
    description: str | None = None
    keep_out: KeepOutConstraint | None = None
    power_floor: PowerFloorConstraint | None = None
    energy_floor: EnergyFloorConstraint | None = None
    thermal_ceiling: ThermalCeilingConstraint | None = None
    thermal_floor: ThermalFloorConstraint | None = None
    torque_ceiling: TorqueCeilingConstraint | None = None
    kinematic_limit: KinematicLimitConstraint | None = None
    temporal: TemporalConstraint | None = None


# --- provenance & top level ------------------------------------------------------


class Provenance(_Model):
    """Reproducibility provenance (conventions.md §5). A SafetySpec is content-addressed so
    a design-time safety claim and an operational reading reproduce (guard.md §5)."""

    input_hashes: list[str] = Field(default_factory=list)
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None
    seed: int | None = None


class SafePose(_Model):
    """An authored safe / charging pose the verified **retreat** (``safe_state``) backup drives
    toward — the lunar-night survival target (guard.md §9.2 "verified safe states"; RM-P1-GUARD-04).

    A body-fixed position in an explicitly named ``frame`` (SI metres; conventions.md §5,
    LUNAR-TR-001 — no implicit Earth/WGS84 frame). The compiler lowers ``position_m`` into the
    keep-out spatial frame; the trusted core steers toward it with bounded control and falls back
    to brake-to-stop if no safe distance-reducing action exists (fail-safe, never fail-open). The
    ``frame`` must match the keep-out geometry's frame (enforced in the loader) so the target and
    the safe set are expressed in one CRS. Attitude is deliberately omitted for the point-mass
    plant and reserved for a future additive, RFC-gated extension.

    ``frame_ref`` is the optional typed Core :class:`ReferenceFrame` sibling of ``frame``
    (RFC-0007), validated against ``require_frame``."""

    frame: str
    position_m: Vec3
    frame_ref: ReferenceFrame | None = None


class AdmissibleDirectives(_Model):
    """The discrete ``MODE``/``TASK`` directives this contract certifies, **by enumeration**
    (RFC-0004 Amendment 2; guard.md §3, §5).

    A directive carries no continuous quantity to project, so the shield cannot *correct* it — the
    trusted core can only certify it against an allowlist, and an admitted directive is re-emitted
    **untouched**. That makes the grant itself the safety decision, which is why it is authored
    *here* — content-addressed, reviewed, and signed with the rest of the contract — rather than in
    deployment configuration. (A ``ModeCommand.mode`` names a SADF ``loads_by_mode`` profile: the
    very power/thermal load profile the survival floors above are stated *against*.)

    **Silence grants nothing.** The gate's effective allowlist is
    ``spec ∩ config`` — :class:`~astro_mine.guard.wrap.ActionPolicy` may only ever **narrow** this
    grant, never widen it — and an *absent* (or empty) ``admissible_directives`` admits **no**
    directive at all, whatever the configuration says. This is the deliberate asymmetry with the
    ``kinematic_limit`` → :class:`~astro_mine.guard.spec.ir.ActionLimits` ceiling, where an absent
    authored limit lets the configured one stand: both merges are the greatest-lower-bound, but the
    identity of a *ceiling*'s meet is ``+∞`` while the identity of a *permission set*'s meet is
    ``∅``. Reading silence as "config stands" would be fail-open-by-silence.

    ``tasks`` is typed as Core's closed :class:`~astro_mine.core.messages.enums.TaskKind`, so an
    unknown task is refused at authoring time (fail-closed). ``modes`` is an open vocabulary (free
    strings from a SADF asset's ``loads_by_mode``); validating a mode name against a concrete asset
    is deferred (RFC-0004 Amendment 2, *Deferred*)."""

    modes: list[str] = Field(default_factory=list)
    tasks: list[TaskKind] = Field(default_factory=list)


class SafetySpec(_Model):
    """A safety contract: an identity, a signal vocabulary, and its hard constraints.

    ``signals`` declares every signal a constraint references (by ``key``); ``constraints``
    is the non-empty list of hard constraints. ``scenario_ref`` is an optional content
    reference to the ScenarioSpec the spec is stated against. Authored and reviewed once,
    then reused across design-time training, sim validation, and operations (guard.md §5).

    ``admissible_directives`` is the reviewed ``MODE``/``TASK`` allowlist the action gate certifies
    against (RFC-0004 Amendment 2); absent ⇒ **no** directive is certifiable."""

    id: str
    name: str
    description: str | None = None
    scenario_ref: str | None = None
    signals: list[SignalRef] = Field(default_factory=list)
    constraints: list[Constraint] = Field(min_length=1)
    provenance: Provenance | None = None
    safe_pose: SafePose | None = None
    admissible_directives: AdmissibleDirectives | None = None


class SafetyDocument(_Model):
    """Top-level SafetySpec document. ``safety_version`` pins the schema minor."""

    safety_version: Literal["0.1"]
    safety: SafetySpec

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this spec (its immutable identity).

        Over the canonical JSON of the model — the platform's one content-address primitive
        (:func:`astro_mine.core.hashing.content_hash_json`) — so the Core manifest can pin
        the spec by hash and two identical specs hash identically across machines
        (guard.md §5 "content-addressed"). The Protobuf wire form
        (:mod:`astro_mine.guard.spec.wire`) is a separate, independently byte-stable
        interchange encoding."""
        return content_hash_json(self.model_dump(mode="json"))


# Resolve the self-referential STLFormula.args forward reference.
STLFormula.model_rebuild()
