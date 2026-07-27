"""SafetySpec loading, validation, and fail-safe semantic checks.

Pipeline (``load_safety_spec``), mirroring the Core objective/SADF loaders
(``astro_mine.core.objective.loader``):

1. parse YAML/JSON (YAML is a JSON superset, so one parser handles both);
2. **structural** validation against the canonical JSON Schema (unknown/typo'd fields,
   bad enum values, missing required, negative margins, …);
3. build the typed :class:`~astro_mine.guard.spec.model.SafetyDocument`;
4. **semantic** checks that exceed JSON Schema — and, critically, the **fail-safe**
   checks: exactly-one tagged-union field set, unique ids, every referenced signal
   declared, and every temporal operator carrying a **finite** interval. An unbounded
   temporal operator has no statically-bounded monitor, so it is rejected rather than
   silently admitted (guard.md §2 principle 4/6; the "fail safe, never fail open" rule).

The semantic checks run for both ``load_safety_spec`` and ``validate_safety_spec`` so the
JSON Schema and Pydantic stay a single structural contract.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core import schema_registry
from astro_mine.core.units import ReferenceFrame
from astro_mine.core.units.validate import UnitsValidationError, require_frame
from astro_mine.guard.spec.enums import ConstraintKind, GeometryKind, TemporalOp
from astro_mine.guard.spec.model import (
    Constraint,
    SafetyDocument,
    STLFormula,
)

__all__ = [
    "SafetySpecError",
    "SafetySpecValidationError",
    "load_safety_spec",
    "load_schema",
    "validate_safety_spec",
]

_SCHEMA_RESOURCE = "schema/safety_spec.schema.json"

#: The constraint tagged-union field carrying each :class:`ConstraintKind`'s payload.
_KIND_FIELD: dict[ConstraintKind, str] = {
    ConstraintKind.KEEP_OUT: "keep_out",
    ConstraintKind.POWER_FLOOR: "power_floor",
    ConstraintKind.ENERGY_FLOOR: "energy_floor",
    ConstraintKind.THERMAL_CEILING: "thermal_ceiling",
    ConstraintKind.THERMAL_FLOOR: "thermal_floor",
    ConstraintKind.TORQUE_CEILING: "torque_ceiling",
    ConstraintKind.KINEMATIC_LIMIT: "kinematic_limit",
    ConstraintKind.TEMPORAL: "temporal",
}
_UNION_FIELDS: tuple[str, ...] = tuple(_KIND_FIELD.values())

#: The keep-out geometry tagged-union field carrying each :class:`GeometryKind`'s payload.
_SHAPE_FIELD: dict[GeometryKind, str] = {
    GeometryKind.BOX: "box",
    GeometryKind.SPHERE: "sphere",
    GeometryKind.HALF_SPACE: "half_space",
}
_SHAPE_UNION_FIELDS: tuple[str, ...] = tuple(_SHAPE_FIELD.values())

#: Which constraint kinds reference a scalar ``signal`` key.
_SCALAR_SIGNAL_KINDS: frozenset[ConstraintKind] = frozenset(
    {
        ConstraintKind.POWER_FLOOR,
        ConstraintKind.ENERGY_FLOOR,
        ConstraintKind.THERMAL_CEILING,
        ConstraintKind.THERMAL_FLOOR,
        ConstraintKind.TORQUE_CEILING,
        ConstraintKind.KINEMATIC_LIMIT,
    }
)

_UNARY_OPS: frozenset[TemporalOp] = frozenset({TemporalOp.NOT})
_UNARY_TEMPORAL_OPS: frozenset[TemporalOp] = frozenset({TemporalOp.ALWAYS, TemporalOp.EVENTUALLY})
_NARY_OPS: frozenset[TemporalOp] = frozenset({TemporalOp.AND, TemporalOp.OR})


class SafetySpecError(Exception):
    """Base class for SafetySpec errors."""


class SafetySpecValidationError(SafetySpecError):
    """Raised when a SafetySpec fails structural or semantic (incl. fail-safe) validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical SafetySpec JSON Schema (shipped inside the package)."""
    text = (
        resources.files("astro_mine.guard.spec")
        .joinpath(_SCHEMA_RESOURCE)
        .read_text(encoding="utf-8")
    )
    schema: dict[str, Any] = json.loads(text)
    return schema


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    # The frame_ref fields `$ref` Core's canonical units.schema.json across files, naming it by
    # its absolute `$id` — public, append-only API (RFC-0009 §1; RFC-0007). jsonschema does not
    # fetch external refs, so `schema_registry()` resolves them offline: it carries every Core
    # schema keyed by its own `$id` (RFC-0009 §2), and `schema` is passed so Guard's own internal
    # `$ref`s resolve too.
    schema = load_schema()
    return Draft202012Validator(schema, registry=schema_registry(schema))


def _parse(source: str | bytes) -> Any:
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    return yaml.safe_load(source)


def _check_structural(data: Any) -> None:
    if not isinstance(data, dict):
        raise SafetySpecValidationError("SafetySpec document must be a YAML/JSON mapping")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise SafetySpecValidationError(f"SafetySpec failed schema validation: {rendered}")


def _check_union_exactly_one(
    obj: object,
    *,
    kind: str,
    fields: tuple[str, ...],
    expected: str,
    what: str,
) -> None:
    """A tagged union must set exactly the one field matching its discriminant."""
    present = [f for f in fields if getattr(obj, f, None) is not None]
    if present != [expected]:
        if expected not in present:
            raise SafetySpecValidationError(
                f"{what} declares kind {kind!r} but its {expected!r} payload is not set"
            )
        others = [f for f in present if f != expected]
        raise SafetySpecValidationError(
            f"{what} declares kind {kind!r} but also sets {others!r}; exactly one payload "
            f"field may be set"
        )


def _check_signal_key(signal: str, declared: set[str], *, where: str) -> None:
    if not signal.strip():
        raise SafetySpecValidationError(f"{where}: signal key must be a non-empty string")
    if signal not in declared:
        raise SafetySpecValidationError(
            f"{where}: signal {signal!r} is not declared in signals[] "
            f"(declared: {sorted(declared)})"
        )


def _check_formula(formula: STLFormula, declared: set[str], *, where: str) -> None:
    """Recursively validate an STL/MTL formula's arity, predicate fields, and — the fail-safe
    invariant — that every temporal operator carries a well-ordered, **finite** interval."""
    op = formula.op
    n = len(formula.args)
    if op == TemporalOp.PREDICATE:
        if formula.args:
            raise SafetySpecValidationError(f"{where}: a predicate node takes no operands")
        if formula.signal is None or formula.cmp is None or formula.threshold is None:
            raise SafetySpecValidationError(
                f"{where}: a predicate node requires signal, cmp, and threshold"
            )
        if formula.interval_s is not None:
            raise SafetySpecValidationError(f"{where}: a predicate node takes no interval")
        _check_signal_key(formula.signal, declared, where=f"{where}.predicate")
        return

    # Non-atomic nodes must not carry predicate fields.
    if formula.signal is not None or formula.cmp is not None or formula.threshold is not None:
        raise SafetySpecValidationError(
            f"{where}: a {op} node must not set predicate fields (signal/cmp/threshold)"
        )

    if op in _UNARY_OPS:
        if n != 1:
            raise SafetySpecValidationError(f"{where}: a {op} node takes exactly one operand")
        if formula.interval_s is not None:
            raise SafetySpecValidationError(f"{where}: a {op} node takes no interval")
    elif op in _NARY_OPS:
        if n < 2:
            raise SafetySpecValidationError(f"{where}: a {op} node takes at least two operands")
        if formula.interval_s is not None:
            raise SafetySpecValidationError(f"{where}: a {op} node takes no interval")
    elif op in _UNARY_TEMPORAL_OPS:
        if n != 1:
            raise SafetySpecValidationError(f"{where}: a {op} node takes exactly one operand")
        _check_interval(formula, where=where)
    elif op == TemporalOp.UNTIL:
        if n != 2:
            raise SafetySpecValidationError(f"{where}: an until node takes exactly two operands")
        _check_interval(formula, where=where)
    else:  # pragma: no cover - enum is exhaustive above
        raise SafetySpecValidationError(f"{where}: unknown temporal op {op!r}")

    for i, child in enumerate(formula.args):
        _check_formula(child, declared, where=f"{where}.args[{i}]")


def _check_interval(formula: STLFormula, *, where: str) -> None:
    """Enforce the fail-safe bound: a temporal operator MUST carry a finite, well-ordered
    interval — an unbounded (missing/infinite) horizon is rejected, never admitted."""
    interval = formula.interval_s
    if interval is None:
        raise SafetySpecValidationError(
            f"{where}: a {formula.op} node requires a bounded interval_s "
            f"(unbounded temporal operators are rejected — fail-safe)"
        )
    if not (math.isfinite(interval.lo) and math.isfinite(interval.hi)):
        raise SafetySpecValidationError(
            f"{where}: interval_s bounds must be finite (got [{interval.lo}, {interval.hi}]) — "
            f"an unbounded horizon has no statically-bounded monitor"
        )
    if interval.hi < interval.lo:
        raise SafetySpecValidationError(
            f"{where}: interval_s must be well-ordered lo <= hi "
            f"(got [{interval.lo}, {interval.hi}])"
        )


def _check_semantics(doc: SafetyDocument) -> None:
    spec = doc.safety

    # Signal vocabulary: keys unique and non-empty; units explicit (SI, non-empty).
    keys = [s.key for s in spec.signals]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise SafetySpecValidationError(
            f"duplicate signal key(s): {', '.join(dupes)} — signal keys must be unique"
        )
    declared: set[str] = set()
    for s in spec.signals:
        if not s.key.strip():
            raise SafetySpecValidationError("signal key must be a non-empty string")
        if not s.unit.strip():
            raise SafetySpecValidationError(
                f"signal {s.key!r}: unit must be an explicit, non-empty SI unit"
            )
        declared.add(s.key)

    # Constraint ids unique.
    ids = [c.id for c in spec.constraints]
    id_dupes = sorted({i for i in ids if ids.count(i) > 1})
    if id_dupes:
        raise SafetySpecValidationError(
            f"duplicate constraint id(s): {', '.join(id_dupes)} — constraint ids must be unique"
        )

    for c in spec.constraints:
        _check_constraint(c, declared)

    # A safe pose (retreat target) must live in the *same* frame as the keep-out geometry, so the
    # target and the safe set are expressed in one CRS (LUNAR-TR-001; guard.md §9.2). An explicit,
    # non-empty frame is required either way (conventions.md §5 — no implicit Earth frame).
    if spec.safe_pose is not None:
        if not spec.safe_pose.frame.strip():
            raise SafetySpecValidationError("safe_pose: frame must be an explicit, non-empty name")
        _check_frame_ref(spec.safe_pose.frame, spec.safe_pose.frame_ref, where="safe_pose")
        keepout_frames = {
            _keep_out_frame(c) for c in spec.constraints if c.kind == ConstraintKind.KEEP_OUT
        }
        mismatched = sorted(f for f in keepout_frames if f != spec.safe_pose.frame)
        if mismatched:
            raise SafetySpecValidationError(
                f"safe_pose frame {spec.safe_pose.frame!r} does not match the keep-out geometry "
                f"frame(s) {mismatched} — the retreat target and the safe set must share a frame"
            )


def _keep_out_frame(c: Constraint) -> str:
    """The frame the keep-out geometry of constraint ``c`` is expressed in."""
    frame, _ = _keep_out_frame_and_ref(c)
    return frame


def _keep_out_frame_and_ref(c: Constraint) -> tuple[str, ReferenceFrame | None]:
    """The keep-out geometry's ``frame`` token and its optional typed ``frame_ref`` (RFC-0007)."""
    assert c.keep_out is not None
    vol = c.keep_out.volume
    if vol.box is not None:
        return vol.box.frame, vol.box.frame_ref
    if vol.sphere is not None:
        return vol.sphere.frame, vol.sphere.frame_ref
    assert vol.half_space is not None
    return vol.half_space.frame, vol.half_space.frame_ref


def _check_frame_ref(frame: str, frame_ref: ReferenceFrame | None, *, where: str) -> None:
    """Resolve a spatial value's frame against Core's ``ReferenceFrame`` guard (RFC-0007).

    The ``frame`` string is a whitespace-free frame-name token (enforced structurally by the
    schema ``pattern``). When the typed ``frame_ref`` sibling is present it is resolved through
    Core's ``require_frame`` (rules 1-2: token names, ``frame_class`` a ``FrameClass`` member) and
    must name the same frame as the token — so the frame is never a bare, unvalidated string."""
    if frame_ref is None:
        return
    try:
        require_frame(frame_ref)
    except UnitsValidationError as exc:
        raise SafetySpecValidationError(f"{where}: invalid frame_ref — {exc}") from exc
    if frame_ref.name != frame:
        raise SafetySpecValidationError(
            f"{where}: frame_ref name {frame_ref.name!r} does not match frame {frame!r} "
            f"— the typed frame and the frame token must name one frame"
        )


def _check_constraint(c: Constraint, declared: set[str]) -> None:
    where = f"constraint {c.id!r}"
    _check_union_exactly_one(
        c,
        kind=str(c.kind),
        fields=_UNION_FIELDS,
        expected=_KIND_FIELD[c.kind],
        what=where,
    )

    if c.kind in _SCALAR_SIGNAL_KINDS:
        payload = getattr(c, _KIND_FIELD[c.kind])
        _check_signal_key(payload.signal, declared, where=where)
        if c.kind == ConstraintKind.KINEMATIC_LIMIT and (
            payload.max_velocity_mps is None and payload.max_accel_mps2 is None
        ):
            raise SafetySpecValidationError(
                f"{where}: a kinematic_limit must set at least one of "
                f"max_velocity_mps / max_accel_mps2"
            )
    elif c.kind == ConstraintKind.KEEP_OUT:
        assert c.keep_out is not None
        volume = c.keep_out.volume
        _check_union_exactly_one(
            volume,
            kind=str(volume.shape),
            fields=_SHAPE_UNION_FIELDS,
            expected=_SHAPE_FIELD[volume.shape],
            what=f"{where}.keep_out.volume",
        )
        if volume.shape == GeometryKind.HALF_SPACE and volume.half_space is not None:
            n = volume.half_space.normal
            if n.x**2 + n.y**2 + n.z**2 == 0.0:
                raise SafetySpecValidationError(
                    f"{where}.keep_out.volume: a half-space normal must be non-zero"
                )
        frame, frame_ref = _keep_out_frame_and_ref(c)
        _check_frame_ref(frame, frame_ref, where=f"{where}.keep_out.volume")
    elif c.kind == ConstraintKind.TEMPORAL:
        assert c.temporal is not None
        _check_formula(c.temporal.formula, declared, where=f"{where}.formula")


def load_safety_spec(source: str | bytes) -> SafetyDocument:
    """Parse, validate, and return a typed SafetySpec document.

    Raises :class:`SafetySpecValidationError` on any structural or semantic failure.
    """
    data = _parse(source)
    _check_structural(data)
    try:
        doc = SafetyDocument.model_validate(data)
    except ValidationError as exc:
        raise SafetySpecValidationError(f"SafetySpec failed model validation: {exc}") from exc
    _check_semantics(doc)
    return doc


def validate_safety_spec(document: Any) -> None:
    """Validate a SafetySpec without returning it.

    Accepts raw YAML/JSON text/bytes, a parsed mapping, or a
    :class:`~astro_mine.guard.spec.model.SafetyDocument`. Raises
    :class:`SafetySpecValidationError` on failure.
    """
    if isinstance(document, SafetyDocument):
        _check_structural(document.model_dump(mode="json"))
        _check_semantics(document)
        return
    if isinstance(document, str | bytes):
        load_safety_spec(document)
        return
    if isinstance(document, dict):
        _check_structural(document)
        try:
            doc = SafetyDocument.model_validate(document)
        except ValidationError as exc:
            raise SafetySpecValidationError(f"SafetySpec failed model validation: {exc}") from exc
        _check_semantics(doc)
        return
    raise SafetySpecValidationError(f"cannot validate object of type {type(document).__name__}")
