# SPDX-License-Identifier: Apache-2.0
"""SafetySpec + the constraint compiler — Guard's declarative safety contract (RM-P1-GUARD-01).

The Guard-**owned**, versioned document of *hard* constraints every Guard layer compiles from
(keep-out volumes, power/energy floors, thermal/torque ceilings, kinematic limits, and STL/MTL
temporal clauses), plus the deterministic compiler that lowers a validated spec into the
analyzable, content-addressed :class:`CompiledSafetyModel` the future Rust safety core
(RM-P1-GUARD-02) enforces. "What is safe" is authored and reviewed once, then reused across
design-time training, sim validation, and operations (guard.md §5). Fail-safe, never fail-open.

The canonical schemas are ``schema/safety_spec.schema.json`` and
``schema/compiled_safety_model.schema.json`` (shipped in-package); the typed models live in
:mod:`~astro_mine.guard.spec.model` / :mod:`~astro_mine.guard.spec.ir`, the closed vocabularies
in :mod:`~astro_mine.guard.spec.enums`.

Public API:

- :func:`load_safety_spec` / :func:`validate_safety_spec` — parse + validate (structural +
  fail-safe semantic);
- :func:`compile_spec` — lower a validated spec to the compiled IR;
- :func:`to_wire` / :func:`from_wire` (and the ``compiled_*`` variants) — byte-stable Protobuf
  round-trip;
- :func:`build_safety_manifest` / :func:`register_safety_spec` — Core-catalog registration;
- :func:`load_schema` — the canonical SafetySpec JSON Schema as a dict.
"""

from __future__ import annotations

from astro_mine.guard.spec import enums, ir, model
from astro_mine.guard.spec.catalog import (
    COMPILED_MODEL_OUTPUT,
    SAFETY_SPEC_INTERFACE_VERSIONS,
    SAFETY_SPEC_OUTPUT,
    build_safety_manifest,
    compiled_content_hash,
    register_safety_spec,
    sign_safety_manifest,
    spec_content_hash,
)
from astro_mine.guard.spec.compiler import (
    DEFAULT_SAMPLE_PERIOD_S,
    CompileError,
    compile_spec,
)
from astro_mine.guard.spec.enums import (
    ConstraintKind,
    GeometryKind,
    OnUncertain,
    PredicateOp,
    SignalSource,
    TemporalOp,
)
from astro_mine.guard.spec.ir import (
    CompiledNode,
    CompiledSafetyModel,
    KeepOutTerm,
    MonitorAutomaton,
    PredicateAtom,
    PredicateTable,
    ResourceBounds,
    ScalarBound,
)
from astro_mine.guard.spec.keyid import signer_id
from astro_mine.guard.spec.loader import (
    SafetySpecError,
    SafetySpecValidationError,
    load_safety_spec,
    load_schema,
    validate_safety_spec,
)
from astro_mine.guard.spec.model import (
    SAFETY_VERSION,
    Constraint,
    Interval,
    KeepOutConstraint,
    KeepOutHalfSpace,
    KeepOutSphere,
    KeepOutVolume,
    SafetyDocument,
    SafetySpec,
    SignalRef,
    STLFormula,
    TemporalConstraint,
)
from astro_mine.guard.spec.signed import (
    LoadedArtifactProvenance,
    guard_verifier,
    load_signed_compiled_model,
    load_signed_safety_spec,
)
from astro_mine.guard.spec.wire import (
    compiled_from_proto,
    compiled_from_wire,
    compiled_to_proto,
    compiled_to_wire,
    from_proto,
    from_wire,
    to_proto,
    to_wire,
)
from astro_mine.seal import (
    SignatureError,
    generate_keypair,
    sign_digest,
    verify_signature,
)

__all__ = [
    "COMPILED_MODEL_OUTPUT",
    "DEFAULT_SAMPLE_PERIOD_S",
    "SAFETY_SPEC_INTERFACE_VERSIONS",
    "SAFETY_SPEC_OUTPUT",
    "SAFETY_VERSION",
    "CompileError",
    "CompiledNode",
    "CompiledSafetyModel",
    "Constraint",
    "ConstraintKind",
    "GeometryKind",
    "Interval",
    "KeepOutConstraint",
    "KeepOutHalfSpace",
    "KeepOutSphere",
    "KeepOutTerm",
    "KeepOutVolume",
    "LoadedArtifactProvenance",
    "MonitorAutomaton",
    "OnUncertain",
    "PredicateAtom",
    "PredicateOp",
    "PredicateTable",
    "ResourceBounds",
    "STLFormula",
    "SafetyDocument",
    "SafetySpec",
    "SafetySpecError",
    "SafetySpecValidationError",
    "ScalarBound",
    "SignalRef",
    "SignalSource",
    "SignatureError",
    "TemporalConstraint",
    "TemporalOp",
    "build_safety_manifest",
    "compile_spec",
    "compiled_content_hash",
    "compiled_from_proto",
    "compiled_from_wire",
    "compiled_to_proto",
    "compiled_to_wire",
    "enums",
    "from_proto",
    "from_wire",
    "generate_keypair",
    "guard_verifier",
    "ir",
    "load_safety_spec",
    "load_schema",
    "load_signed_compiled_model",
    "load_signed_safety_spec",
    "model",
    "register_safety_spec",
    "sign_digest",
    "sign_safety_manifest",
    "signer_id",
    "spec_content_hash",
    "to_proto",
    "to_wire",
    "validate_safety_spec",
    "verify_signature",
]
