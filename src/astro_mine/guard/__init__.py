# SPDX-License-Identifier: Apache-2.0
"""Astro-Mine-Guard — runtime safety assurance.

The verifiable shield that wraps any decision producer — a learned policy from
Learn, a planner from Mind, an allocation from Allocate, or a hand-written
controller — so declared **hard constraints cannot be violated**, whatever the
wrapped component proposes. A declarative ``SafetySpec`` compiles into runtime
monitors (STL/MTL), reachability/CBF shields, and a simplex backup; an ``arbiter``
combines them into a single certified action per tick, exposed as a ``PolicyShield``
that implements the Core Policy/Planner API. A compromised or pathological policy
can degrade performance — never safety.

Safety-critical. Fail safe, never fail open. The trusted safety core (arbiter,
shields, monitors, backup, spec evaluator) is a Rust extension —
``astro_mine.guard._core`` (RM-P1-GUARD-02) — imported directly where needed; it is
kept out of this package's import path so the spec tooling loads without the compiled
core present. See ``docs/architecture/guard.md``.
"""

from __future__ import annotations

# `_version` is imported first (it sorts first and imports nothing from this package) so
# `__version__` is bound before the spec-tooling re-exports below — `spec.catalog` imports it from
# this package, and a heavier import placed first would see a partially-initialized module.
#
# The spec-tooling surface is all _core-free (the compiled Rust core stays out of the import path,
# per the module docstring); it is re-exported here so `from astro_mine.guard import
# load_safety_spec, compile_spec, ...` works without spelunking into submodules. `falsify` and the
# shield runtime, which need _core, are deliberately NOT re-exported; nor is the CLI entry point
# (`astro_mine.guard.cli:main`), so importing this package stays light.
from astro_mine.guard._version import __version__
from astro_mine.guard.reference import anchor_safety_spec_text, load_anchor_safety_spec
from astro_mine.guard.spec import (
    CompiledSafetyModel,
    CompileError,
    SafetyDocument,
    SafetySpec,
    SafetySpecError,
    SafetySpecValidationError,
    compile_spec,
    compiled_content_hash,
    load_safety_spec,
    spec_content_hash,
    validate_safety_spec,
)

__all__ = [
    "CompileError",
    "CompiledSafetyModel",
    "SafetyDocument",
    "SafetySpec",
    "SafetySpecError",
    "SafetySpecValidationError",
    "__version__",
    "anchor_safety_spec_text",
    "compile_spec",
    "compiled_content_hash",
    "load_anchor_safety_spec",
    "load_safety_spec",
    "spec_content_hash",
    "validate_safety_spec",
]
