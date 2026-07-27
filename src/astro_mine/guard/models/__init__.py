"""``astro_mine.guard.models`` — untrusted constraint-source adapters (RM-P1-GUARD-04).

The ``models/`` layer resolves a ``SafetySpec``'s abstract constraint sources against **Core-typed**
Fleet SADF budgets and Worlds terrain/illumination (guard.md §3, §6), producing the thresholds a
profile author binds and the per-tick signal vector the ``PolicyShield`` feeds the trusted Rust
core. It sits deliberately **outside** the trusted computing base (guard.md §3): a wrong adapter can
make a *threshold* or a *signal* wrong, but the safety guarantee — fail-safe, never fail-open — is
enforced entirely in the Rust core, and every unresolved signal degrades to ``NaN`` (⇒ verified
backup). No sibling imports: the adapters read only ``astro_mine.core`` (the narrow waist)."""

from __future__ import annotations

from astro_mine.guard.models.profile import (
    ANCHOR_MAX_HISTORY_CAP,
    ANCHOR_SAMPLE_PERIOD_S,
    NIGHT_SURVIVAL_ENERGY_FRACTION,
    anchor_core_config,
    build_anchor_resolver,
    compile_anchor,
    load_anchor_document,
    with_sadf_thresholds,
    with_safe_pose,
)
from astro_mine.guard.models.resolver import WorldsFleetSignalResolver, WorldsSignalKind
from astro_mine.guard.models.sadf import SadfBudgets
from astro_mine.guard.models.worlds import WorldsTerrain, slope_deg_from_normal

__all__ = [
    "ANCHOR_MAX_HISTORY_CAP",
    "ANCHOR_SAMPLE_PERIOD_S",
    "NIGHT_SURVIVAL_ENERGY_FRACTION",
    "SadfBudgets",
    "WorldsFleetSignalResolver",
    "WorldsSignalKind",
    "WorldsTerrain",
    "anchor_core_config",
    "build_anchor_resolver",
    "compile_anchor",
    "load_anchor_document",
    "slope_deg_from_normal",
    "with_sadf_thresholds",
    "with_safe_pose",
]
