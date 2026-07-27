"""Anchor-scenario profile builder — wires the lunar-polar SafetySpec to Fleet/Worlds (GUARD-04).

The authoring-side glue that turns the reviewed anchor ``SafetySpec`` artifact (shipped as package
data — :data:`astro_mine.guard.reference.ANCHOR_SAFETY_SPEC_RESOURCE`) into a compiled model + the
runtime signal resolver
the ``PolicyShield`` drives, binding SADF budgets into the spec's thresholds and choosing the
survival-horizon sample period (guard.md §6; scenario §10). Untrusted authoring code — the
resulting compiled model is a reviewed, content-addressed safety artifact enforced by the Rust core.

**Sample period vs. history bound (the survival-horizon trade-off).** The anchor's two ~14-day
survival monitors need a ring buffer of ``ceil(horizon / sample_period_s)`` samples, and the *same*
period is the step the recover layer predicts its retreat/hold one tick ahead with. A coarse period
shrinks the buffer but makes the retreat prediction physically meaningless (it degrades to braking);
a fine period keeps control crisp but needs a large buffer. The anchor is authored at the natural
**1 Hz** control/telemetry cadence (:data:`ANCHOR_SAMPLE_PERIOD_S`) — so the retreat law predicts on
a 1 s step and the predictive monitors get a ~5 s lead — and the ~1.21 M-sample survival window is
carried by raising the core's history cap (:data:`ANCHOR_MAX_HISTORY_CAP`). The resulting ring
buffers (~tens of MB) suit a ground/sim deployment; a flight-edge build would coarsen or re-encode
the survival clause (deferred)."""

from __future__ import annotations

from astro_mine.guard.models.resolver import WorldsFleetSignalResolver, WorldsSignalKind
from astro_mine.guard.models.sadf import SadfBudgets
from astro_mine.guard.models.worlds import WorldsTerrain
from astro_mine.guard.reference import load_anchor_safety_spec
from astro_mine.guard.spec.compiler import compile_spec
from astro_mine.guard.spec.enums import ConstraintKind
from astro_mine.guard.spec.ir import CompiledSafetyModel
from astro_mine.guard.spec.model import Constraint, SafePose, SafetyDocument
from astro_mine.guard.wrap import CoreConfig

__all__ = [
    "ANCHOR_MAX_HISTORY_CAP",
    "ANCHOR_SAMPLE_PERIOD_S",
    "NIGHT_SURVIVAL_ENERGY_FRACTION",
    "anchor_core_config",
    "build_anchor_resolver",
    "compile_anchor",
    "load_anchor_document",
    "with_sadf_thresholds",
    "with_safe_pose",
]

#: The anchor control/telemetry cadence — the compiled sample period (see module docstring).
ANCHOR_SAMPLE_PERIOD_S = 1.0
#: History cap sized for the ~14-day survival window at 1 Hz (1_209_600 samples), with headroom.
ANCHOR_MAX_HISTORY_CAP = 1 << 21  # 2_097_152
#: The stored-energy reserve the night-survival floor holds (fraction of battery capacity).
NIGHT_SURVIVAL_ENERGY_FRACTION = 0.15


def load_anchor_document() -> SafetyDocument:
    """Load and validate the reviewed anchor SafetySpec from package data (offline, wheel-safe)."""
    return load_anchor_safety_spec()


def _rebind_constraint(c: Constraint, budgets: SadfBudgets) -> Constraint:
    """Return ``c`` with its threshold rebound to the matching SADF budget, where the budget
    supplies one (else unchanged). Authoring-time, provenance-bound (guard.md §6)."""
    kind = c.kind
    if kind == ConstraintKind.POWER_FLOOR and budgets.power_floor_w is not None:
        assert c.power_floor is not None
        return c.model_copy(
            update={
                "power_floor": c.power_floor.model_copy(update={"floor_w": budgets.power_floor_w})
            }
        )
    if kind == ConstraintKind.ENERGY_FLOOR:
        floor_j = budgets.energy_floor_j(NIGHT_SURVIVAL_ENERGY_FRACTION)
        if floor_j is not None:
            assert c.energy_floor is not None
            return c.model_copy(
                update={"energy_floor": c.energy_floor.model_copy(update={"floor_j": floor_j})}
            )
    if kind == ConstraintKind.THERMAL_CEILING and budgets.thermal_operating_max_k is not None:
        assert c.thermal_ceiling is not None
        return c.model_copy(
            update={
                "thermal_ceiling": c.thermal_ceiling.model_copy(
                    update={"limit_k": budgets.thermal_operating_max_k}
                )
            }
        )
    if kind == ConstraintKind.THERMAL_FLOOR and budgets.thermal_survival_min_k is not None:
        assert c.thermal_floor is not None
        return c.model_copy(
            update={
                "thermal_floor": c.thermal_floor.model_copy(
                    update={"limit_k": budgets.thermal_survival_min_k}
                )
            }
        )
    if kind == ConstraintKind.TORQUE_CEILING and budgets.max_actuator_torque_nm is not None:
        assert c.torque_ceiling is not None
        return c.model_copy(
            update={
                "torque_ceiling": c.torque_ceiling.model_copy(
                    update={"max_nm": budgets.max_actuator_torque_nm}
                )
            }
        )
    return c


def with_sadf_thresholds(document: SafetyDocument, budgets: SadfBudgets) -> SafetyDocument:
    """Return ``document`` with its power/energy/thermal/torque thresholds rebound to ``budgets``
    (Fleet SADF), leaving every other field intact. The output is re-validated when compiled."""
    spec = document.safety
    rebound = [_rebind_constraint(c, budgets) for c in spec.constraints]
    return document.model_copy(update={"safety": spec.model_copy(update={"constraints": rebound})})


def with_safe_pose(document: SafetyDocument, safe_pose: SafePose) -> SafetyDocument:
    """Return ``document`` with its authored retreat target set to ``safe_pose``."""
    return document.model_copy(
        update={"safety": document.safety.model_copy(update={"safe_pose": safe_pose})}
    )


def compile_anchor(
    *,
    sample_period_s: float = ANCHOR_SAMPLE_PERIOD_S,
    budgets: SadfBudgets | None = None,
    safe_pose: SafePose | None = None,
) -> CompiledSafetyModel:
    """Compile the anchor profile, optionally rebinding thresholds from SADF ``budgets`` and/or
    overriding the authored retreat target with ``safe_pose``."""
    document = load_anchor_document()
    if budgets is not None:
        document = with_sadf_thresholds(document, budgets)
    if safe_pose is not None:
        document = with_safe_pose(document, safe_pose)
    return compile_spec(document, sample_period_s=sample_period_s)


def anchor_core_config(**overrides: object) -> CoreConfig:
    """The ``PolicyShield`` core config sized for the anchor: the history cap carries the ~14-day
    survival window at 1 Hz. Any keyword overrides the corresponding field."""
    params: dict[str, object] = {"max_history_cap": ANCHOR_MAX_HISTORY_CAP}
    params.update(overrides)
    return CoreConfig(**params)  # type: ignore[arg-type]


def build_anchor_resolver(
    *,
    terrain: WorldsTerrain | None = None,
    budgets: SadfBudgets | None = None,
) -> WorldsFleetSignalResolver:
    """The runtime signal resolver for the anchor: the illumination-keyed charging window and the
    (optional) terrain slope resolve from Worlds; everything else from the observation; unresolved
    ⇒ ``NaN`` (fail-safe). ``budgets`` is retained on the resolver for provenance."""
    return WorldsFleetSignalResolver(
        terrain=terrain,
        budgets=budgets,
        worlds_bindings={
            "charging_window_active": WorldsSignalKind.CHARGING_WINDOW,
            "slope_deg": WorldsSignalKind.SLOPE_DEG,
        },
    )
