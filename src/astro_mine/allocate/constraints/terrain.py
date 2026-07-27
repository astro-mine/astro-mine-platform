"""``build_terrain_constraints`` — Worlds traversability → keep-out constraints (RM-P1-ALLOC-03).

Lifts the [Worlds](worlds.md) traversability layer (slope, illumination, regolith bearing) and the
asset's declared mobility limits into IR **keep-out** constraints: a ``(task, asset)`` pair whose
location the asset cannot traverse is forbidden (its assignment variable is pinned to ``0``).
Allocate invents no physics — it *reads* Worlds' truth through the Core
:class:`~astro_mine.core.world.protocol.WorldProvider` contract and compares it against the asset's
SADF-declared limits (``mobility.contact[*].max_slope_deg`` / ``max_ground_pressure_pa``). The
quantitative traversal *duration* is not re-derived either: it comes from the cached cost table
(:mod:`astro_mine.allocate.constraints.costs`), with a declared, degraded-flagged fallback when the
table is silent.

No sibling import: the terrain truth arrives only through the Core ``WorldProvider`` in the
:class:`~astro_mine.allocate.ConstraintContext` (allocate.md §6).
"""

from __future__ import annotations

import math

from astro_mine.allocate.api.model import AllocationRequest, ConstraintContext, Task
from astro_mine.allocate.constraints.config import ConstraintConfig, TerrainPolicy
from astro_mine.allocate.constraints.costs import CostTable
from astro_mine.allocate.constraints.result import ConstraintFinding, Pair, TerrainResult
from astro_mine.allocate.model.ir.model import AllocationIR
from astro_mine.allocate.model.ir.utils import assignment_pairs
from astro_mine.core.sadf import Asset
from astro_mine.core.world.model import IlluminationState, SurfacePoint

__all__ = ["build_terrain_constraints", "keepout_constraint_id", "slope_deg"]


def keepout_constraint_id(task_id: str, asset_id: str) -> str:
    """The stable id of the keep-out constraint pinning ``assign(task, asset) = 0``."""
    return f"keepout::{task_id}::{asset_id}"


def slope_deg(
    surface_normal: tuple[float, float, float], gravity: tuple[float, float, float]
) -> float:
    """Local terrain slope (degrees): the angle between the surface normal and local vertical.

    Local vertical is ``-gravity`` (up). Reads Worlds' geometry — it does not model it. Returns
    ``0.0`` for a degenerate (zero-length) normal or gravity so a malformed sample never keeps a
    task out spuriously; the caller treats such a case as un-assessable, not as a cliff.
    """
    nx, ny, nz = surface_normal
    gx, gy, gz = gravity
    n_mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    g_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
    if n_mag == 0.0 or g_mag == 0.0:
        return 0.0
    # up = -gravity/|gravity|; cos(theta) = dot(normal, up) / |normal|.
    cos_theta = -(nx * gx + ny * gy + nz * gz) / (n_mag * g_mag)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


def _asset_max_slope_deg(asset: Asset | None, policy: TerrainPolicy) -> float:
    """The steepest slope an asset may traverse — the min over its SADF contact elements, or the
    declared policy default when the asset declares none."""
    if asset is None or asset.mobility is None:
        return policy.default_max_slope_deg
    limits = [c.max_slope_deg for c in asset.mobility.contact if c.max_slope_deg is not None]
    return min(limits) if limits else policy.default_max_slope_deg


def _asset_ground_pressure_pa(asset: Asset | None) -> float | None:
    """The asset's worst-case ground pressure (max over contact elements), or ``None``."""
    if asset is None or asset.mobility is None:
        return None
    pressures = [
        c.max_ground_pressure_pa
        for c in asset.mobility.contact
        if c.max_ground_pressure_pa is not None
    ]
    return max(pressures) if pressures else None


def _task_by_id(request: AllocationRequest) -> dict[str, Task]:
    return {t.task_id: t for t in request.tasks}


def build_terrain_constraints(
    request: AllocationRequest,
    base_ir: AllocationIR,
    context: ConstraintContext,
    *,
    config: ConstraintConfig,
    costs: CostTable,
) -> TerrainResult:
    """Derive terrain keep-out constraints and per-pair durations for a compiled request."""
    policy = config.terrain
    pairs = assignment_pairs(base_ir)
    tasks = _task_by_id(request)

    forbidden: set[Pair] = set()
    findings: list[ConstraintFinding] = []
    degraded: set[str] = set()
    durations: dict[Pair, float] = {}

    for task_id in sorted(pairs):
        task = tasks[task_id]
        eligible = pairs[task_id]

        # Resolve each pair's duration: the cached cost table first (the measured, asset-specific
        # truth), then the task's own declared nominal ``duration_s``, then a degraded-flagged zero.
        # Allocate never *invents* a duration — it falls back to a declared one and says so.
        for asset_id in eligible:
            entry = costs.lookup(task_id, asset_id)
            if entry is not None and entry.duration_s is not None:
                durations[(task_id, asset_id)] = entry.duration_s
            elif task.duration_s > 0.0:
                durations[(task_id, asset_id)] = task.duration_s
                degraded.add("cost.nominal_duration")
            else:
                durations[(task_id, asset_id)] = 0.0
                degraded.add("cost.missing_duration")

        # Keep-out needs a world sample at the task's location. No location or no WorldProvider ⇒
        # un-assessable: degrade loudly, forbid nothing (a false keep-out is worse than none here).
        if task.location is None:
            continue
        if context.world is None:
            degraded.add("terrain.no_world_data")
            continue

        center = task.location.center_m
        sample: SurfacePoint = context.world.sample((center.x, center.y, center.z))
        task_slope = slope_deg(sample.surface_normal, sample.gravity)
        shadowed = sample.illumination.state is IlluminationState.SHADOW
        bearing = sample.regolith.bearing_capacity_pa

        for asset_id in eligible:
            asset = context.assets.get(asset_id)
            reasons: list[ConstraintFinding] = []

            max_slope = _asset_max_slope_deg(asset, policy)
            if task_slope > max_slope:
                reasons.append(
                    ConstraintFinding(
                        code="terrain.slope_keepout",
                        detail=f"slope {task_slope:.1f}° exceeds {asset_id} limit {max_slope:.1f}°",
                        task_id=task_id,
                        asset_id=asset_id,
                        constraint_id=keepout_constraint_id(task_id, asset_id),
                    )
                )

            if policy.require_illuminated and shadowed:
                reasons.append(
                    ConstraintFinding(
                        code="terrain.shadow_keepout",
                        detail=f"{task_id} location is in shadow but the task requires light",
                        task_id=task_id,
                        asset_id=asset_id,
                        constraint_id=keepout_constraint_id(task_id, asset_id),
                    )
                )

            pressure = _asset_ground_pressure_pa(asset)
            if (
                policy.enforce_bearing_capacity
                and pressure is not None
                and bearing is not None
                and pressure * policy.ground_pressure_margin > bearing
            ):
                reasons.append(
                    ConstraintFinding(
                        code="terrain.bearing_keepout",
                        detail=(
                            f"{asset_id} ground pressure {pressure:.0f} Pa exceeds regolith "
                            f"bearing capacity {bearing:.0f} Pa"
                        ),
                        task_id=task_id,
                        asset_id=asset_id,
                        constraint_id=keepout_constraint_id(task_id, asset_id),
                    )
                )

            if reasons:
                forbidden.add((task_id, asset_id))
                findings.extend(reasons)

    # The ``assign(task, asset) = 0`` keep-out constraints are emitted once in ``compose`` over the
    # union of every builder's forbidden pairs, so terrain and comms never collide on an id.
    return TerrainResult(
        forbidden=frozenset(forbidden),
        findings=tuple(findings),
        degraded=tuple(sorted(degraded)),
        durations=durations,
    )
