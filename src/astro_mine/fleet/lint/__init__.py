# SPDX-License-Identifier: Apache-2.0
"""Physical-plausibility lint.

Catches assets that are *schema-valid but physically impossible* — the tier above
Core's structural/semantic gate (:func:`astro_mine.core.sadf.validate_sadf`). Three
rule groups, mirroring the conventions Fleet enforces (CONTRIBUTING.md; fleet.md §10):

- **inertia / mass** — a rigid body needs positive mass and a positive-definite
  inertia tensor (``mass.positive``, ``inertia.positive_definite``);
- **power balance** — declared power cannot be negative, and an asset that declares
  its own supply must be able to meet its worst-case load and hold its power floor
  (``power.negative``, ``power.deficit``, ``power.floor``);
- **sensor sanity** — field-of-view, range, noise, and footprint values must be
  physically sensible (``sensor.fov``, ``sensor.range``, ``sensor.noise``,
  ``sensor.footprint``, ``sensor.depth``).

It runs over a single resolved :class:`~astro_mine.core.sadf.model.Asset` — the same
object a resolved template (or an imported description) produces — so it is the engine
``fleet lint`` runs over assets and, when a parameter sampler lands, over sampled
template parameters too. Sub-assembly contents are linted when the referenced asset is
itself linted (no ref resolution here).

Backlog: RM-P0-FLEET-03 -- astro-mine-fleet#3
"""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.core.sadf.model import Asset, Body, PowerBudget, Sensor
from astro_mine.core.units import is_si_unit

__all__ = ["PlausibilityFinding", "lint_asset"]


@dataclass(frozen=True)
class PlausibilityFinding:
    """A single physical-plausibility problem.

    ``rule`` is a stable, dotted rule id (e.g. ``inertia.positive_definite``); ``path``
    locates the offending element within the asset (a JSON-pointer-ish dotted path, e.g.
    ``asset.bodies[0].inertia_kg_m2``); ``message`` is human-readable. All findings are
    errors in Phase 0 — any finding fails the lint.
    """

    rule: str
    path: str
    message: str


def lint_asset(asset: Asset) -> list[PlausibilityFinding]:
    """Run physical-plausibility checks over a resolved SADF asset.

    Returns the findings in document order; an empty list means the asset is plausible.
    Pure structural reasoning over declared values — no physics engine, no dependencies.
    """
    findings: list[PlausibilityFinding] = []
    for i, body in enumerate(asset.bodies):
        findings.extend(_check_body(body, f"asset.bodies[{i}]"))
    if asset.power is not None:
        findings.extend(_check_power(asset.power, "asset.power"))
    for i, sensor in enumerate(asset.sensors):
        findings.extend(_check_sensor(sensor, f"asset.sensors[{i}]"))
    return findings


def _check_resource_target(sensor: Sensor, path: str) -> list[PlausibilityFinding]:
    """A sensor's declared resource unit must be a unit the platform knows.

    `ResourceTarget.si_unit` is a free string in the schema, so a plausible-looking token like
    ``mass_kg`` validates and then silently fails to score: Sim renders the gauge's *declared*
    unit verbatim, and Bench matches `water_mass` on the exact token `kg`. The reading is emitted,
    filtered out, and the tank reads as empty — a full plant indistinguishable from an idle one
    (astro-mine-fleet#40). Cheap to state here, and it fails the asset instead of the scorecard.
    """
    resource = sensor.resource
    if resource is None or is_si_unit(resource.si_unit):
        return []
    return [
        PlausibilityFinding(
            "sensor.resource_unit",
            f"{path}.resource.si_unit",
            f"sensor {sensor.name!r} declares si_unit={resource.si_unit!r}, which is not a known "
            "unit token — a consumer matching on the platform vocabulary will skip its readings",
        )
    ]


# --- inertia / mass --------------------------------------------------------------


def _check_body(body: Body, path: str) -> list[PlausibilityFinding]:
    findings: list[PlausibilityFinding] = []
    if body.mass_kg <= 0.0:
        findings.append(
            PlausibilityFinding(
                "mass.positive",
                path,
                f"body {body.name!r} has non-positive mass_kg={body.mass_kg}",
            )
        )
    if not _is_positive_definite(body):
        findings.append(
            PlausibilityFinding(
                "inertia.positive_definite",
                f"{path}.inertia_kg_m2",
                f"body {body.name!r} inertia tensor is not positive-definite "
                f"(Sylvester's criterion: a leading principal minor is <= 0)",
            )
        )
    return findings


def _is_positive_definite(body: Body) -> bool:
    """Sylvester's criterion: a symmetric matrix is positive-definite iff every leading
    principal minor is strictly positive. Exact and deterministic — no eigensolver."""
    t = body.inertia_kg_m2
    a, b, c = t.ixx, t.iyy, t.izz
    d, e, f = t.ixy, t.ixz, t.iyz
    minor1 = a
    minor2 = a * b - d * d
    minor3 = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)
    return minor1 > 0.0 and minor2 > 0.0 and minor3 > 0.0


# --- power balance ---------------------------------------------------------------


def _check_power(power: PowerBudget, path: str) -> list[PlausibilityFinding]:
    findings: list[PlausibilityFinding] = []

    def negative(value: float | None) -> bool:
        return value is not None and value < 0.0

    # Non-negativity: negative power is impossible regardless of architecture.
    for i, source in enumerate(power.sources):
        if negative(source.nominal_power_w):
            findings.append(
                PlausibilityFinding(
                    "power.negative",
                    f"{path}.sources[{i}]",
                    f"source {source.name!r} has negative nominal_power_w={source.nominal_power_w}",
                )
            )
    for i, store in enumerate(power.storage):
        for field, value in (
            ("capacity_j", store.capacity_j),
            ("max_charge_w", store.max_charge_w),
            ("max_discharge_w", store.max_discharge_w),
        ):
            if negative(value):
                findings.append(
                    PlausibilityFinding(
                        "power.negative",
                        f"{path}.storage[{i}]",
                        f"storage {store.name!r} has negative {field}={value}",
                    )
                )
    for i, load in enumerate(power.loads_by_mode):
        if negative(load.power_w):
            findings.append(
                PlausibilityFinding(
                    "power.negative",
                    f"{path}.loads_by_mode[{i}]",
                    f"mode {load.mode!r} has negative power_w={load.power_w}",
                )
            )
    if negative(power.floor_w):
        findings.append(
            PlausibilityFinding("power.negative", path, f"floor_w is negative ({power.floor_w})")
        )

    # Adequacy: only assets that declare their own supply are held to it — an
    # externally-powered payload may declare loads with no on-board source.
    declares_supply = bool(power.sources) or bool(power.storage)
    if declares_supply:
        peak_supply = sum(s.nominal_power_w for s in power.sources) + sum(
            (st.max_discharge_w or 0.0) for st in power.storage
        )
        peak_demand = max((m.power_w for m in power.loads_by_mode), default=0.0)
        if peak_demand > peak_supply:
            findings.append(
                PlausibilityFinding(
                    "power.deficit",
                    path,
                    f"peak load {peak_demand} W exceeds peak supply {peak_supply} W "
                    f"(sources + declared storage discharge)",
                )
            )
        if power.floor_w is not None and power.floor_w > peak_supply:
            findings.append(
                PlausibilityFinding(
                    "power.floor",
                    path,
                    f"power floor {power.floor_w} W exceeds peak supply {peak_supply} W",
                )
            )
    return findings


# --- sensor sanity ---------------------------------------------------------------


def _check_sensor(sensor: Sensor, path: str) -> list[PlausibilityFinding]:
    findings: list[PlausibilityFinding] = []
    findings.extend(_check_resource_target(sensor, path))
    om = sensor.observation_model
    if om is None:
        return findings
    obs_path = f"{path}.observation_model"
    if om.fov_deg is not None and not (0.0 < om.fov_deg <= 360.0):
        findings.append(
            PlausibilityFinding(
                "sensor.fov",
                obs_path,
                f"sensor {sensor.name!r} fov_deg={om.fov_deg} is outside (0, 360]",
            )
        )
    if om.range_m is not None and om.range_m <= 0.0:
        findings.append(
            PlausibilityFinding(
                "sensor.range",
                obs_path,
                f"sensor {sensor.name!r} range_m={om.range_m} is not positive",
            )
        )
    if om.noise_sigma is not None and om.noise_sigma < 0.0:
        findings.append(
            PlausibilityFinding(
                "sensor.noise",
                obs_path,
                f"sensor {sensor.name!r} noise_sigma={om.noise_sigma} is negative",
            )
        )
    if om.footprint_m2 is not None and om.footprint_m2 <= 0.0:
        findings.append(
            PlausibilityFinding(
                "sensor.footprint",
                obs_path,
                f"sensor {sensor.name!r} footprint_m2={om.footprint_m2} is not positive",
            )
        )
    if om.depth_response_m is not None and om.depth_response_m <= 0.0:
        findings.append(
            PlausibilityFinding(
                "sensor.depth",
                obs_path,
                f"sensor {sensor.name!r} depth_response_m={om.depth_response_m} is not positive",
            )
        )
    return findings
