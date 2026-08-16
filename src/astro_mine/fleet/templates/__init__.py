"""Parametric asset families — the broadened orbital/surface/manipulation/logistics/isru
templates (RM-P1-FLEET-10).

Each :class:`~astro_mine.fleet.params.Family` here is a base SADF skeleton plus a
range-checked parameter set and **derived-quantity rules** (mass scales drive inertia,
power draw, contact footprint, ISRU throughput — ``fleet.md`` §3 "Asset template"). The
ranges deliberately extend **beyond the anchor minimum** the P0 reference library pins
(``fleet.md`` §12 Phase 1: "broaden the parametric families"): a "10-500 kg rover" is one
family, not fifty files (``fleet.md`` §2.3). The anchor library assets remain the curated
exemplars; these families generate the wider design space around them.

Derived quantities are engineered so a resolved asset is **schema-valid and physically
plausible across the whole range** (``fleet.md`` §7, §10): a solid-cuboid inertia tensor is
positive-definite by construction, and mode power draws are fractions of the declared supply
so the power balance holds at every parameter value. Every family also **applies the richer
Core capability taxonomy** (``fleet.md`` §2.5) — autonomy-negotiation tags spanning mobility,
excavation, manipulation, sensing, comms, power, and ISRU — validated against Core's closed
vocabulary (:mod:`astro_mine.fleet.capabilities`).

Resolution is deterministic (:meth:`Family.resolve`); the same bindings yield the same
canonical SADF bytes.

Backlog: RM-P1-FLEET-10 -- astro-mine-fleet#21
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from astro_mine.core.sadf import SadfDocument
from astro_mine.fleet._core import CORE_INTERFACES
from astro_mine.fleet.capabilities import as_tags
from astro_mine.fleet.params import Family, ParamError, ParamSpec

__all__ = ["FAMILIES", "families", "get_family", "resolve_family"]

_SADF = CORE_INTERFACES["sadf"]

# --- excavator blade reference geometry -------------------------------------------
#
# The `manipulation-excavator` family's digging blade, stated once at the reference chassis mass and
# scaled per-variant in `_build_manipulation`. These are the library excavator's numbers
# (`library/manipulation/excavator.sadf.yaml`), so the family's default variant *is* that asset's
# bucket rather than a second, silently-divergent opinion about how big an excavator bucket is.
#: Chassis mass the blade dimensions below are quoted at (kg) — the family's `chassis_mass_kg`
#: default and the library excavator's chassis.
_BLADE_REFERENCE_MASS_KG = 200.0
#: Cutting width (m) — the span of soil the blade engages. A granular engine sizes its particle bed
#: from this.
_BLADE_WIDTH_M = 0.40
#: Fore-aft depth of the bucket pan (m).
_BLADE_DEPTH_M = 0.35
#: Blade height (m) — how deep the tool can cut in one pass. A granular engine sizes its tool from
#: this.
_BLADE_HEIGHT_M = 0.15
#: Lever arm (m) from the bucket joint to the cutting edge — turns the bucket motor's torque into
#: the force the blade can actually put into the ground.
_BLADE_LEVER_M = 0.5


# --- derived-quantity helpers ----------------------------------------------------


def _r(value: float, digits: int = 3) -> float:
    """Round a derived quantity to a stable precision (deterministic canonical bytes)."""
    return round(float(value), digits)


def _dims(
    mass: float, density: float, aspect: tuple[float, float, float]
) -> tuple[float, float, float]:
    """A characteristic bounding box (m) for *mass* at *density* (kg/m³), scaled per axis."""
    side = (mass / density) ** (1.0 / 3.0)
    return (_r(side * aspect[0]), _r(side * aspect[1]), _r(side * aspect[2]))


def _box_inertia(mass: float, dims: tuple[float, float, float]) -> dict[str, float]:
    """Solid-cuboid inertia about the COM — positive-definite for any positive *dims*."""
    lx, ly, lz = dims
    return {
        "ixx": _r(mass * (ly * ly + lz * lz) / 12.0),
        "iyy": _r(mass * (lx * lx + lz * lz) / 12.0),
        "izz": _r(mass * (lx * lx + ly * ly) / 12.0),
    }


def _body(
    name: str, frame: str, mass: float, dims: tuple[float, float, float], com_z: float
) -> dict[str, Any]:
    return {
        "name": name,
        "frame": frame,
        "mass_kg": _r(mass),
        "center_of_mass_m": {"x": 0.0, "y": 0.0, "z": _r(com_z)},
        "inertia_kg_m2": _box_inertia(mass, dims),
    }


def _solar_power(
    solar_w: float,
    loads: Sequence[tuple[str, float]],
    *,
    rhu_w: float = 0.0,
    discharge_frac: float = 0.8,
) -> dict[str, Any]:
    """A self-powered budget: solar (+ optional RHU) source, battery, and fractional loads.

    Each load is a fraction of the solar power; kept below ``1 + discharge_frac`` so the peak
    demand never exceeds ``solar + battery discharge`` — the power balance holds for every
    parameter value (lint ``power.deficit``/``power.floor``).
    """
    sources = [{"name": "body_solar", "kind": "solar", "nominal_power_w": _r(solar_w)}]
    if rhu_w:
        sources.append({"name": "rhu_pack", "kind": "rhu", "nominal_power_w": _r(rhu_w)})
    storage = [
        {
            "name": "main_battery",
            "capacity_j": _r(solar_w * 2.0e4),
            "max_charge_w": _r(0.5 * solar_w),
            "max_discharge_w": _r(discharge_frac * solar_w),
        }
    ]
    return {
        "sources": sources,
        "storage": storage,
        "floor_w": _r(0.1 * solar_w),
        "loads_by_mode": [{"mode": mode, "power_w": _r(frac * solar_w)} for mode, frac in loads],
    }


def _identity(
    family: str, variant: str, version: str, kind: str, name: str, desc: str
) -> dict[str, Any]:
    return {
        # Bare kebab-case, no component prefix, no version (conventions.md §13). This used to
        # mint `astro-mine.fleet.<family>.<variant>`, so every templated asset was born
        # non-conforming and the gate at publish would have refused it.
        "id": f"{family}-{variant}",
        "name": name,
        "version": version,
        "kind": kind,
        "description": desc,
        "labels": {"family": family},
    }


def _capabilities(tags: Sequence[str]) -> list[str]:
    """Validate applied tags against Core's vocabulary and return their canonical values."""
    return [tag.value for tag in as_tags(tags)]


def _thermal(op_min: float, op_max: float, dissipation: float) -> dict[str, Any]:
    return {
        "operating_range_k": {"min": op_min, "max": op_max},
        "survival_range_k": {"min": op_min - 30.0, "max": op_max + 30.0},
        "dissipation_w": _r(dissipation),
        "surface_coupling": True,
    }


# --- family builders -------------------------------------------------------------


def _build_orbital(variant: str, version: str, p: Mapping[str, float]) -> dict[str, Any]:
    mass = p["bus_mass_kg"]
    solar = p["solar_power_w"]
    dims = _dims(mass, 150.0, (1.0, 1.0, 0.8))
    return {
        "identity": _identity(
            "orbital-relay",
            variant,
            version,
            "orbiter",
            "Relay Orbiter",
            "Parametric relay/comms orbiter: store-and-forward relay + direct-to-Earth downlink.",
        ),
        "capabilities": _capabilities(
            [
                "mobility.orbiter",
                "comms.relay",
                "comms.direct_to_earth",
                "comms.dtn",
                "comms.dsn",
                "power.generation",
                "power.storage",
                "power.distribution",
                "sensing.imaging",
                "sensing.imu",
            ]
        ),
        "core_interface_versions": {"sadf": _SADF},
        "root_frame": "bus",
        "frames": [
            {"name": "bus"},
            {
                "name": "hga",
                "parent": "bus",
                "transform": {
                    "translation_m": {"x": 0.6, "y": 0.0, "z": 0.0},
                    "rotation_quat_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        ],
        "bodies": [_body("bus", "bus", mass, dims, 0.0)],
        "power": _solar_power(
            solar, [("safe", 0.15), ("nominal", 0.45), ("downlink", 0.8)], discharge_frac=0.8
        ),
        "thermal": _thermal(250.0, 320.0, 0.3 * solar),
        "sensors": [
            {
                "name": "star_tracker",
                "kind": "imaging",
                "frame": "bus",
                "observation_model": {"range_m": 1.0e9, "fov_deg": 8.0},
            },
            {"name": "nav_imu", "kind": "imu", "frame": "bus"},
        ],
        "comms": [
            {
                "name": "s_band_relay",
                "band": "s_band",
                "node_role": "space",
                "antenna": {
                    "gain_dbi": 12.0,
                    "boresight_frame": "hga",
                    "pointing_accuracy_deg": 1.0,
                },
                "tx_power_w": _r(0.06 * solar),
                "protocols": ["ccsds_tc_tm", "cfdp"],
                "relay": True,
            },
            {
                "name": "x_band_dte",
                "band": "x_band",
                "node_role": "space",
                "antenna": {
                    "gain_dbi": 28.0,
                    "boresight_frame": "hga",
                    "pointing_accuracy_deg": 0.3,
                },
                "tx_power_w": _r(0.1 * solar),
                "protocols": ["ccsds_tc_tm"],
                "relay": False,
            },
        ],
        "mobility": {"regimes": ["proximity_orbit"]},
        "fidelity_profiles": [
            {
                "tier": "massmodel",
                "determinism_class": "bit_exact",
                "detail": "bus point-mass + power",
            },
            {
                "tier": "kinematic",
                "determinism_class": "bit_exact",
                "detail": "rigid bus + attitude",
            },
        ],
    }


def _build_surface(variant: str, version: str, p: Mapping[str, float]) -> dict[str, Any]:
    mass = p["chassis_mass_kg"]
    solar = p["solar_power_w"]
    reach = p["sensor_range_m"]
    dims = _dims(mass, 400.0, (1.2, 1.0, 0.6))
    return {
        "identity": _identity(
            "surface-rover",
            variant,
            version,
            "rover",
            "Surface Rover",
            "Parametric wheeled surface rover with a nav/perception sensor suite.",
        ),
        "capabilities": _capabilities(
            [
                "mobility.wheeled",
                "sensing.imaging",
                "sensing.imu",
                "sensing.odometry",
                "sensing.lidar",
                "sensing.ranging",
                "power.generation",
                "power.storage",
            ]
        ),
        "core_interface_versions": {"sadf": _SADF},
        "root_frame": "body",
        "frames": [
            {"name": "body"},
            {
                "name": "mast",
                "parent": "body",
                "transform": {
                    "translation_m": {"x": 0.0, "y": 0.0, "z": _r(dims[2] + 0.5)},
                    "rotation_quat_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        ],
        "bodies": [_body("chassis", "body", mass, dims, 0.5 * dims[2])],
        "power": _solar_power(
            solar,
            [("idle", 0.15), ("drive", 0.6), ("perceive", 0.5)],
            rhu_w=8.0,
            discharge_frac=0.9,
        ),
        "thermal": _thermal(120.0, 330.0, 0.3 * solar),
        "sensors": [
            {"name": "nav_imu", "kind": "imu", "frame": "body"},
            {"name": "wheel_odometry", "kind": "odometry", "frame": "body"},
            {
                "name": "hazcam",
                "kind": "imaging",
                "frame": "mast",
                "observation_model": {"range_m": _r(reach), "fov_deg": 70.0},
            },
            {
                "name": "nav_lidar",
                "kind": "lidar",
                "frame": "mast",
                "observation_model": {
                    "range_m": _r(3.0 * reach),
                    "fov_deg": 360.0,
                    "noise_sigma": 0.02,
                },
            },
        ],
        "mobility": {
            "regimes": ["surface"],
            "contact": [
                {
                    "kind": "wheel",
                    "dimensions_m": {"x": 0.30, "y": 0.15, "z": 0.30},
                    "footprint_m2": 0.045,
                    "max_ground_pressure_pa": 12000.0,
                    "max_slope_deg": 25.0,
                }
            ],
        },
        "payload": {"capacity_kg": _r(0.05 * mass)},
        "fidelity_profiles": [
            {
                "tier": "massmodel",
                "determinism_class": "bit_exact",
                "detail": "chassis point-mass + power",
            },
            {
                "tier": "kinematic",
                "determinism_class": "bit_exact",
                "detail": "rigid chassis + wheels",
            },
        ],
    }


def _build_manipulation(variant: str, version: str, p: Mapping[str, float]) -> dict[str, Any]:
    mass = p["chassis_mass_kg"]
    solar = p["solar_power_w"]
    torque = p["bucket_torque_nm"]
    dims = _dims(mass, 500.0, (1.3, 1.0, 0.6))
    arm_mass = 0.2 * mass
    arm_dims = _dims(arm_mass, 800.0, (2.0, 0.6, 0.6))
    # The blade is a *length*, so it scales with the cube root of chassis mass — the same rule
    # `_dims` applies to every other bounding box here. Referenced to the library excavator, whose
    # bucket these constants are (200 kg chassis, 2000 N.m bucket motor).
    blade_scale = (mass / _BLADE_REFERENCE_MASS_KG) ** (1.0 / 3.0)
    blade_footprint_m2 = (_BLADE_WIDTH_M * blade_scale) * (_BLADE_DEPTH_M * blade_scale)
    return {
        "identity": _identity(
            "manipulation-excavator",
            variant,
            version,
            "excavator",
            "Excavator Rover",
            "Parametric wheeled excavator with a single-DOF bucket for digging regolith.",
        ),
        "capabilities": _capabilities(
            [
                "mobility.wheeled",
                "excavation.bucket",
                "excavation.drill",
                "manipulation.arm",
                "sensing.imu",
                "sensing.imaging",
                "power.generation",
                "power.storage",
            ]
        ),
        "core_interface_versions": {"sadf": _SADF},
        "root_frame": "body",
        "frames": [
            {"name": "body"},
            {
                "name": "bucket",
                "parent": "body",
                "transform": {
                    "translation_m": {"x": _r(dims[0]), "y": 0.0, "z": 0.2},
                    "rotation_quat_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        ],
        "bodies": [
            _body("chassis", "body", mass, dims, 0.5 * dims[2]),
            _body("bucket_arm", "bucket", arm_mass, arm_dims, 0.0),
        ],
        "joints": [
            {
                "name": "bucket_pitch",
                "type": "revolute",
                "parent_body": "chassis",
                "child_body": "bucket_arm",
                "axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                "limits": {
                    "position_rad": {"min": -0.2, "max": 1.4},
                    "velocity_rad_s": 0.6,
                    "effort_nm": _r(torque),
                },
            }
        ],
        "actuators": [
            {
                "name": "bucket_motor",
                "target_joint": "bucket_pitch",
                "torque_nm": _r(torque),
                "velocity": 0.6,
                "power_draw_w": _r(0.4 * torque),
            }
        ],
        "power": _solar_power(
            solar,
            [("idle", 0.16), ("drive", 0.56), ("excavate", 0.9)],
            rhu_w=8.0,
            discharge_frac=0.9,
        ),
        "thermal": _thermal(120.0, 330.0, 0.35 * solar),
        "sensors": [
            {"name": "nav_imu", "kind": "imu", "frame": "body"},
            {
                "name": "bucket_cam",
                "kind": "imaging",
                "frame": "bucket",
                "observation_model": {"range_m": 8.0, "fov_deg": 60.0},
            },
        ],
        "mobility": {
            "regimes": ["surface"],
            "contact": [
                {
                    "kind": "wheel",
                    "dimensions_m": {"x": 0.35, "y": 0.20, "z": 0.35},
                    "footprint_m2": 0.07,
                    "max_ground_pressure_pa": 15000.0,
                    "max_slope_deg": 20.0,
                },
                # The digging blade — the element that makes this family an *excavator* to a
                # physics engine rather than a rover carrying a bucket-shaped decoration. It is the
                # sole declaration that routes an asset to a granular (DEM / learned-surrogate)
                # contact model, so a parametric excavator without one cannot reach the tier it
                # exists to exercise (fleet#37; mirrors `library/manipulation/excavator.sadf.yaml`).
                #
                # Scaled, not pasted: a blade is a linear dimension, so it grows with the cube root
                # of chassis mass (the same rule `_dims` uses), and its pressure ceiling is
                # *derived* — the bucket motor's torque through the blade's lever arm, spread over
                # the pan. At the family defaults (200 kg, 2000 N.m) that lands on the library
                # excavator's 0.40 x 0.35 x 0.15 m pan and its ~29 kPa ceiling.
                {
                    "kind": "tool",
                    "dimensions_m": {
                        "x": _r(_BLADE_WIDTH_M * blade_scale),
                        "y": _r(_BLADE_DEPTH_M * blade_scale),
                        "z": _r(_BLADE_HEIGHT_M * blade_scale),
                    },
                    "footprint_m2": _r(blade_footprint_m2),
                    "max_ground_pressure_pa": _r(
                        torque / (_BLADE_LEVER_M * blade_scale) / blade_footprint_m2
                    ),
                    # It digs on gentler ground than it drives on: breaking traction mid-cut is how
                    # an excavator tips over.
                    "max_slope_deg": 15.0,
                },
            ],
        },
        "payload": {"capacity_kg": _r(0.25 * mass)},
        "fidelity_profiles": [
            {
                "tier": "massmodel",
                "determinism_class": "bit_exact",
                "detail": "chassis+bucket point-mass",
            },
            {
                "tier": "kinematic",
                "determinism_class": "bit_exact",
                "detail": "rigid chassis + bucket",
            },
            {
                "tier": "articulated",
                "determinism_class": "tolerance",
                "detail": "bucket joint + contact",
            },
        ],
    }


def _build_logistics(variant: str, version: str, p: Mapping[str, float]) -> dict[str, Any]:
    mass = p["chassis_mass_kg"]
    solar = p["solar_power_w"]
    cargo = p["cargo_capacity_kg"]
    dims = _dims(mass, 500.0, (1.4, 1.0, 0.6))
    return {
        "identity": _identity(
            "logistics-hauler",
            variant,
            version,
            "hauler",
            "Hauler Rover",
            "Parametric high-capacity wheeled hauler that carries regolith between sites.",
        ),
        "capabilities": _capabilities(
            [
                "mobility.wheeled",
                "sensing.imu",
                "sensing.odometry",
                "sensing.imaging",
                "power.generation",
                "power.storage",
                "power.distribution",
                "carrier.dispenser",
                "reusable",
            ]
        ),
        "core_interface_versions": {"sadf": _SADF},
        "root_frame": "body",
        "frames": [{"name": "body"}],
        "bodies": [_body("chassis", "body", mass, dims, 0.5 * dims[2])],
        "power": _solar_power(
            solar,
            [("idle", 0.15), ("drive_empty", 0.55), ("drive_loaded", 0.9)],
            discharge_frac=0.9,
        ),
        "thermal": _thermal(120.0, 330.0, 0.3 * solar),
        "sensors": [
            {"name": "nav_imu", "kind": "imu", "frame": "body"},
            {
                "name": "nav_cam",
                "kind": "imaging",
                "frame": "body",
                "observation_model": {"range_m": 25.0, "fov_deg": 80.0},
            },
        ],
        "mobility": {
            "regimes": ["surface"],
            "contact": [
                {
                    "kind": "wheel",
                    "dimensions_m": {"x": 0.40, "y": 0.25, "z": 0.40},
                    "footprint_m2": 0.10,
                    "max_ground_pressure_pa": 18000.0,
                    "max_slope_deg": 18.0,
                }
            ],
        },
        "payload": {
            "capacity_kg": _r(cargo),
            "slots": [
                {
                    "name": "cargo_bin",
                    "frame": "body",
                    "accepts": ["regolith"],
                    "max_mass_kg": _r(cargo),
                }
            ],
        },
        "fidelity_profiles": [
            {
                "tier": "massmodel",
                "determinism_class": "bit_exact",
                "detail": "chassis point-mass + cargo",
            },
            {
                "tier": "kinematic",
                "determinism_class": "bit_exact",
                "detail": "rigid chassis + wheels",
            },
        ],
    }


def _build_isru(variant: str, version: str, p: Mapping[str, float]) -> dict[str, Any]:
    mass = p["plant_mass_kg"]
    throughput = p["throughput_kg_hr"]
    plant_power = p["plant_power_w"]
    dims = _dims(mass, 400.0, (1.0, 1.0, 1.2))
    return {
        "identity": _identity(
            "isru-plant",
            variant,
            version,
            "isru_plant",
            "ISRU Plant",
            "Parametric fixed ISRU plant: thermal extraction, purification, electrolysis, storage.",
        ),
        "capabilities": _capabilities(
            [
                "isru.thermal_extraction",
                "isru.electrolysis",
                "isru.purification",
                "isru.storage",
                "power.distribution",
                "sensing.thermal",
                "sensing.imaging",
            ]
        ),
        "core_interface_versions": {"sadf": _SADF},
        "root_frame": "base",
        "frames": [
            {"name": "base"},
            {
                "name": "hopper",
                "parent": "base",
                "transform": {
                    "translation_m": {"x": 0.0, "y": 0.0, "z": _r(dims[2])},
                    "rotation_quat_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        ],
        "bodies": [_body("plant_chassis", "base", mass, dims, 0.5 * dims[2])],
        # External surface power infrastructure: loads only (no on-board supply) -> the power
        # balance check does not apply (a deliberate modelling choice; see isru-plant anchor).
        "power": {
            "loads_by_mode": [
                {"mode": "idle", "power_w": _r(0.07 * plant_power)},
                {"mode": "extract", "power_w": _r(plant_power)},
                {"mode": "purify", "power_w": _r(0.4 * plant_power)},
            ]
        },
        "thermal": _thermal(200.0, 420.0, 0.3 * plant_power),
        "sensors": [
            {
                "name": "hopper_temp",
                "kind": "thermal_sensor",
                "frame": "hopper",
                "observation_model": {"noise_sigma": 0.5},
            },
            {
                "name": "feed_cam",
                "kind": "imaging",
                "frame": "hopper",
                "observation_model": {"range_m": 5.0, "fov_deg": 50.0},
            },
            {
                "name": "water_gauge",
                "kind": "resource_storage",
                "frame": "base",
                # `kg` is the Core unit token; `mass_kg` is not one, and a gauge declaring it
                # renders readings Bench's `water_mass` silently skips (astro-mine-fleet#40).
                "resource": {"species": "water", "si_unit": "kg"},
            },
        ],
        "payload": {
            "capacity_kg": _r(2.0 * mass),
            "isru": {"throughput_kg_hr": _r(throughput), "plant_power_w": _r(plant_power)},
        },
        "fidelity_profiles": [
            {
                "tier": "massmodel",
                "determinism_class": "bit_exact",
                "detail": "plant point-mass + throughput",
            },
            {
                "tier": "kinematic",
                "determinism_class": "bit_exact",
                "detail": "rigid plant, fixed pose",
            },
        ],
    }


# --- family registry -------------------------------------------------------------

FAMILIES: dict[str, Family] = {
    "orbital-relay": Family(
        name="orbital-relay",
        kind="orbiter",
        summary="Relay/comms orbiter (bus mass, solar power, battery).",
        params=(
            ParamSpec("bus_mass_kg", 100.0, 2000.0, 250.0, "kg", "spacecraft bus dry mass"),
            ParamSpec("solar_power_w", 200.0, 4000.0, 600.0, "W", "solar array nominal power"),
            ParamSpec("battery_capacity_mj", 1.0, 100.0, 5.0, "MJ", "battery energy capacity"),
        ),
        build_asset=_build_orbital,
    ),
    "surface-rover": Family(
        name="surface-rover",
        kind="rover",
        summary="Wheeled surface rover (chassis mass, solar power, sensor range).",
        params=(
            ParamSpec("chassis_mass_kg", 10.0, 500.0, 180.0, "kg", "chassis dry mass"),
            ParamSpec("solar_power_w", 50.0, 800.0, 210.0, "W", "solar array nominal power"),
            ParamSpec("sensor_range_m", 5.0, 100.0, 20.0, "m", "hazcam sensing range"),
        ),
        build_asset=_build_surface,
    ),
    "manipulation-excavator": Family(
        name="manipulation-excavator",
        kind="excavator",
        summary="Wheeled excavator (chassis mass, bucket torque, solar power).",
        params=(
            ParamSpec("chassis_mass_kg", 100.0, 1000.0, 200.0, "kg", "chassis dry mass"),
            ParamSpec("bucket_torque_nm", 500.0, 8000.0, 2000.0, "Nm", "bucket joint peak torque"),
            ParamSpec("solar_power_w", 100.0, 1200.0, 250.0, "W", "solar array nominal power"),
        ),
        build_asset=_build_manipulation,
    ),
    "logistics-hauler": Family(
        name="logistics-hauler",
        kind="hauler",
        summary="High-capacity wheeled hauler (chassis mass, cargo capacity, solar power).",
        params=(
            ParamSpec("chassis_mass_kg", 100.0, 2000.0, 300.0, "kg", "chassis dry mass"),
            ParamSpec("cargo_capacity_kg", 50.0, 3000.0, 500.0, "kg", "cargo/regolith capacity"),
            ParamSpec("solar_power_w", 100.0, 1000.0, 300.0, "W", "solar array nominal power"),
        ),
        build_asset=_build_logistics,
    ),
    "isru-plant": Family(
        name="isru-plant",
        kind="isru_plant",
        summary="Fixed ISRU plant (plant mass, throughput, plant power).",
        params=(
            ParamSpec("plant_mass_kg", 200.0, 5000.0, 600.0, "kg", "plant dry mass"),
            ParamSpec("throughput_kg_hr", 1.0, 200.0, 10.0, "kg/hr", "water throughput"),
            ParamSpec("plant_power_w", 500.0, 20000.0, 1500.0, "W", "peak extraction power"),
        ),
        build_asset=_build_isru,
    ),
}


def families() -> list[str]:
    """The parametric family handles, sorted (the menu :func:`resolve_family` accepts)."""
    return sorted(FAMILIES)


def get_family(name: str) -> Family:
    """The :class:`~astro_mine.fleet.params.Family` named *name*.

    Raises :class:`~astro_mine.fleet.params.ParamError` for an unknown family (with the menu).
    """
    try:
        return FAMILIES[name]
    except KeyError:
        raise ParamError(f"unknown family {name!r}; available: {families()}") from None


def resolve_family(
    name: str,
    overrides: Mapping[str, float] | None = None,
    *,
    variant: str = "custom",
    version: str = "0.1.0",
) -> SadfDocument:
    """Resolve family *name* with *overrides* to a concrete, Core-validated SADF document."""
    return get_family(name).resolve(overrides, variant=variant, version=version)
