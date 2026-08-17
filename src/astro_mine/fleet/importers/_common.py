# SPDX-License-Identifier: Apache-2.0
"""Shared importer core: a small intermediate representation (IR), the math that maps
robot-description conventions onto SADF, and the one SADF builder both parsers feed.

URDF and SDF differ only in *parsing*; once each produces an :class:`IRModel`, the body/
frame/joint/geometry construction — and the geometry export — is identical, so it lives
here once (RM-P0-FLEET-02).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import trimesh
from numpy.typing import NDArray

from astro_mine.core.sadf import SadfDocument, validate_sadf
from astro_mine.core.sadf.enums import GeometryRole, JointType
from astro_mine.core.sadf.model import (
    Asset,
    Body,
    Frame,
    Identity,
    Inertia,
    Joint,
    JointLimits,
    Quat,
    Transform,
    Vec3,
)
from astro_mine.fleet import geometry
from astro_mine.fleet._core import CORE_INTERFACES

_SADF_VERSION: Final = "0.1"  # Final -> Literal["0.1"], matching SadfDocument.sadf_version


class ImportError_(Exception):
    """A robot description could not be mapped onto SADF."""


# --- intermediate representation -------------------------------------------------


@dataclass
class IRGeometry:
    """A geometry already normalized into its body frame, tagged visual/collision."""

    role: GeometryRole
    mesh: trimesh.Trimesh


@dataclass
class IRLink:
    """A rigid link: its frame (relative to ``parent_frame``), mass properties, geometry.

    A link with no inertial (a massless/virtual frame, common in URDF) carries
    ``mass_kg=None`` and yields a frame + geometry but no body.
    """

    name: str
    parent_frame: str | None
    transform: Transform | None
    mass_kg: float | None
    com_m: Vec3 | None
    inertia: Inertia | None
    geometries: list[IRGeometry] = field(default_factory=list)


@dataclass
class IRJoint:
    name: str
    type: JointType
    parent_link: str
    child_link: str
    axis: Vec3 | None
    limits: JointLimits | None


@dataclass
class IRModel:
    name: str
    root_frame: str
    links: list[IRLink]
    joints: list[IRJoint]


# --- math: description conventions -> SADF ---------------------------------------


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Quat:
    """Fixed-axis roll-pitch-yaw (URDF/SDF convention, R = Rz·Ry·Rx) to a unit
    quaternion, scalar-last."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return Quat(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


def _rotmat_to_quat(r: NDArray[np.float64]) -> Quat:
    """A 3x3 rotation matrix to a unit quaternion (scalar-last), numerically robust."""
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] >= r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return Quat(x=x, y=y, z=z, w=w)


def matrix_to_transform(m: NDArray[np.float64]) -> Transform | None:
    """A 4x4 homogeneous matrix to a SADF :class:`Transform`, or ``None`` if it is the
    identity (so identity-posed frames stay clean)."""
    mat = np.asarray(m, dtype=float)
    if np.allclose(mat, np.eye(4)):
        return None
    translation = mat[:3, 3]
    return Transform(
        translation_m=Vec3(
            x=float(translation[0]), y=float(translation[1]), z=float(translation[2])
        ),
        rotation_quat_xyzw=_rotmat_to_quat(mat[:3, :3]),
    )


def inertia_in_body_frame(tensor: NDArray[np.float64], rotation: NDArray[np.float64]) -> Inertia:
    """Rotate a 3x3 inertia tensor from the inertial frame into the body frame
    (I_body = R·I·Rᵀ), then project to SADF's six independent components."""
    i = rotation @ np.asarray(tensor, dtype=float) @ rotation.T
    return Inertia(
        ixx=float(i[0, 0]),
        iyy=float(i[1, 1]),
        izz=float(i[2, 2]),
        ixy=float(i[0, 1]),
        ixz=float(i[0, 2]),
        iyz=float(i[1, 2]),
    )


def vec3(values: NDArray[np.float64]) -> Vec3:
    a = np.asarray(values, dtype=float).reshape(3)
    return Vec3(x=float(a[0]), y=float(a[1]), z=float(a[2]))


def resolve_mesh_path(uri: str, base_dir: Path) -> Path:
    """Resolve a URDF/SDF mesh reference to a local file.

    Strips ``package://<pkg>/`` and ``model://<pkg>/`` (the remainder is resolved
    against ``base_dir``) and ``file://``; absolute paths are kept as-is. Phase-0 keeps
    resolution local — no ROS/Gazebo package search paths (deferred, P1).
    """
    ref = uri
    for scheme in ("package://", "model://"):
        if ref.startswith(scheme):
            ref = ref[len(scheme) :].split("/", 1)[-1]  # drop scheme + package name
            break
    else:
        if ref.startswith("file://"):
            ref = ref[len("file://") :]
    path = Path(ref)
    return path if path.is_absolute() else base_dir / path


# --- IR -> SADF ------------------------------------------------------------------


def build_sadf(model: IRModel, *, assets_dir: Path, uri_prefix: str) -> SadfDocument:
    """Assemble (and validate) a SADF document from an :class:`IRModel`, writing each
    link's geometry as USD + glTF artifacts under ``assets_dir``."""
    link_frames = {link.name for link in model.links}
    frames: list[Frame] = []
    if model.root_frame not in link_frames:
        frames.append(Frame(name=model.root_frame, parent=None))

    bodies: list[Body] = []
    geometry_refs = []
    for link in model.links:
        frames.append(Frame(name=link.name, parent=link.parent_frame, transform=link.transform))
        if link.mass_kg is not None and link.com_m is not None and link.inertia is not None:
            bodies.append(
                Body(
                    name=link.name,
                    frame=link.name,
                    mass_kg=link.mass_kg,
                    center_of_mass_m=link.com_m,
                    inertia_kg_m2=link.inertia,
                )
            )
        for idx, g in enumerate(link.geometries):
            stem = f"{link.name}__{g.role.value}{idx}"
            geometry_refs.extend(
                geometry.write_geometry(
                    g.mesh,
                    role=g.role,
                    stem=stem,
                    frame=link.name,
                    assets_dir=assets_dir,
                    uri_prefix=uri_prefix,
                )
            )

    # SADF joints connect *bodies* (Core requires each joint endpoint to be a declared body;
    # loader referential-closure check, RM-P1-CORE-05). A URDF/SDF joint to a massless link
    # (e.g. a fixed sensor mount) has no body on that endpoint; that rigid attachment is
    # already carried by the link's frame parenthood, so we drop the (unrealizable) joint
    # rather than emit a dangling body reference.
    body_frames = {b.name for b in bodies}
    joints = [
        Joint(
            name=j.name,
            type=j.type,
            parent_body=j.parent_link,
            child_body=j.child_link,
            axis=j.axis,
            limits=j.limits,
        )
        for j in model.joints
        if j.parent_link in body_frames and j.child_link in body_frames
    ]

    asset = Asset(
        identity=_identity(model.name),
        core_interface_versions={"sadf": CORE_INTERFACES["sadf"]},
        frames=frames,
        root_frame=model.root_frame,
        geometry=geometry_refs,
        bodies=bodies,
        joints=joints,
    )
    doc = SadfDocument(sadf_version=_SADF_VERSION, asset=asset)
    validate_sadf(doc)  # an import that produced invalid SADF is a bug — fail loud
    return doc


#: The namespace an imported asset's identity is minted under. A description format carries a
#: single robot name, not SADF's identity block, so the id has to be synthesized from it.
_IMPORTED_PREFIX: Final = "imported."


def _identity(name: str) -> Identity:
    """Mint an :class:`Identity` from a description's robot name — **idempotently**.

    An exported description names the robot by its SADF *id*, so re-importing one would otherwise
    mint ``imported.imported.rover`` and every round trip would grow another prefix. A name that is
    already in the imported namespace is therefore taken as the id it plainly is, which is what
    makes ``URDF → SADF → URDF → SADF`` reach a fixed point (fleet.md §10 round-trip tests).
    """
    asset_id = name if name.startswith(_IMPORTED_PREFIX) else f"{_IMPORTED_PREFIX}{name}"
    return Identity(id=asset_id, name=name, version="0.1.0", kind="imported")


_URDF_JOINT_TYPES: dict[str, JointType] = {
    "fixed": JointType.FIXED,
    "revolute": JointType.REVOLUTE,
    "continuous": JointType.CONTINUOUS,
    "prismatic": JointType.PRISMATIC,
}


def joint_type(raw: str) -> JointType:
    """Map a URDF/SDF joint-type string to the SADF enum, or fail with a clear error
    for unsupported kinds (floating/planar/ball/…)."""
    try:
        return _URDF_JOINT_TYPES[raw]
    except KeyError:
        raise ImportError_(
            f"unsupported joint type {raw!r} (supported: {sorted(_URDF_JOINT_TYPES)})"
        ) from None
