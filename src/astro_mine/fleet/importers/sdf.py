# SPDX-License-Identifier: Apache-2.0
"""SDF -> SADF parsing (RM-P0-FLEET-02), via the stdlib XML parser.

Maps one ``<model>`` from an SDF/Gazebo file onto the shared
:class:`~astro_mine.fleet.importers._common.IRModel`. SDF link ``<pose>``s are taken
relative to the model frame (the pre-1.7 default), so each link becomes a frame under a
synthetic model root; ``relative_to`` frame graphs and multi-model worlds are deferred
(P1). Geometry, inertia-frame rotation, and the SADF build are shared with the URDF path.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray

from astro_mine.core.sadf.enums import GeometryRole, JointType
from astro_mine.core.sadf.model import JointLimits, Range
from astro_mine.fleet import geometry
from astro_mine.fleet.importers import _common
from astro_mine.fleet.importers._common import (
    ImportError_,
    IRGeometry,
    IRJoint,
    IRLink,
    IRModel,
)


def parse_sdf(path: str | Path) -> IRModel:
    """Parse the first ``<model>`` of an SDF file into an :class:`IRModel`."""
    p = Path(path)
    if not p.is_file():
        raise ImportError_(f"SDF file not found: {p}")
    try:
        model = ET.parse(p).getroot().find(".//model")
    except ET.ParseError as exc:
        raise ImportError_(f"cannot parse SDF {p}: {exc}") from None
    if model is None:
        raise ImportError_(f"SDF {p} contains no <model>")
    name = model.get("name") or "model"
    base_dir = p.parent

    links = [_link(link, name, base_dir) for link in model.findall("link")]
    joints = [_joint(j) for j in model.findall("joint")]
    return IRModel(name=name, root_frame=name, links=links, joints=joints)


def _floats(text: str | None, n: int) -> list[float]:
    values = [float(tok) for tok in (text or "").split()]
    if len(values) != n:
        raise ImportError_(f"expected {n} numbers, got {text!r}")
    return values


def _pose_matrix(elem: ET.Element | None) -> NDArray[np.float64]:
    """A ``<pose>x y z roll pitch yaw</pose>`` (or absent → identity) to a 4x4 matrix."""
    mat = np.eye(4)
    pose = None if elem is None else elem.find("pose")
    if pose is None:
        return mat
    x, y, z, roll, pitch, yaw = _floats(pose.text, 6)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    mat[:3, :3] = rz @ ry @ rx
    mat[:3, 3] = (x, y, z)
    return mat


def _link(link: ET.Element, root: str, base_dir: Path) -> IRLink:
    name = link.get("name") or "link"
    transform = _common.matrix_to_transform(_pose_matrix(link))
    # A link named like the model *is* the model frame, not a child of it. Without this it would
    # be parented to itself — which is what an SDF written by Fleet's own exporter looks like,
    # since that exporter emits a link for every SADF frame including the root.
    parent = None if name == root else root

    mass = com = inertia = None
    inertial = link.find("inertial")
    if inertial is not None:
        origin = _pose_matrix(inertial)
        mass = float(_floats(inertial.findtext("mass"), 1)[0])
        com = _common.vec3(origin[:3, 3])
        inertia = _common.inertia_in_body_frame(
            _inertia_tensor(inertial.find("inertia")), origin[:3, :3]
        )

    geometries: list[IRGeometry] = []
    visuals = [_mesh(v, base_dir) for v in link.findall("visual")]
    collisions = [_mesh(c, base_dir) for c in link.findall("collision")]
    geometries += [IRGeometry(GeometryRole.VISUAL, m) for m in visuals]
    geometries += [IRGeometry(GeometryRole.COLLISION, m) for m in (collisions or visuals)]
    return IRLink(name, parent, transform, mass, com, inertia, geometries)


def _inertia_tensor(elem: ET.Element | None) -> NDArray[np.float64]:
    def g(tag: str) -> float:
        return 0.0 if elem is None else float((elem.findtext(tag) or "0").strip())

    ixx, iyy, izz = g("ixx"), g("iyy"), g("izz")
    ixy, ixz, iyz = g("ixy"), g("ixz"), g("iyz")
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)


def _mesh(elem: ET.Element, base_dir: Path) -> trimesh.Trimesh:
    geom = elem.find("geometry")
    shape = next(iter(geom), None) if geom is not None else None
    if shape is None:
        raise ImportError_("visual/collision has no <geometry>")
    scale: tuple[float, float, float] | None = None
    if shape.tag == "box":
        base = trimesh.creation.box(extents=np.asarray(_floats(shape.findtext("size"), 3)))
    elif shape.tag == "cylinder":
        base = trimesh.creation.cylinder(
            radius=_floats(shape.findtext("radius"), 1)[0],
            height=_floats(shape.findtext("length"), 1)[0],
        )
    elif shape.tag == "sphere":
        base = trimesh.creation.icosphere(radius=_floats(shape.findtext("radius"), 1)[0])
    elif shape.tag == "mesh":
        uri = shape.findtext("uri") or ""
        base = geometry.load_mesh(_common.resolve_mesh_path(uri, base_dir))
        if shape.findtext("scale"):
            sx, sy, sz = _floats(shape.findtext("scale"), 3)
            scale = (sx, sy, sz)
    else:
        raise ImportError_(f"unsupported SDF geometry <{shape.tag}>")
    return geometry.normalize_mesh(base, scale=scale, transform=_pose_matrix(elem))


def _joint(joint: ET.Element) -> IRJoint:
    name = joint.get("name") or "joint"
    jtype = _common.joint_type(joint.get("type") or "")
    parent = (joint.findtext("parent") or "").strip()
    child = (joint.findtext("child") or "").strip()
    if not parent or not child:
        raise ImportError_(f"joint {name!r} is missing <parent> or <child>")

    axis_elem = joint.find("axis")
    axis = None
    if jtype is not JointType.FIXED and axis_elem is not None and axis_elem.findtext("xyz"):
        axis = _common.vec3(np.asarray(_floats(axis_elem.findtext("xyz"), 3)))
    return IRJoint(name, jtype, parent, child, axis, _limits(axis_elem, jtype))


def _limits(axis_elem: ET.Element | None, jtype: JointType) -> JointLimits | None:
    limit = None if axis_elem is None else axis_elem.find("limit")
    if limit is None:
        return None

    def val(tag: str) -> float | None:
        text = limit.findtext(tag)
        return None if text is None else float(text)

    lower, upper = val("lower"), val("upper")
    position = (
        Range(min=lower, max=upper)
        if jtype is not JointType.CONTINUOUS and lower is not None and upper is not None
        else None
    )
    velocity, effort = val("velocity"), val("effort")
    if position is None and velocity is None and effort is None:
        return None
    return JointLimits(position_rad=position, velocity_rad_s=velocity, effort_nm=effort)
