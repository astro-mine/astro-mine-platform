# SPDX-License-Identifier: Apache-2.0
"""URDF -> SADF parsing (RM-P0-FLEET-02), via ``yourdfpy``.

Maps a URDF robot onto the shared :class:`~astro_mine.fleet.importers._common.IRModel`:
links -> frames + bodies (inertia rotated into the body frame), the joint tree ->
frame parents, joints -> joints, and visual/collision geometry -> normalized meshes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
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

_IDENTITY4 = np.eye(4)


def parse_urdf(path: str | Path) -> IRModel:
    """Parse a URDF file into an :class:`IRModel`."""
    p = Path(path)
    if not p.is_file():
        raise ImportError_(f"URDF file not found: {p}")
    robot = yourdfpy.URDF.load(
        str(p), load_meshes=False, build_scene_graph=False, build_collision_scene_graph=False
    ).robot
    base_dir = p.parent

    child_links = {j.child for j in robot.joints}
    roots = [link.name for link in robot.links if link.name not in child_links]
    if len(roots) != 1:
        raise ImportError_(f"URDF must have exactly one root link, found {sorted(roots)}")
    root = roots[0]
    parent_of = {j.child: (j.parent, _mat(j.origin)) for j in robot.joints}

    links = [_link(link, root, parent_of, base_dir) for link in robot.links]
    joints = [_joint(j) for j in robot.joints]
    return IRModel(name=robot.name or "robot", root_frame=root, links=links, joints=joints)


def _mat(origin: NDArray[np.float64] | None) -> NDArray[np.float64]:
    return _IDENTITY4 if origin is None else np.asarray(origin, dtype=float)


def _link(
    link: yourdfpy.Link,
    root: str,
    parent_of: dict[str, tuple[str, NDArray[np.float64]]],
    base_dir: Path,
) -> IRLink:
    if link.name == root:
        parent_frame, transform = None, None
    else:
        parent_name, origin = parent_of[link.name]
        parent_frame, transform = parent_name, _common.matrix_to_transform(origin)

    mass = com = inertia = None
    if link.inertial is not None:
        origin = _mat(link.inertial.origin)
        mass = float(link.inertial.mass)
        com = _common.vec3(origin[:3, 3])
        inertia = _common.inertia_in_body_frame(
            np.asarray(link.inertial.inertia, dtype=float), origin[:3, :3]
        )

    geometries = _geometries(link, base_dir)
    return IRLink(link.name, parent_frame, transform, mass, com, inertia, geometries)


def _geometries(link: yourdfpy.Link, base_dir: Path) -> list[IRGeometry]:
    visuals = [_mesh(v.geometry, v.origin, base_dir) for v in link.visuals]
    collisions = [_mesh(c.geometry, c.origin, base_dir) for c in link.collisions]
    out = [IRGeometry(GeometryRole.VISUAL, m) for m in visuals]
    # Fall back to the visual meshes for collision so a hull is always generated.
    out += [IRGeometry(GeometryRole.COLLISION, m) for m in (collisions or visuals)]
    return out


def _mesh(
    geom: yourdfpy.Geometry, origin: NDArray[np.float64] | None, base_dir: Path
) -> trimesh.Trimesh:
    scale: tuple[float, float, float] | None = None
    if geom.box is not None:
        base = trimesh.creation.box(extents=np.asarray(geom.box.size, dtype=float))
    elif geom.cylinder is not None:
        base = trimesh.creation.cylinder(radius=geom.cylinder.radius, height=geom.cylinder.length)
    elif geom.sphere is not None:
        base = trimesh.creation.icosphere(radius=geom.sphere.radius)
    elif geom.mesh is not None:
        base = geometry.load_mesh(_common.resolve_mesh_path(geom.mesh.filename, base_dir))
        if geom.mesh.scale is not None:
            s = np.asarray(geom.mesh.scale, dtype=float)
            scale = (float(s[0]), float(s[1]), float(s[2]))
    else:
        raise ImportError_("geometry has no box/cylinder/sphere/mesh")
    return geometry.normalize_mesh(base, scale=scale, transform=_mat(origin))


def _joint(joint: yourdfpy.Joint) -> IRJoint:
    jtype = _common.joint_type(joint.type)
    axis = None if jtype is JointType.FIXED or joint.axis is None else _common.vec3(joint.axis)
    return IRJoint(joint.name, jtype, joint.parent, joint.child, axis, _limits(joint, jtype))


def _limits(joint: yourdfpy.Joint, jtype: JointType) -> JointLimits | None:
    lim = joint.limit
    if lim is None:
        return None
    # continuous joints are unbounded in position; revolute/prismatic carry [lower, upper].
    position = None
    if jtype is not JointType.CONTINUOUS and lim.lower is not None and lim.upper is not None:
        position = Range(min=float(lim.lower), max=float(lim.upper))
    velocity = None if lim.velocity is None else float(lim.velocity)
    effort = None if lim.effort is None else float(lim.effort)
    if position is None and velocity is None and effort is None:
        return None
    return JointLimits(position_rad=position, velocity_rad_s=velocity, effort_nm=effort)
