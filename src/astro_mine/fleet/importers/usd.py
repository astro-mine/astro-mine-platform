"""USD stage -> SADF parsing (RM-P0-FLEET-02), via ``pxr``.

The inverse of :mod:`astro_mine.fleet.exporters.usd`, and what makes the **USD → SADF → USD**
round trip fleet.md §10 asks for a real test rather than an assertion about a writer.

Reads the stage Fleet *writes*: an ``Xform`` tree (→ frames), ``UsdPhysics.MassAPI`` (→ bodies,
multiplying the diagonal inertia back out through its principal axes), and ``UsdPhysics``
revolute/prismatic/fixed joints (→ joints, inverting the joint-frame rotation that carries a
non-basis axis). A general Omniverse/Isaac stage — variants, materials, instancing, articulation
roots — is *not* a SADF asset, and is out of scope (see ``exporters.LOSS_CONTRACT``).

Geometry rides on the Xform prims as references to the mesh stages the SADF already points at;
each is read back into the body frame and re-emitted through the shared geometry pipeline, so a
USD import produces the same normalized USD + glTF + collision-hull + LOD artifacts as a URDF one.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.sadf.enums import GeometryRole, JointType
from astro_mine.core.sadf.model import JointLimits, Quat, Range, Transform
from astro_mine.fleet import geometry
from astro_mine.fleet.exporters.usd import AXIS_TOKENS
from astro_mine.fleet.importers import _common
from astro_mine.fleet.importers._common import (
    ImportError_,
    IRGeometry,
    IRJoint,
    IRLink,
    IRModel,
)

__all__ = ["parse_usd"]

#: UsdPhysics joint prim types, mapped onto the SADF enum. A revolute joint that Fleet tagged
#: ``astroMine:jointType = "continuous"`` is unbounded; see :func:`_joint_type`.
_JOINT_TYPES = {
    "PhysicsRevoluteJoint": JointType.REVOLUTE,
    "PhysicsPrismaticJoint": JointType.PRISMATIC,
    "PhysicsFixedJoint": JointType.FIXED,
}


def parse_usd(path: str | Path) -> IRModel:
    """Parse a USD stage into an :class:`IRModel`."""
    from pxr import Usd, UsdGeom

    p = Path(path)
    if not p.is_file():
        raise ImportError_(f"USD file not found: {p}")
    stage = Usd.Stage.Open(str(p))
    if stage is None:
        raise ImportError_(f"cannot open USD stage {p}")

    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        raise ImportError_(f"USD stage {p} declares no default prim to root the asset at")

    # Frame prims are the Xforms that are *not* joints and not the geometry hung under a link.
    frames = [
        prim
        for prim in Usd.PrimRange(root_prim)
        if prim.IsA(UsdGeom.Xform) and not _is_joint(prim) and not _is_geometry(prim)
    ]
    if not frames:  # pragma: no cover - the default prim is itself an Xform
        raise ImportError_(f"USD stage {p} holds no Xform frames")

    root = frames[0].GetName()
    joints = [_joint(prim) for prim in Usd.PrimRange(root_prim) if _is_joint(prim) and _named(prim)]
    links = [_link(prim, root, p.parent) for prim in frames]

    # A Fleet-written stage names joints between *frames*; SADF joints connect *bodies*, and the
    # shared builder drops any joint whose endpoint carries no body — the same rule the URDF path
    # applies to a massless link, so a synthesized fixed joint collapses back into frame parenthood.
    return IRModel(name=root_prim.GetName(), root_frame=root, links=links, joints=joints)


def _is_joint(prim: Any) -> bool:
    return prim.GetTypeName() in _JOINT_TYPES


def _is_geometry(prim: Any) -> bool:
    """Geometry Xforms carry the ``astroMine:role`` attribute the exporter stamps on them."""
    return bool(prim.GetAttribute("astroMine:role"))


def _named(prim: Any) -> bool:
    return bool(prim.GetName())


# --- links -----------------------------------------------------------------------


def _link(prim: Any, root: str, base_dir: Path) -> IRLink:
    from pxr import UsdPhysics

    name = prim.GetName()
    parent = prim.GetParent()
    parent_frame = None if name == root or not parent.IsValid() else parent.GetName()
    transform = None if parent_frame is None else _local_transform(prim)

    mass = com = inertia = None
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass, com, inertia = _mass(UsdPhysics.MassAPI(prim))

    return IRLink(name, parent_frame, transform, mass, com, inertia, _geometry(prim, base_dir))


def _local_transform(prim: Any) -> Transform | None:
    """An ``Xform``'s local pose as a SADF :class:`Transform`.

    Reads the **authored ops** — the ``translate`` and ``orient`` the exporter wrote — rather than
    the matrix USD composes from them. Both describe the same pose, but composing a quaternion into
    a matrix and factoring it back out costs ~2e-16, and that is enough to stop the round trip being
    a fixed point. Reading the quaternion that is actually on disk costs nothing and is exact.

    A stage authored with any other op (a bare ``matrix``, Euler ``rotateXYZ``, a ``scale``) is not
    one Fleet wrote; it is decomposed from the composed matrix instead — lossier, but it reads.
    """
    from pxr import UsdGeom

    translation = np.zeros(3)
    rotation: Quat | None = None
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            value = op.Get()
            translation = np.array([value[0], value[1], value[2]], dtype=float)
        elif op_type == UsdGeom.XformOp.TypeOrient:
            quat = op.Get()
            imaginary = quat.GetImaginary()
            rotation = Quat(
                x=float(imaginary[0]),
                y=float(imaginary[1]),
                z=float(imaginary[2]),
                w=float(quat.GetReal()),
            )
        else:  # a foreign stage: fall back to the composed matrix
            return _common.matrix_to_transform(_local_pose(UsdGeom.Xform(prim)))

    if rotation is None and not translation.any():
        return None  # the identity pose, which SADF spells as no transform at all
    return Transform(
        translation_m=_common.vec3(translation),
        rotation_quat_xyzw=rotation or Quat(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _local_pose(xform: Any) -> NDArray[np.float64]:
    """An ``Xform``'s composed local transform as a 4x4 matrix."""
    matrix = xform.GetLocalTransformation()
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)], dtype=float).T


def _mass(api: Any) -> tuple[float, Any, Any]:
    """``UsdPhysics.MassAPI`` → SADF mass, centre of mass, and the full inertia tensor.

    USD stores the inertia *diagonalized*: principal moments plus the quaternion of the axes they
    are expressed in. Multiplying back out (``I = R·diag·Rᵀ``) recovers the tensor the exporter
    diagonalized, to machine precision — this is the step that makes the USD round trip exact.
    """
    mass = float(api.GetMassAttr().Get() or 0.0)
    centre = api.GetCenterOfMassAttr().Get()
    com = _common.vec3(np.array([centre[0], centre[1], centre[2]], dtype=float))

    diagonal = api.GetDiagonalInertiaAttr().Get()
    axes = _rotation(api.GetPrincipalAxesAttr().Get())
    tensor = axes @ np.diag([float(diagonal[i]) for i in range(3)]) @ axes.T
    return mass, com, _common.inertia_in_body_frame(tensor, np.eye(3))


def _rotation(quat: Any) -> NDArray[np.float64]:
    """A ``Gf.Quat*`` (scalar-first) as a 3x3 rotation matrix."""
    imaginary = quat.GetImaginary()
    x, y, z, w = (
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(quat.GetReal()),
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:  # pragma: no cover - USD normalizes the quaternions it stores
        raise ImportError_("USD rotation quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _geometry(prim: Any, base_dir: Path) -> list[IRGeometry]:
    """The meshes referenced by the geometry Xforms under a link.

    Only the **full-resolution** (``lod=0``) visual is read: the coarser tiers are *derived* from
    it, and the shared builder regenerates them on the way back out. Re-importing a decimated tier
    as if it were source geometry would decimate a decimation, so each round trip would erode the
    mesh a little further.
    """
    out: list[IRGeometry] = []
    for child in prim.GetChildren():
        role_attr = child.GetAttribute("astroMine:role")
        lod_attr = child.GetAttribute("astroMine:lod")
        if not role_attr or (lod_attr and int(lod_attr.Get() or 0) != 0):
            continue
        role = GeometryRole(str(role_attr.Get()))
        for uri in _references(child):
            path = base_dir / uri
            if not path.is_file():
                raise ImportError_(f"USD prim {child.GetName()!r} references a missing {uri!r}")
            out.append(IRGeometry(role, geometry.load_geometry(path)))
    return out


def _references(prim: Any) -> list[str]:
    """The asset paths a prim references, in authored order.

    ``referenceList``'s items come back as USD list-editing proxies, not lists, so each is
    materialized before being read.
    """
    return [
        str(reference.assetPath)
        for spec in prim.GetPrimStack()
        for items in (spec.referenceList.prependedItems, spec.referenceList.explicitItems)
        for reference in list(items)
        if reference.assetPath
    ]


# --- joints ----------------------------------------------------------------------


def _joint(prim: Any) -> IRJoint:
    from pxr import UsdPhysics

    joint = UsdPhysics.Joint(prim)
    parent = _target(joint.GetBody0Rel())
    child = _target(joint.GetBody1Rel())
    if parent is None or child is None:
        raise ImportError_(f"USD joint {prim.GetName()!r} is missing body0 or body1")

    jtype = _joint_type(prim)
    axis = None
    if jtype is not JointType.FIXED:
        # Undo the joint-frame rotation the exporter used to carry a non-basis axis: the SADF axis
        # is the token's basis vector expressed back in the child body's frame.
        align = _rotation(prim.GetAttribute("physics:localRot1").Get())
        token = str(prim.GetAttribute("physics:axis").Get() or "Z")
        axis = _common.vec3(align @ AXIS_TOKENS[token])

    return IRJoint(prim.GetName(), jtype, parent, child, axis, _limits(prim, jtype))


def _joint_type(prim: Any) -> JointType:
    """The SADF joint type, honouring the ``continuous`` marker the exporter stamps.

    USD has no unbounded-revolute *type*: a continuous joint is a revolute one with no limits, and
    an unbounded revolute is spelled the same way. The exporter therefore records the distinction
    explicitly rather than leave the reader to guess, since guessing would silently turn every
    limit-less revolute into a continuous joint.
    """
    marker = prim.GetAttribute("astroMine:jointType")
    if marker and str(marker.Get()) == "continuous":
        return JointType.CONTINUOUS
    return _JOINT_TYPES[prim.GetTypeName()]


def _target(relationship: Any) -> str | None:
    targets = relationship.GetTargets()
    return str(targets[0]).rsplit("/", 1)[-1] if targets else None


def _limits(prim: Any, jtype: JointType) -> JointLimits | None:
    """The joint's position bounds. Effort/velocity are not on a UsdPhysics joint (see the
    ``usd.drive_limits_dropped`` entry in ``exporters.LOSS_CONTRACT``)."""
    if jtype in {JointType.FIXED, JointType.CONTINUOUS}:
        return None
    lower = prim.GetAttribute("physics:lowerLimit").Get()
    upper = prim.GetAttribute("physics:upperLimit").Get()
    if lower is None or upper is None:
        return None
    scale = math.pi / 180.0 if jtype is JointType.REVOLUTE else 1.0  # USD angles are degrees
    return JointLimits(position_rad=Range(min=float(lower) * scale, max=float(upper) * scale))
