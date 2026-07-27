"""SADF -> USD stage (RM-P0-FLEET-01/02), the Sim/Studio interop form.

USD is the one export target that can hold nearly all of the kinematic model, so it is the
highest-fidelity of the three (conventions.md §3 "USD preferred, glTF for web"; fleet.md §11):

- the **frame tree** is an ``Xform`` hierarchy — USD nests natively, so nothing is flattened;
- **mass properties** ride on ``UsdPhysics.MassAPI``. USD wants a *diagonal* inertia plus the
  principal axes, so the SADF tensor is diagonalized (``I = R·diag·Rᵀ``); the reader multiplies
  it back out, and the two agree to machine precision;
- **joints** are ``UsdPhysics`` revolute/prismatic/fixed prims. USD restricts a joint axis to the
  tokens ``X``/``Y``/``Z``, so an arbitrary SADF axis is carried in the joint frame's rotation
  (``localRot0``/``localRot1``) instead — the standard encoding, and exactly invertible;
- the **whole LOD ladder** survives: every tier becomes its own ``Xform`` under the link, tagged
  with a ``lod`` attribute, because USD (unlike URDF/SDF) can carry variant detail levels.

The stage is ASCII (``.usda``) so it is diffable and byte-reproducible (conventions.md §11), and
references the geometry the SADF document already points at rather than copying it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.sadf import SadfDocument
from astro_mine.core.sadf.enums import GeometryFormat, GeometryRole
from astro_mine.core.sadf.model import Vec3
from astro_mine.fleet.exporters import _common
from astro_mine.fleet.exporters._common import (
    ExportError,
    ExportResult,
    LossFinding,
    Realization,
    RealizedJoint,
    RealizedLink,
)

__all__ = ["export_usd"]

#: The basis vector each ``UsdPhysics`` axis token names.
AXIS_TOKENS: dict[str, NDArray[np.float64]] = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}

#: The token an arbitrary axis is rotated onto when it is not already a basis vector.
_DEFAULT_TOKEN = "Z"


def export_usd(
    doc: SadfDocument,
    out: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> ExportResult:
    """Write ``doc`` as a USD stage at ``out``; return the artifact and the losses.

    Unlike the URDF/SDF writers this copies no meshes: SADF already carries USD geometry, so the
    stage *references* each ref's ``.usda`` at its relative uri. ``base_dir`` (default: ``out``'s
    parent) is what those uris resolve against; the references stay relative, so the stage is
    portable.
    """
    from pxr import Usd, UsdGeom

    out_path = Path(out)
    source = Path(base_dir) if base_dir is not None else out_path.parent
    model = _common.realize(doc)  # fidelity=None: USD keeps every LOD tier
    losses = list(model.losses)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # Usd.Stage.CreateNew refuses to clobber
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)  # the SADF body frame is z-up
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # SI, always (conventions.md §5)

    root = UsdGeom.Xform.Define(stage, f"/{_token(model.root)}")
    stage.SetDefaultPrim(root.GetPrim())
    paths = _xforms(stage, model, losses, source)
    _joints(stage, model, paths, losses)

    if any(link.has_inertia for link in model.links):
        losses.append(_PRECISION_LOSS)
    stage.GetRootLayer().Save()
    return ExportResult(path=out_path, losses=tuple(losses))


#: UsdPhysics declares ``physics:mass`` as ``float`` and ``physics:centerOfMass`` /
#: ``physics:diagonalInertia`` as ``float3`` — **single** precision. SADF carries doubles. The
#: schema is not ours to widen (a double-precision custom attribute is exactly the private
#: side-channel conventions.md §1 forbids, and Isaac would not read it), so the narrowing is
#: reported instead: a USD round trip preserves mass properties to float32, ~1e-7 relative.
_PRECISION_LOSS = LossFinding(
    "usd.float32_precision",
    "asset.bodies",
    "UsdPhysics stores mass, centre of mass, and inertia as 32-bit floats; SADF carries 64-bit "
    "doubles. Mass properties therefore survive a USD round trip to ~1e-7 relative, not exactly. "
    "Frame poses are unaffected (they are authored double-precision)",
)


def _token(name: str) -> str:
    """A USD-legal prim name: identifiers only, so a hyphenated SADF frame still lands."""
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    return cleaned if cleaned[:1].isalpha() or cleaned[:1] == "_" else f"_{cleaned}"


def _xforms(
    stage: Any, model: Realization, losses: list[LossFinding], base_dir: Path
) -> dict[str, str]:
    """Define an ``Xform`` per frame, nested by the joint tree; return frame → prim path."""
    from pxr import UsdGeom

    parent_of = {joint.child: joint.parent for joint in model.joints}
    joint_of = {joint.child: joint for joint in model.joints}

    paths: dict[str, str] = {model.root: f"/{_token(model.root)}"}

    def path_of(frame: str) -> str:
        if frame not in paths:
            paths[frame] = f"{path_of(parent_of[frame])}/{_token(frame)}"
        return paths[frame]

    for link in model.links:
        prim_path = path_of(link.name)
        xform = UsdGeom.Xform.Define(stage, prim_path)
        if link.name != model.root:
            _set_pose(xform, joint_of[link.name])
        if link.has_inertia:
            _mass(xform.GetPrim(), link, losses)
        _geometry(stage, prim_path, link, base_dir, losses)
    return paths


def _set_pose(xform: Any, joint: RealizedJoint) -> None:
    """Author a translate + orient op — never a bare matrix, which USD tools cannot decompose.

    **Double precision, explicitly.** ``AddOrientOp()`` defaults to ``PrecisionFloat``, which would
    quietly narrow every frame rotation to float32 — and a frame pose, unlike a UsdPhysics mass
    attribute, is *not* pinned to float32 by any schema. Narrowing it would be a loss we chose
    rather than one the format forced on us, and the whole point of the loss contract is that we do
    not do that.

    **The authored quaternion, not one factored back out of a matrix.** USD stores an orientation as
    a quaternion, and so does SADF — so when the realization carried the SADF transform through
    verbatim (:attr:`RealizedJoint.transform`), it is written straight through. Round-tripping it
    via the 4x4 (``quat → matrix → quat``) is *not* the identity: it costs ~2e-16 per lap, which
    is exactly enough to stop the USD round trip being a fixed point.
    """
    from pxr import Gf, UsdGeom

    declared = joint.transform
    if declared is not None:
        translation = declared.translation_m
        quat = declared.rotation_quat_xyzw
        position = Gf.Vec3d(translation.x, translation.y, translation.z)
        rotation = Gf.Quatd(quat.w, Gf.Vec3d(quat.x, quat.y, quat.z))
    else:  # a joint that re-parents a frame: the pose only exists as the composed matrix
        matrix = joint.origin
        position = Gf.Vec3d(*(float(v) for v in matrix[:3, 3]))
        rotation = _quat(matrix[:3, :3], double=True)

    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(position)
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation)


def _quat(rotation: NDArray[np.float64], *, double: bool = False) -> Any:
    """A 3x3 rotation as a USD quaternion (scalar-first).

    ``double`` picks ``Gf.Quatd`` over ``Gf.Quatf``: frame poses are authored double-precision,
    while ``UsdPhysics``' ``principalAxes`` and joint ``localRot`` attributes are ``quatf`` **by
    schema** and cannot be widened without inventing a private attribute no other tool would read.
    """
    from pxr import Gf

    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
        )
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2.0
        parts = [0.0, 0.0, 0.0]
        parts[i] = 0.25 * s
        parts[j] = (rotation[j, i] + rotation[i, j]) / s
        parts[k] = (rotation[k, i] + rotation[i, k]) / s
        w = (rotation[k, j] - rotation[j, k]) / s
        x, y, z = parts
    if double:
        return Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _mass(prim: Any, link: RealizedLink, losses: list[LossFinding]) -> None:
    """Apply ``UsdPhysics`` rigid-body + mass APIs, diagonalizing the SADF inertia tensor."""
    from pxr import Gf, UsdPhysics

    mass, com, inertia = _common.composite_inertial(link.bodies)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(float(com.x), float(com.y), float(com.z)))

    tensor = np.array(
        [
            [inertia.ixx, inertia.ixy, inertia.ixz],
            [inertia.ixy, inertia.iyy, inertia.iyz],
            [inertia.ixz, inertia.iyz, inertia.izz],
        ],
        dtype=float,
    )
    moments, axes = np.linalg.eigh(tensor)
    if np.linalg.det(axes) < 0.0:
        axes[:, 0] *= -1.0  # a reflection is not an orientation
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(m) for m in moments)))
    api.CreatePrincipalAxesAttr(_quat(axes))

    off_diagonal = abs(inertia.ixy) + abs(inertia.ixz) + abs(inertia.iyz)
    if off_diagonal > 0.0:
        losses.append(
            LossFinding(
                "usd.inertia_diagonalized",
                f"asset.frames[{link.name!r}].inertia",
                f"link {link.name!r} has products of inertia; UsdPhysics stores a diagonal "
                "tensor plus principal axes, so it is diagonalized. The physical tensor is "
                "preserved (R·diag·Rᵀ) but the six SADF components are re-expressed",
            )
        )


def _geometry(
    stage: Any, link_path: str, link: RealizedLink, base_dir: Path, losses: list[LossFinding]
) -> None:
    """Reference each USD geometry ref under the link, one ``Xform`` per role/LOD tier."""
    from pxr import Sdf, UsdGeom, UsdPhysics

    for index, ref in enumerate(link.geometry):
        if ref.format is not GeometryFormat.USD:
            continue  # the glTF twin of the same mesh; USD references USD
        if not (base_dir / ref.uri).is_file():
            losses.append(
                LossFinding(
                    "geometry.unresolved",
                    f"asset.geometry[{ref.uri!r}]",
                    f"cannot resolve the geometry {ref.uri!r} this asset references; the link is "
                    "exported without it",
                )
            )
            continue
        name = f"{ref.role.value}_{index}_lod{ref.lod}"
        prim = UsdGeom.Xform.Define(stage, f"{link_path}/{name}").GetPrim()
        prim.GetReferences().AddReference(ref.uri)
        # Keep the LOD level and the role on the prim: USD can carry the whole ladder, and a
        # consumer (Sim picking a collision proxy, Studio picking a preview tier) must be able
        # to tell the tiers apart without re-deriving them from the file name.
        prim.CreateAttribute("astroMine:lod", Sdf.ValueTypeNames.Int).Set(int(ref.lod))
        prim.CreateAttribute("astroMine:role", Sdf.ValueTypeNames.Token).Set(ref.role.value)
        if ref.role is GeometryRole.COLLISION:
            UsdPhysics.CollisionAPI.Apply(prim)


def _joints(
    stage: Any, model: Realization, paths: dict[str, str], losses: list[LossFinding]
) -> None:
    """Define a ``UsdPhysics`` joint per edge of the link tree."""
    from pxr import Gf, Sdf, UsdPhysics

    define = {
        "revolute": UsdPhysics.RevoluteJoint.Define,
        "continuous": UsdPhysics.RevoluteJoint.Define,
        "prismatic": UsdPhysics.PrismaticJoint.Define,
        "fixed": UsdPhysics.FixedJoint.Define,
    }
    for joint in model.joints:
        joint_path = f"{paths[joint.child]}/{_token(joint.name)}"
        prim = define[joint.type](stage, joint_path)
        prim.CreateBody0Rel().SetTargets([paths[joint.parent]])
        prim.CreateBody1Rel().SetTargets([paths[joint.child]])

        token, align = _axis_frame(joint.axis)
        # The joint frame sits at the child's origin, rotated so the axis token lands on the SADF
        # axis. localRot0 carries the child's own rotation as well, so the two joint frames — one
        # expressed in each body — coincide, which is what UsdPhysics requires.
        prim.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in joint.origin[:3, 3])))
        prim.CreateLocalRot0Attr(_quat(joint.origin[:3, :3] @ align))
        prim.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        prim.CreateLocalRot1Attr(_quat(align))

        if joint.type != "fixed":
            prim.CreateAxisAttr(token)
        _limits(prim, joint, losses)

        # `continuous` is a revolute joint with no bound. USD says so by leaving the limits off,
        # which is indistinguishable from an unbounded revolute — so the distinction is recorded
        # explicitly rather than guessed at on the way back.
        if joint.type == "continuous":
            attr = prim.GetPrim().CreateAttribute("astroMine:jointType", Sdf.ValueTypeNames.Token)
            attr.Set("continuous")


def _axis_frame(axis: Vec3 | None) -> tuple[str, NDArray[np.float64]]:
    """The ``(token, rotation)`` pair encoding a SADF joint axis in USD's basis-token model.

    When the axis *is* a basis vector, the token names it and the rotation is the identity — the
    common case, and it keeps the stage readable. Otherwise the axis is carried by a rotation
    taking ``_DEFAULT_TOKEN`` onto it, which the reader inverts exactly.
    """
    if axis is None:
        return _DEFAULT_TOKEN, np.eye(3)
    vector = np.array([axis.x, axis.y, axis.z], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ExportError("joint axis has zero length")
    unit = vector / norm

    for token, basis in AXIS_TOKENS.items():
        if np.allclose(unit, basis):
            return token, np.eye(3)
    return _DEFAULT_TOKEN, _align(AXIS_TOKENS[_DEFAULT_TOKEN], unit)


def _align(source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """The rotation matrix taking the unit vector ``source`` onto the unit vector ``target``."""
    axis = np.cross(source, target)
    length = float(np.linalg.norm(axis))
    if length == 0.0:  # parallel or antiparallel
        if float(np.dot(source, target)) > 0.0:
            return np.eye(3)
        # 180°: any perpendicular axis will do; pick a stable one so the stage is reproducible
        perpendicular = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(perpendicular, source))) > 0.9:
            perpendicular = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, perpendicular)
        axis /= np.linalg.norm(axis)
        angle = math.pi
    else:
        axis /= length
        angle = math.atan2(length, float(np.dot(source, target)))

    # Rodrigues' rotation formula: R = I + sin(t)*K + (1 - cos(t))*K^2, with K the cross-product
    # matrix of the (unit) rotation axis.
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    return np.asarray(rotation, dtype=float)


def _limits(prim: Any, joint: RealizedJoint, losses: list[LossFinding]) -> None:
    """Author the joint's position bounds; report the effort/velocity USD's joint cannot hold."""
    limits = joint.limits
    if limits is None or joint.type == "fixed":
        return
    if limits.position_rad is not None:
        # UsdPhysics revolute limits are *degrees*; prismatic limits are stage units (metres).
        scale = 180.0 / math.pi if joint.type in {"revolute", "continuous"} else 1.0
        prim.CreateLowerLimitAttr(float(limits.position_rad.min * scale))
        prim.CreateUpperLimitAttr(float(limits.position_rad.max * scale))
    if limits.effort_nm is not None or limits.velocity_rad_s is not None:
        losses.append(
            LossFinding(
                "usd.drive_limits_dropped",
                f"asset.joints[{joint.name!r}].limits",
                f"joint {joint.name!r} declares effort/velocity limits; a UsdPhysics joint holds "
                "only position bounds (effort and velocity belong to a PhysicsDriveAPI, which "
                "describes an actuator, not the joint). SADF stays authoritative for them",
            )
        )
