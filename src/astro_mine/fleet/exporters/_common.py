# SPDX-License-Identifier: Apache-2.0
"""Shared exporter core: SADF → a link tree, the loss contract, and mesh materialization.

URDF, SDF, and USD differ only in *serialization*. What they share is the hard part: SADF's
model is a **frame tree** with bodies hung off frames and joints drawn between *bodies*, while
every robot-description format wants a **link tree** — one node per rigid part, each with at
most one parent joint, an inertial, and its geometry. :func:`realize` does that mapping once
(RM-P0-FLEET-02); the three writers only spell the result.

**The mapping** (fleet.md §11 "bidirectional URDF/SDF ↔ SADF converters", SADF authoritative):

- one **link per SADF frame** — frames are the kinematic skeleton, and are what geometry,
  sensors, and payload slots name. A frame carrying no body becomes a massless link (legal in
  URDF/SDF, an inertia-free ``Xform`` in USD), which is exactly what the importers read back
  as a frame with no body;
- a link's **inertial** is the composition of every body on that frame (mass sum, mass-weighted
  centre of mass, parallel-axis inertia sum);
- a **joint per SADF joint**, redrawn between the *frames* its endpoint bodies sit on, with the
  origin recomputed as the child frame's pose **relative to the parent frame** — so world poses
  survive even when the joint topology disagrees with the frame hierarchy;
- every frame not claimed by a joint is attached to its frame parent by a synthesized **fixed**
  joint. That rigid attachment is real — it is what the frame's parenthood *means* — and it is
  the inverse of the importers' rule that drops a joint to a massless link because the frame
  already carries the attachment.

Whatever cannot survive that mapping is **reported, never dropped in silence** (fleet.md §11's
"lossy-but-documented" open question): each writer returns :class:`LossFinding`s in the same
shape ``fleet lint`` uses (rule id / path / message), and :data:`LOSS_CONTRACT` names every
lossy edge per direction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import trimesh
from numpy.typing import NDArray

from astro_mine.core.sadf import SadfDocument
from astro_mine.core.sadf.enums import FidelityTier, GeometryFormat, GeometryRole
from astro_mine.core.sadf.model import (
    Asset,
    Body,
    Frame,
    GeometryRef,
    Inertia,
    Joint,
    JointLimits,
    Quat,
    Transform,
    Vec3,
)
from astro_mine.fleet import geometry

__all__ = [
    "ExportError",
    "ExportResult",
    "LossFinding",
    "Realization",
    "RealizedJoint",
    "RealizedLink",
    "fmt_float",
    "realize",
]

_IDENTITY4: Final = np.eye(4)

#: Suffix of a synthesized fixed joint, which realizes a frame's rigid parenthood in a format
#: that can only express attachment as a joint. Deterministic, so a re-export is byte-stable.
FIXED_JOINT_SUFFIX: Final = "__fixed"


class ExportError(Exception):
    """A SADF asset could not be mapped onto a robot-description format."""


@dataclass(frozen=True)
class LossFinding:
    """One thing a converter could not carry across, in ``fleet lint``'s diagnostic shape.

    ``rule`` is a stable dotted id (e.g. ``urdf.actuators_dropped``) a consumer can filter on;
    ``path`` locates the source element (``asset.actuators[0]``); ``message`` says what was lost
    and what — if anything — survived instead. Unlike a lint finding, a loss finding is **not an
    error**: a lossy export is expected and documented (fleet.md §11), so ``fleet export`` still
    succeeds and simply reports.
    """

    rule: str
    path: str
    message: str


@dataclass(frozen=True)
class ExportResult:
    """What one export produced: the document, the meshes beside it, and what it cost."""

    path: Path
    mesh_paths: tuple[Path, ...] = ()
    losses: tuple[LossFinding, ...] = ()


@dataclass
class RealizedLink:
    """One node of the link tree: a SADF frame, with the mass and geometry that ride on it."""

    name: str  # the SADF frame name — links are named by frame, not by body
    bodies: list[Body]
    geometry: list[GeometryRef]
    world: NDArray[np.float64]  # pose in the root frame, for formats that want absolute poses

    @property
    def has_inertia(self) -> bool:
        return bool(self.bodies)


@dataclass
class RealizedJoint:
    """One edge of the link tree, between *frames*, with the child's pose in the parent."""

    name: str
    type: str  # JointType value, or "fixed" for a synthesized attachment
    parent: str  # parent frame name
    child: str  # child frame name
    origin: NDArray[np.float64]  # child frame's 4x4 pose relative to the parent frame
    axis: Vec3 | None = None
    limits: JointLimits | None = None
    synthesized: bool = False  # not a SADF joint: a frame's rigid parenthood made explicit
    #: The SADF transform ``origin`` was taken from **verbatim**, when it was — i.e. when the joint
    #: agrees with the frame hierarchy, which is the case for essentially every asset. A writer that
    #: can store a quaternion (USD can; URDF and SDF cannot, they want roll-pitch-yaw) authors this
    #: instead of factoring one back out of ``origin``: quat → matrix → quat is not the identity,
    #: and that ~1e-16 is what separates a round trip that is a fixed point from one that drifts.
    transform: Transform | None = None


@dataclass
class Realization:
    """A SADF asset as a single-rooted link tree, plus the losses the mapping incurred."""

    name: str
    root: str
    links: list[RealizedLink]
    joints: list[RealizedJoint]
    losses: list[LossFinding] = field(default_factory=list)

    def link(self, name: str) -> RealizedLink:
        return next(link for link in self.links if link.name == name)


# --- SADF -> link tree -----------------------------------------------------------


def realize(doc: SadfDocument, *, fidelity: FidelityTier | None = None) -> Realization:
    """Map a SADF document onto a single-rooted link tree (see the module docstring).

    ``fidelity`` selects the visual LOD tier each link's geometry is taken at
    (:func:`~astro_mine.fleet.geometry.lod_for_tier`); ``None`` keeps every tier, which only
    the USD writer — the one format that can carry a LOD ladder — uses.

    Raises :class:`ExportError` if the asset cannot be a tree: a frame cycle, a frame claimed
    as the child of two joints, or a joint that would re-parent the root.
    """
    asset = doc.asset
    losses: list[LossFinding] = []
    frames = {frame.name: frame for frame in asset.frames}
    if asset.root_frame not in frames:
        # An asset may name a root frame it never declares (Core allows an empty `frames`).
        frames = {asset.root_frame: Frame(name=asset.root_frame), **frames}

    world = _world_poses(frames, asset.root_frame)
    links = _links(asset, frames, world, fidelity, losses)
    joints = _joints(asset, frames, world, losses)
    _assert_tree(asset.root_frame, frames, joints)

    losses += _dropped_blocks(asset)
    return Realization(
        name=asset.identity.id, root=asset.root_frame, links=links, joints=joints, losses=losses
    )


def _world_poses(frames: dict[str, Frame], root: str) -> dict[str, NDArray[np.float64]]:
    """Every frame's 4x4 pose in the root frame, composing the declared parent chain."""
    poses: dict[str, NDArray[np.float64]] = {}

    def resolve(name: str, seen: tuple[str, ...]) -> NDArray[np.float64]:
        if name in poses:
            return poses[name]
        if name in seen:
            raise ExportError(f"frame cycle: {' -> '.join([*seen, name])}")
        frame = frames[name]
        local = _transform_matrix(frame.transform)
        parent = frame.parent
        if parent is None or name == root:
            pose = local
        elif parent not in frames:  # pragma: no cover - Core's loader closes frame refs
            raise ExportError(f"frame {name!r} names an undeclared parent {parent!r}")
        else:
            pose = resolve(parent, (*seen, name)) @ local
        poses[name] = pose
        return pose

    for name in frames:
        resolve(name, ())
    return poses


def _links(
    asset: Asset,
    frames: dict[str, Frame],
    world: dict[str, NDArray[np.float64]],
    fidelity: FidelityTier | None,
    losses: list[LossFinding],
) -> list[RealizedLink]:
    """One link per frame, carrying the bodies and geometry that name that frame."""
    lod = None if fidelity is None else geometry.lod_for_tier(fidelity)
    refs = geometry.select_geometry(asset.geometry, lod=lod)

    links = []
    for name in frames:
        bodies = [body for body in asset.bodies if body.frame == name]
        if len(bodies) > 1:
            losses.append(
                LossFinding(
                    "link.bodies_merged",
                    f"asset.frames[{name!r}]",
                    f"frame {name!r} carries {len(bodies)} bodies "
                    f"({', '.join(b.name for b in bodies)}); a link has one inertial, so they are "
                    "composed into one (mass sum, parallel-axis inertia) and their names are lost",
                )
            )
        links.append(
            RealizedLink(
                name=name,
                bodies=bodies,
                geometry=[ref for ref in refs if ref.frame == name],
                world=world[name],
            )
        )
    return links


def _joints(
    asset: Asset,
    frames: dict[str, Frame],
    world: dict[str, NDArray[np.float64]],
    losses: list[LossFinding],
) -> list[RealizedJoint]:
    """SADF joints redrawn between frames, plus a fixed joint for every unclaimed frame."""
    frame_of = {body.name: body.frame for body in asset.bodies}
    claimed: dict[str, str] = {}  # child frame -> the joint that claims it
    joints: list[RealizedJoint] = []

    def origin(parent: str, child: str) -> tuple[NDArray[np.float64], Transform | None]:
        """The child frame's pose in the parent frame, and the SADF transform it came from.

        When the joint's parent *is* the child's declared frame parent — which it is for every
        asset whose joint topology follows its frame hierarchy, i.e. essentially all of them —
        the answer is the child's declared transform, verbatim. Taking it verbatim rather than
        composing to the root and inverting back keeps the number **bit-exact**: ``0.6 - 0.2``
        is not ``0.4`` in binary floating point, and that drift would otherwise land in every
        export of a two-deep frame chain. Only a joint that genuinely re-parents a frame pays
        the compose/decompose (and the tolerance that comes with it), and gets no verbatim
        transform to hand a writer.
        """
        if frames[child].parent == parent:
            declared = frames[child].transform
            return _transform_matrix(declared), declared
        return _relative(world[parent], world[child]), None

    for index, joint in enumerate(asset.joints):
        parent, child = frame_of[joint.parent_body], frame_of[joint.child_body]
        if parent == child:
            raise ExportError(
                f"joint {joint.name!r} connects two bodies on the same frame {child!r}; "
                "a link cannot be jointed to itself"
            )
        if child in claimed:
            raise ExportError(
                f"frame {child!r} is the child of two joints ({claimed[child]!r} and "
                f"{joint.name!r}); a link tree gives each link one parent joint"
            )
        if child == asset.root_frame:
            raise ExportError(
                f"joint {joint.name!r} makes the root frame {child!r} a joint child; "
                "the root of a link tree has no parent"
            )
        claimed[child] = joint.name
        pose, declared = origin(parent, child)
        joints.append(
            RealizedJoint(
                name=joint.name,
                type=joint.type.value,
                parent=parent,
                child=child,
                origin=pose,
                axis=joint.axis,
                limits=joint.limits,
                transform=declared,
            )
        )
        _joint_losses(joint, f"asset.joints[{index}]", losses)

    for name, frame in frames.items():
        if name == asset.root_frame or name in claimed:
            continue
        parent = frame.parent if frame.parent is not None else asset.root_frame
        pose, declared = origin(parent, name)
        joints.append(
            RealizedJoint(
                name=f"{name}{FIXED_JOINT_SUFFIX}",
                type="fixed",
                parent=parent,
                child=name,
                origin=pose,
                synthesized=True,
                transform=declared,
            )
        )
    return joints


def _joint_losses(joint: Joint, path: str, losses: list[LossFinding]) -> None:
    """Flag the joint fields a link-tree format cannot state as SADF does."""
    limits = joint.limits
    if limits is not None and limits.effort_nm is not None and joint.type.value == "prismatic":
        losses.append(
            LossFinding(
                "joint.effort_unit",
                f"{path}.limits.effort_nm",
                f"joint {joint.name!r} is prismatic, so its SADF `effort_nm` is a **force** in "
                "URDF/SDF (newtons, not newton-metres); the number is carried across unchanged "
                "and the unit is reinterpreted by the format, exactly as on import",
            )
        )


def _assert_tree(root: str, frames: dict[str, Frame], joints: list[RealizedJoint]) -> None:
    """Every frame must be reachable from the root.

    :func:`_joints` already gives each frame exactly one parent edge, so the reachable part of
    the graph is a tree by construction. What that does *not* rule out is a **detached cycle** —
    frame ``a`` parented to ``b`` while a joint parents ``b`` to ``a`` — which is a closed loop
    hanging off nothing. It shows up here as an unreachable frame.
    """
    children: dict[str, list[str]] = {}
    for joint in joints:
        children.setdefault(joint.parent, []).append(joint.child)

    seen = {root}
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), ()):
            seen.add(child)
            stack.append(child)

    orphans = sorted(set(frames) - seen)
    if orphans:
        raise ExportError(
            f"frames unreachable from the root {root!r}: {orphans} — the frame hierarchy and the "
            "joint topology form a closed loop, which no link tree can express"
        )


def _dropped_blocks(asset: Asset) -> list[LossFinding]:
    """The SADF blocks no robot-description format has a home for.

    URDF, SDF, and USD describe *kinematics, mass, and geometry*. SADF describes a spacecraft:
    power and thermal budgets, sensor observation models, comms links, ISRU throughput, mobility
    envelopes, capability tags. None of that survives an export — which is the whole reason SADF
    is authoritative and the export is an interop artifact, not a source of truth (fleet.md §11).
    """
    blocks: list[tuple[str, Sequence[object], str]] = [
        ("actuators", asset.actuators, "actuator torque/force/power draw"),
        ("sensors", asset.sensors, "sensor kinds, poses, and observation models"),
        ("comms", asset.comms, "comms links, antennas, and protocols"),
        ("capabilities", asset.capabilities, "capability tags (the negotiation vocabulary)"),
        ("fidelity_profiles", asset.fidelity_profiles, "fidelity profiles"),
    ]
    losses = [
        LossFinding(
            "asset.block_dropped",
            f"asset.{name}",
            f"{label} ({len(value)} declared) — no robot-description format states them; SADF "
            "stays authoritative and Sim reads them from it, not from this export",
        )
        for name, value, label in blocks
        if value
    ]
    singles = [
        ("power", asset.power, "the power budget"),
        ("thermal", asset.thermal, "the thermal budget"),
        ("mobility", asset.mobility, "the mobility/contact envelope"),
        ("propulsion", asset.propulsion, "the propulsion budget"),
        ("payload", asset.payload, "the payload slots"),
        ("return", asset.return_, "the return/Earth-interface spec"),
        ("interfaces", asset.interfaces, "the observation/action space handles"),
    ]
    losses += [
        LossFinding(
            "asset.block_dropped",
            f"asset.{name}",
            f"{label} — no robot-description format states it; SADF stays authoritative",
        )
        for name, value, label in singles
        if value is not None
    ]
    return losses


# --- geometry --------------------------------------------------------------------


def mesh_for(ref: GeometryRef, base_dir: Path) -> trimesh.Trimesh:
    """Load the mesh a geometry ref points at, in the body frame.

    Resolves the ref's (relative) ``uri`` against the SADF document's directory — the frame the
    importers wrote it in — and reads it back through
    :func:`~astro_mine.fleet.geometry.load_geometry`, which undoes glTF's y-up axis convention.
    """
    return geometry.load_geometry(base_dir / ref.uri)


def write_meshes(
    links: list[RealizedLink],
    *,
    base_dir: Path,
    assets_dir: Path,
    uri_prefix: str,
    losses: list[LossFinding],
) -> dict[int, str]:
    """Re-materialize every link's geometry as OBJ beside the export; return ref-id → uri.

    A URDF/SDF consumer reads OBJ/STL/DAE, not the glTF and USD that SADF carries, so an export
    that merely *pointed* at the SADF geometry would not open in MuJoCo, Gazebo, or rviz. Each
    ref is therefore read back into the body frame and rewritten as OBJ under ``assets_dir``,
    keeping the ref's stem so the artifact is traceable to the ref that produced it.

    A ref whose file is missing or unreadable is reported (``geometry.unresolved``) and skipped:
    a mesh-less link is still a valid, useful link, and Phase-0's reference library is entirely
    mesh-free.
    """
    uris: dict[int, str] = {}
    for link in links:
        # Only the refs a writer will actually reference: SADF carries every mesh twice (USD *and*
        # glTF), and materializing both would write each OBJ twice under the same name.
        for ref in usable_refs(link):
            stem = Path(ref.uri).stem
            try:
                mesh = mesh_for(ref, base_dir)
            except geometry.GeometryError as exc:
                losses.append(
                    LossFinding(
                        "geometry.unresolved",
                        f"asset.geometry[{ref.uri!r}]",
                        f"cannot read the geometry {ref.uri!r} this asset references ({exc}); "
                        "the link is exported without it",
                    )
                )
                continue
            uri = f"{uri_prefix}{stem}.obj"
            geometry.write_obj(mesh, assets_dir / f"{stem}.obj")
            uris[id(ref)] = uri
    return uris


def usable_refs(link: RealizedLink) -> list[GeometryRef]:
    """One geometry ref per (role, LOD) — the glTF form, or USD where no glTF was written.

    SADF carries every mesh twice (USD *and* glTF, fleet.md §11). Emitting both into a URDF
    would double every visual, so one is chosen: glTF, because it is the form
    :func:`~astro_mine.fleet.geometry.load_geometry` reads without a USD dependency.
    """
    by_slot: dict[tuple[GeometryRole, int], GeometryRef] = {}
    for ref in link.geometry:
        slot = (ref.role, ref.lod)
        if slot not in by_slot or ref.format is GeometryFormat.GLTF:
            by_slot[slot] = ref
    order = {id(ref): i for i, ref in enumerate(link.geometry)}
    return sorted(by_slot.values(), key=lambda ref: order[id(ref)])


def proxy_meshes(link: RealizedLink) -> list[trimesh.Trimesh]:
    """Inertia-equivalent proxy boxes for a link that declares mass but no mesh.

    Every Phase-0 reference asset is mesh-free, so a preview of one would otherwise be an empty
    scene. The proxy is *derived*, never asserted as the asset's geometry — callers flag it.
    """
    meshes = []
    for body in link.bodies:
        box = geometry.mass_proxy_mesh(body.mass_kg, body.inertia_kg_m2)
        offset = np.eye(4)
        offset[:3, 3] = (body.center_of_mass_m.x, body.center_of_mass_m.y, body.center_of_mass_m.z)
        box.apply_transform(offset)
        meshes.append(box)
    return meshes


# --- mass composition ------------------------------------------------------------


def composite_inertial(bodies: list[Body]) -> tuple[float, Vec3, Inertia]:
    """Compose N bodies on one frame into a single (mass, centre of mass, inertia).

    Mass sums; the centre of mass is the mass-weighted mean; the inertia is each body's tensor
    carried to the composite centre of mass by the **parallel-axis theorem**
    (``I += m(|d|^2 E - d dT)``) and summed. Physically exact — what is lost is only the *names*
    of the bodies that went in (flagged as ``link.bodies_merged``).

    The single-body case — every frame of every Phase-0 reference asset — returns the body's own
    values untouched. Not an optimization: ``m·c/m`` is *not* ``c`` in binary floating point, and
    a centre of mass that comes back as ``-0.10000000000000002`` would put float noise into an
    otherwise byte-exact export for no reason at all.
    """
    if len(bodies) == 1:
        body = bodies[0]
        return body.mass_kg, body.center_of_mass_m, body.inertia_kg_m2

    mass = sum(body.mass_kg for body in bodies)
    centres = np.array([_vec(body.center_of_mass_m) for body in bodies])
    weights = np.array([body.mass_kg for body in bodies])
    com = (weights[:, None] * centres).sum(axis=0) / mass

    tensor = np.zeros((3, 3))
    for body, centre in zip(bodies, centres, strict=True):
        offset = centre - com
        tensor += _tensor(body.inertia_kg_m2) + body.mass_kg * (
            float(offset @ offset) * np.eye(3) - np.outer(offset, offset)
        )
    return mass, Vec3(x=float(com[0]), y=float(com[1]), z=float(com[2])), _inertia(tensor)


def _tensor(inertia: Inertia) -> NDArray[np.float64]:
    return np.array(
        [
            [inertia.ixx, inertia.ixy, inertia.ixz],
            [inertia.ixy, inertia.iyy, inertia.iyz],
            [inertia.ixz, inertia.iyz, inertia.izz],
        ],
        dtype=float,
    )


def _inertia(tensor: NDArray[np.float64]) -> Inertia:
    return Inertia(
        ixx=float(tensor[0, 0]),
        iyy=float(tensor[1, 1]),
        izz=float(tensor[2, 2]),
        ixy=float(tensor[0, 1]),
        ixz=float(tensor[0, 2]),
        iyz=float(tensor[1, 2]),
    )


# --- transforms ------------------------------------------------------------------


def _vec(v: Vec3) -> NDArray[np.float64]:
    return np.array([v.x, v.y, v.z], dtype=float)


def _transform_matrix(transform: Transform | None) -> NDArray[np.float64]:
    """A SADF transform as a 4x4 homogeneous matrix (``None`` → identity)."""
    if transform is None:
        return _IDENTITY4.copy()
    matrix = np.eye(4)
    matrix[:3, :3] = quat_to_matrix(transform.rotation_quat_xyzw)
    matrix[:3, 3] = _vec(transform.translation_m)
    return matrix


def _relative(parent: NDArray[np.float64], child: NDArray[np.float64]) -> NDArray[np.float64]:
    """``child``'s pose expressed in ``parent``'s frame — an exact rigid-transform inverse."""
    rotation, translation = parent[:3, :3], parent[:3, 3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse @ child


def quat_to_matrix(q: Quat) -> NDArray[np.float64]:
    """A scalar-last unit quaternion as a 3x3 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:  # pragma: no cover - the schema requires a unit quaternion
        raise ExportError("rotation quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_rpy(matrix: NDArray[np.float64]) -> tuple[float, float, float]:
    """A 3x3 rotation as fixed-axis roll-pitch-yaw (``R = Rz·Ry·Rx``), the URDF/SDF convention."""
    pitch = math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0]))
    if math.isclose(abs(matrix[2, 0]), 1.0, abs_tol=1e-12):  # gimbal lock: fold roll into yaw
        return 0.0, pitch, math.atan2(-matrix[0, 1], matrix[1, 1])
    return (
        math.atan2(matrix[2, 1], matrix[2, 2]),
        pitch,
        math.atan2(matrix[1, 0], matrix[0, 0]),
    )


def fmt_float(value: float) -> str:
    """Format a float for XML: shortest round-tripping form, no locale, no ``-0``.

    ``repr``-grade precision keeps a re-import bit-exact; normalizing ``-0.0`` keeps the bytes
    stable regardless of which side of zero a rotation happened to land on (conventions.md §11).
    """
    return repr(float(value) + 0.0 if value == 0.0 else float(value))


def fmt_vec(values: NDArray[np.float64] | tuple[float, ...]) -> str:
    return " ".join(fmt_float(float(v)) for v in values)
