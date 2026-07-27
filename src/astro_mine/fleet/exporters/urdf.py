"""SADF -> URDF (RM-P0-FLEET-01/02), the ROS-ecosystem interop form.

The inverse of :mod:`astro_mine.fleet.importers.urdf`. Serializes the link tree
:func:`~astro_mine.fleet.exporters._common.realize` builds — one ``<link>`` per SADF frame, one
``<joint>`` per SADF joint plus one per rigid frame attachment — as a stdlib-XML URDF, with each
link's geometry re-materialized as body-frame OBJ beside it.

**Inertials.** URDF puts the inertia tensor in the frame its ``<inertial><origin>`` names; SADF
puts it in the *body* frame, about the centre of mass. The exporter therefore writes ``rpy="0 0
0"`` and the SADF tensor as-is — the same physical tensor, in an equal-and-simpler frame. An
import that rotated a tensor in (``I' = R·I·Rᵀ``) round-trips to this form, not to its original
XML: the *physics* is preserved exactly, the *spelling* is normalized (see ``LOSS_CONTRACT``).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from astro_mine.core.sadf import SadfDocument
from astro_mine.core.sadf.enums import FidelityTier, GeometryRole
from astro_mine.core.sadf.model import JointLimits
from astro_mine.fleet import geometry
from astro_mine.fleet.exporters import _common
from astro_mine.fleet.exporters._common import (
    ExportResult,
    LossFinding,
    Realization,
    RealizedJoint,
    RealizedLink,
    fmt_float,
    fmt_vec,
)

__all__ = ["export_urdf"]

#: URDF requires a bounded ``<limit>`` on revolute and prismatic joints. SADF does not, so an
#: unbounded one is reported rather than silently given an invented range.
_MUST_BE_BOUNDED = {"revolute", "prismatic"}


def export_urdf(
    doc: SadfDocument,
    out: str | Path,
    *,
    base_dir: str | Path | None = None,
    assets_dir: str | Path | None = None,
    fidelity: FidelityTier = FidelityTier.ARTICULATED,
) -> ExportResult:
    """Write ``doc`` as a URDF at ``out``; return the artifact, its meshes, and the losses.

    ``base_dir`` is where the document's geometry uris resolve from (default: ``out``'s parent);
    ``assets_dir`` is where the OBJ meshes are written (default: ``<out-stem>_meshes``).
    ``fidelity`` picks the visual LOD tier — URDF has no LOD concept, so exactly one tier is
    emitted per link and the rest are reported as dropped.
    """
    out_path = Path(out)
    source = Path(base_dir) if base_dir is not None else out_path.parent
    default_meshes = out_path.parent / f"{out_path.stem}_meshes"
    meshes = Path(assets_dir) if assets_dir is not None else default_meshes

    model = _common.realize(doc, fidelity=fidelity)
    losses = list(model.losses)
    losses += _lod_losses(doc, fidelity)
    uris = _common.write_meshes(
        model.links,
        base_dir=source,
        assets_dir=meshes,
        uri_prefix=_uri_prefix(out_path, meshes),
        losses=losses,
    )

    root = _robot(model, uris, losses)
    ET.indent(root, space="  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8"
    )
    written = tuple(sorted(meshes / Path(uri).name for uri in uris.values()))
    return ExportResult(path=out_path, mesh_paths=written, losses=tuple(losses))


def _uri_prefix(out: Path, meshes: Path) -> str:
    """The mesh directory, relative to the URDF that will reference it (never absolute).

    An absolute path baked into an export makes it unreproducible on any other machine
    (conventions.md §11), so the reference is always relative to the document.
    """
    rel = os.path.relpath(str(meshes), str(out.parent or Path()))
    return rel.replace(os.sep, "/") + "/"


def _lod_losses(doc: SadfDocument, fidelity: FidelityTier) -> list[LossFinding]:
    """Report the visual LOD tiers URDF cannot carry (it has no LOD concept at all)."""
    tiers = {ref.lod for ref in doc.asset.geometry if ref.role is GeometryRole.VISUAL}
    if len(tiers) <= 1:
        return []
    kept = geometry.lod_for_tier(fidelity)
    return [
        LossFinding(
            "urdf.lod_dropped",
            "asset.geometry",
            f"the asset carries visual LOD tiers {sorted(tiers)}; URDF has no LOD concept, so "
            f"only lod={kept} (fidelity {fidelity.value!r}) is exported. Re-export with another "
            "--fidelity to emit a different tier, or use USD, which carries the whole ladder",
        )
    ]


# --- XML -------------------------------------------------------------------------


def _robot(model: Realization, uris: dict[int, str], losses: list[LossFinding]) -> ET.Element:
    root = ET.Element("robot", {"name": model.name})
    for link in model.links:
        _link(root, link, uris)
    for joint in model.joints:
        _joint(root, joint, losses)
    return root


def _link(parent: ET.Element, link: RealizedLink, uris: dict[int, str]) -> None:
    element = ET.SubElement(parent, "link", {"name": link.name})
    if link.has_inertia:
        mass, com, inertia = _common.composite_inertial(link.bodies)
        inertial = ET.SubElement(element, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": fmt_vec((com.x, com.y, com.z)), "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": fmt_float(mass)})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": fmt_float(inertia.ixx),
                "ixy": fmt_float(inertia.ixy),
                "ixz": fmt_float(inertia.ixz),
                "iyy": fmt_float(inertia.iyy),
                "iyz": fmt_float(inertia.iyz),
                "izz": fmt_float(inertia.izz),
            },
        )
    for ref in _common.usable_refs(link):
        uri = uris.get(id(ref))
        if uri is None:  # its mesh could not be read; already reported
            continue
        tag = "collision" if ref.role is GeometryRole.COLLISION else "visual"
        node = ET.SubElement(element, tag)
        # The mesh is already expressed in the link frame (the importers normalized it there),
        # so its <origin> is the identity — URDF's origin is a *pose*, not a second chance to
        # re-express the vertices.
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "mesh", {"filename": uri})


def _joint(parent: ET.Element, joint: RealizedJoint, losses: list[LossFinding]) -> None:
    element = ET.SubElement(parent, "joint", {"name": joint.name, "type": joint.type})
    ET.SubElement(
        element,
        "origin",
        {
            "xyz": fmt_vec(joint.origin[:3, 3]),
            "rpy": fmt_vec(_common.matrix_to_rpy(joint.origin[:3, :3])),
        },
    )
    ET.SubElement(element, "parent", {"link": joint.parent})
    ET.SubElement(element, "child", {"link": joint.child})
    if joint.axis is not None:
        ET.SubElement(element, "axis", {"xyz": fmt_vec((joint.axis.x, joint.axis.y, joint.axis.z))})
    _limit(element, joint, losses)


def _limit(parent: ET.Element, joint: RealizedJoint, losses: list[LossFinding]) -> None:
    """URDF's ``<limit>``: bounds on a revolute/prismatic joint, effort and velocity on any."""
    limits: JointLimits | None = joint.limits
    if limits is None:
        if joint.type in _MUST_BE_BOUNDED:
            losses.append(_unbounded(joint))
        return

    attrs: dict[str, str] = {}
    if limits.position_rad is not None:
        attrs["lower"] = fmt_float(limits.position_rad.min)
        attrs["upper"] = fmt_float(limits.position_rad.max)
    elif joint.type in _MUST_BE_BOUNDED:
        losses.append(_unbounded(joint))
    # URDF demands effort and velocity on a <limit>; SADF leaves them optional. Zero is URDF's
    # own "unspecified" (it is what every tool writes when it does not know), and unlike an
    # invented bound it claims nothing about the hardware.
    attrs["effort"] = fmt_float(limits.effort_nm if limits.effort_nm is not None else 0.0)
    attrs["velocity"] = fmt_float(
        limits.velocity_rad_s if limits.velocity_rad_s is not None else 0.0
    )
    ET.SubElement(parent, "limit", attrs)


def _unbounded(joint: RealizedJoint) -> LossFinding:
    return LossFinding(
        "urdf.unbounded_joint",
        f"asset.joints[{joint.name!r}].limits",
        f"joint {joint.name!r} is {joint.type} but declares no position range; URDF requires a "
        "bounded <limit> on revolute/prismatic joints, so the export omits it and a strict URDF "
        "consumer will reject the joint. Declare `limits.position_rad` on the SADF joint",
    )
