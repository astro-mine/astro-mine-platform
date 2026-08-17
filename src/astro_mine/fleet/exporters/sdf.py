# SPDX-License-Identifier: Apache-2.0
"""SADF -> SDF (RM-P0-FLEET-01/02), the Gazebo interop form.

The inverse of :mod:`astro_mine.fleet.importers.sdf`, and a sibling of the URDF writer: the same
link tree, spelled in SDF's element-per-value XML.

**Poses are model-frame.** SDF ≤1.6 reads a ``<link><pose>`` relative to the *model* frame, and
that is the dialect Fleet's SDF importer speaks; ``relative_to`` frame graphs are a 1.7+ feature
it defers. The exporter therefore writes each link's **composed root-relative pose**, which is
what that dialect means, rather than a parent-relative one it would misread. World poses are
preserved exactly; what is lost is the *hierarchy* — a re-import parents every link directly to
the model root (see ``LOSS_CONTRACT``, ``sdf.frames_flattened``).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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

__all__ = ["export_sdf"]

#: The SDF version the exporter targets. 1.7 is the current Gazebo dialect; poses are written in
#: the model frame, which 1.7 still accepts as the default when no ``relative_to`` is given.
SDF_VERSION = "1.7"


def export_sdf(
    doc: SadfDocument,
    out: str | Path,
    *,
    base_dir: str | Path | None = None,
    assets_dir: str | Path | None = None,
    fidelity: FidelityTier = FidelityTier.ARTICULATED,
) -> ExportResult:
    """Write ``doc`` as an SDF model at ``out``; return the artifact, its meshes, and the losses.

    Arguments mirror :func:`~astro_mine.fleet.exporters.urdf.export_urdf`.
    """
    out_path = Path(out)
    source = Path(base_dir) if base_dir is not None else out_path.parent
    default_meshes = out_path.parent / f"{out_path.stem}_meshes"
    meshes = Path(assets_dir) if assets_dir is not None else default_meshes

    model = _common.realize(doc, fidelity=fidelity)
    losses = list(model.losses)
    losses += _flatten_loss(model)
    losses += _lod_losses(doc, fidelity)
    rel = os.path.relpath(str(meshes), str(out_path.parent or Path())).replace(os.sep, "/") + "/"
    uris = _common.write_meshes(
        model.links, base_dir=source, assets_dir=meshes, uri_prefix=rel, losses=losses
    )

    root = _sdf(model, uris)
    ET.indent(root, space="  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8"
    )
    written = tuple(sorted(meshes / Path(uri).name for uri in uris.values()))
    return ExportResult(path=out_path, mesh_paths=written, losses=tuple(losses))


def _flatten_loss(model: Realization) -> list[LossFinding]:
    """Flag the frame hierarchy the model-frame pose convention flattens."""
    nested = [joint.child for joint in model.joints if joint.parent != model.root]
    if not nested:
        return []
    return [
        LossFinding(
            "sdf.frames_flattened",
            "asset.frames",
            f"{len(nested)} frame(s) are nested below a non-root parent ({', '.join(nested)}); "
            "link <pose> is written in the model frame (SDF ≤1.6 semantics, the dialect Fleet's "
            "importer reads), so world poses survive exactly but a re-import parents every frame "
            "directly to the model root",
        )
    ]


def _lod_losses(doc: SadfDocument, fidelity: FidelityTier) -> list[LossFinding]:
    tiers = {ref.lod for ref in doc.asset.geometry if ref.role is GeometryRole.VISUAL}
    if len(tiers) <= 1:
        return []
    return [
        LossFinding(
            "sdf.lod_dropped",
            "asset.geometry",
            f"the asset carries visual LOD tiers {sorted(tiers)}; SDF has no LOD concept, so only "
            f"lod={geometry.lod_for_tier(fidelity)} (fidelity {fidelity.value!r}) is exported",
        )
    ]


# --- XML -------------------------------------------------------------------------


def _sdf(model: Realization, uris: dict[int, str]) -> ET.Element:
    root = ET.Element("sdf", {"version": SDF_VERSION})
    element = ET.SubElement(root, "model", {"name": model.name})
    for link in model.links:
        _link(element, link, uris)
    for joint in model.joints:
        _joint(element, joint)
    return root


def _pose(parent: ET.Element, matrix: NDArray[np.float64]) -> None:
    """SDF's ``<pose>x y z roll pitch yaw</pose>``."""
    xyz = fmt_vec(matrix[:3, 3])
    rpy = fmt_vec(_common.matrix_to_rpy(matrix[:3, :3]))
    ET.SubElement(parent, "pose").text = f"{xyz} {rpy}"


def _link(parent: ET.Element, link: RealizedLink, uris: dict[int, str]) -> None:
    element = ET.SubElement(parent, "link", {"name": link.name})
    _pose(element, link.world)  # model-frame, per the module docstring

    if link.has_inertia:
        mass, com, inertia = _common.composite_inertial(link.bodies)
        inertial = ET.SubElement(element, "inertial")
        ET.SubElement(inertial, "pose").text = f"{fmt_vec((com.x, com.y, com.z))} 0.0 0.0 0.0"
        ET.SubElement(inertial, "mass").text = fmt_float(mass)
        tensor = ET.SubElement(inertial, "inertia")
        for tag, value in (
            ("ixx", inertia.ixx),
            ("ixy", inertia.ixy),
            ("ixz", inertia.ixz),
            ("iyy", inertia.iyy),
            ("iyz", inertia.iyz),
            ("izz", inertia.izz),
        ):
            ET.SubElement(tensor, tag).text = fmt_float(value)

    for index, ref in enumerate(_common.usable_refs(link)):
        uri = uris.get(id(ref))
        if uri is None:  # its mesh could not be read; already reported
            continue
        collision = ref.role is GeometryRole.COLLISION
        tag = "collision" if collision else "visual"
        node = ET.SubElement(element, tag, {"name": f"{link.name}_{tag}{index}"})
        mesh = ET.SubElement(ET.SubElement(node, "geometry"), "mesh")
        ET.SubElement(mesh, "uri").text = uri


def _joint(parent: ET.Element, joint: RealizedJoint) -> None:
    element = ET.SubElement(parent, "joint", {"name": joint.name, "type": joint.type})
    ET.SubElement(element, "parent").text = joint.parent
    ET.SubElement(element, "child").text = joint.child
    if joint.axis is None:
        return
    axis = ET.SubElement(element, "axis")
    ET.SubElement(axis, "xyz").text = fmt_vec((joint.axis.x, joint.axis.y, joint.axis.z))
    _limit(axis, joint.limits)


def _limit(parent: ET.Element, limits: JointLimits | None) -> None:
    """SDF's ``<axis><limit>``: only the bounds the SADF joint actually declares.

    Unlike URDF, SDF defaults a missing ``<effort>``/``<velocity>`` to unlimited, so an omission
    here says "unbounded" rather than "zero" — no invented value is needed.
    """
    if limits is None:
        return
    limit = ET.SubElement(parent, "limit")
    if limits.position_rad is not None:
        ET.SubElement(limit, "lower").text = fmt_float(limits.position_rad.min)
        ET.SubElement(limit, "upper").text = fmt_float(limits.position_rad.max)
    if limits.effort_nm is not None:
        ET.SubElement(limit, "effort").text = fmt_float(limits.effort_nm)
    if limits.velocity_rad_s is not None:
        ET.SubElement(limit, "velocity").text = fmt_float(limits.velocity_rad_s)
