# SPDX-License-Identifier: Apache-2.0
"""SADF -> URDF/SDF/USD exporters + the asset preview (RM-P0-FLEET-01, RM-P0-FLEET-02).

The mirror of :mod:`astro_mine.fleet.importers`, completing the **bidirectional URDF/SDF ↔ SADF
converters** fleet.md §11 recommends. SADF stays authoritative: an export is an *interop artifact*
for a foreign ecosystem — ROS/Gazebo (URDF/SDF), Sim/Studio/Isaac (USD), View (glTF preview) —
never a source of truth, and never round-tripped back over the asset it came from.

- :func:`export_urdf` / :func:`export_sdf` — the ROS-ecosystem forms, with body-frame OBJ meshes
  written beside them (a URDF consumer reads OBJ/STL/DAE, not the glTF and USD SADF carries).
- :func:`export_usd` — the Sim/Studio form: the whole frame tree, ``UsdPhysics`` mass and joints,
  and every LOD tier.
- :func:`render_preview` — one composed, posed glTF/USD file: the asset thumbnail.

**Every export is lossy, and says so.** No robot-description format can hold a spacecraft's power
budget, thermal envelope, sensor observation models, or capability tags — which is precisely why
SADF exists (fleet.md §11 "lossy-but-documented"). Each exporter returns an
:class:`~astro_mine.fleet.exporters._common.ExportResult` whose ``losses`` are
:class:`~astro_mine.fleet.exporters._common.LossFinding`\\ s in the same shape ``fleet lint``
reports (rule id / path / message), and :data:`LOSS_CONTRACT` states, per direction, exactly what
survives and what does not. ``fleet export --json`` prints them; nothing is dropped in silence.

Backlog: RM-P0-FLEET-01, RM-P0-FLEET-02 --
astro-mine-fleet#31
"""

from __future__ import annotations

from pathlib import Path

from astro_mine.core.sadf import SadfDocument
from astro_mine.core.sadf.enums import FidelityTier
from astro_mine.fleet.exporters._common import (
    ExportError,
    ExportResult,
    LossFinding,
    Realization,
    realize,
)
from astro_mine.fleet.exporters.render import PREVIEW_FORMATS, render_preview
from astro_mine.fleet.exporters.sdf import export_sdf
from astro_mine.fleet.exporters.urdf import export_urdf
from astro_mine.fleet.exporters.usd import export_usd

__all__ = [
    "FORMATS",
    "LOSS_CONTRACT",
    "PREVIEW_FORMATS",
    "ExportError",
    "ExportResult",
    "LossFinding",
    "Realization",
    "export_description",
    "export_sdf",
    "export_urdf",
    "export_usd",
    "realize",
    "render_preview",
]

#: The export targets, by ``--format`` name.
FORMATS: tuple[str, ...] = ("urdf", "sdf", "usd")

#: The **fidelity contract** (fleet.md §11's open question, answered): what each converter
#: direction preserves and what it cannot, keyed by ``"<from>-><to>"``.
#:
#: The rule of the table: a converter loses a field only where the *target format has no way to
#: state it*. Nothing is dropped for convenience, and every drop is reported at runtime as a
#: :class:`LossFinding` whose ``rule`` is one of the ids named here.
LOSS_CONTRACT: dict[str, tuple[str, ...]] = {
    "sadf->urdf": (
        "PRESERVED: the kinematic tree (one link per SADF frame), link masses, centres of mass, "
        "inertia tensors, frame poses, joint types/axes/position bounds, and visual + collision "
        "geometry (re-materialized as body-frame OBJ).",
        "asset.block_dropped: power, thermal, sensors, comms, actuators, mobility, propulsion, "
        "payload, capability tags, and fidelity profiles. URDF describes a robot's kinematics; "
        "SADF describes a spacecraft. Sim reads these from SADF, never from a URDF.",
        "link.bodies_merged: several SADF bodies on one frame become one URDF inertial "
        "(physically exact — mass sum + parallel-axis inertia — but their *names* are lost).",
        "body names: a URDF link is named after its SADF *frame*, because frames are what "
        "geometry and sensors reference. A re-import therefore names each body after its frame.",
        "inertial rotation: URDF's <inertial><origin rpy> is written as zero and the tensor in "
        "body-frame axes. An import that rotated a tensor in (I' = R·I·Rᵀ) round-trips to this "
        "equal-and-simpler spelling: the physics is exact, the XML is normalized.",
        "urdf.lod_dropped: URDF has no LOD concept; one visual tier is emitted (--fidelity picks "
        "which). Use USD to carry the whole ladder.",
        "urdf.unbounded_joint: URDF requires a bounded <limit> on revolute/prismatic joints; a "
        "SADF joint that declares none is exported without one.",
        "joint.effort_unit: a prismatic joint's SADF `effort_nm` is a *force* (N) in URDF — the "
        "number crosses unchanged, the format reinterprets the unit (as it did on import).",
    ),
    "sadf->sdf": (
        "PRESERVED: everything URDF preserves, plus SDF's own optional-limit semantics (a missing "
        "effort/velocity means unbounded, so none has to be invented).",
        "asset.block_dropped / link.bodies_merged / sdf.lod_dropped: as for URDF.",
        "sdf.frames_flattened: link <pose> is written in the *model* frame (SDF ≤1.6 semantics — "
        "the dialect Fleet's SDF importer reads). World poses survive exactly; the frame "
        "*hierarchy* does not, so a re-import parents every frame directly to the model root.",
    ),
    "sadf->usd": (
        "PRESERVED: the frame tree (USD nests natively — nothing is flattened), masses, centres "
        "of mass, inertia tensors (as diagonal + principal axes), joint types/axes/position "
        "bounds, geometry references, and — uniquely — **every LOD tier**.",
        "asset.block_dropped: as for URDF/SDF.",
        "usd.float32_precision: UsdPhysics stores mass, centre of mass, and inertia as 32-bit "
        "floats (SADF carries doubles), so mass properties survive a USD round trip to ~1e-7 "
        "relative, not exactly. Frame poses are authored double-precision and are unaffected.",
        "usd.inertia_diagonalized: UsdPhysics stores a diagonal tensor + principal axes, so a "
        "tensor with products of inertia is diagonalized. R·diag·Rᵀ recovers it exactly.",
        "usd.drive_limits_dropped: a UsdPhysics joint holds position bounds only; effort and "
        "velocity describe an actuator (PhysicsDriveAPI), not the joint.",
        "glTF refs: the stage references the USD twin of each mesh, not the glTF one — a USD "
        "stage references USD. The glTF refs stay in the SADF, which remains authoritative.",
    ),
    "urdf->sadf": (
        "PRESERVED: links → frames + bodies (inertia rotated into the body frame), the joint "
        "tree, joint limits, and visual/collision geometry (normalized to USD + glTF, with a "
        "convex collision hull and LOD tiers generated).",
        "a joint to a *massless* link is dropped: SADF joints connect bodies, and the rigid "
        "attachment it expressed is already carried by the link's frame parenthood. Exporting "
        "back re-synthesizes it as a fixed joint, so the round trip closes.",
        "<transmission>, <gazebo>, <mimic>, and <safety_controller> have no SADF counterpart and "
        "are not read.",
    ),
    "sdf->sadf": (
        "PRESERVED: as for URDF, with link <pose> read in the model frame.",
        "SDF 1.7 `relative_to` frame graphs, multi-model worlds, plugins, and sensors are not "
        "read (deferred, P1).",
    ),
    "usd->sadf": (
        "PRESERVED: the Xform tree → frames, UsdPhysics mass/inertia → bodies, UsdPhysics joints "
        "→ joints (axis tokens and joint-frame rotations inverted back to a SADF axis), and "
        "referenced geometry.",
        "the reader targets the stage *Fleet writes* — an Xform tree with UsdPhysics APIs. A "
        "general Omniverse/Isaac stage (variants, materials, instancing, articulation roots) is "
        "not a SADF asset and is not read (deferred).",
    ),
}


def export_description(
    doc: SadfDocument,
    out: str | Path,
    *,
    fmt: str,
    base_dir: str | Path | None = None,
    assets_dir: str | Path | None = None,
    fidelity: FidelityTier = FidelityTier.ARTICULATED,
) -> ExportResult:
    """Export ``doc`` to ``fmt`` (``"urdf"``, ``"sdf"``, or ``"usd"``) at ``out``.

    The dispatching front door ``fleet export`` uses. ``base_dir`` is where the document's
    geometry uris resolve from; ``assets_dir`` is where URDF/SDF write their OBJ meshes (USD
    references the SADF geometry in place and ignores it); ``fidelity`` picks the visual LOD tier
    for the formats that can only carry one.
    """
    if fmt == "usd":
        return export_usd(doc, out, base_dir=base_dir)
    if fmt == "urdf":
        return export_urdf(doc, out, base_dir=base_dir, assets_dir=assets_dir, fidelity=fidelity)
    if fmt == "sdf":
        return export_sdf(doc, out, base_dir=base_dir, assets_dir=assets_dir, fidelity=fidelity)
    raise ExportError(f"unknown export format {fmt!r} (expected one of {FORMATS})")
