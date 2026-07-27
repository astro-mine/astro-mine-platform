"""SADF -> URDF/SDF/USD exporters, the preview, and the loss contract (RM-P0-FLEET-01/02).

The *round-trip* tests — the fleet.md §10 test class — live in ``test_roundtrip.py``. This file
covers what each writer emits, the link-tree mapping it emits it from, and the lossy edges it is
required to report rather than swallow (fleet.md §11 "lossy-but-documented").
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yourdfpy

from astro_mine.core.sadf import SadfDocument, load_sadf
from astro_mine.core.sadf.enums import FidelityTier, GeometryRole
from astro_mine.fleet import exporters, importers, library
from astro_mine.fleet.exporters import ExportError

from .conftest import VALID_SADF

# --- fixtures --------------------------------------------------------------------

URDF_SOURCE = """<?xml version="1.0"?>
<robot name="rover">
  <link name="base">
    <inertial><origin xyz="0 0 0.1"/><mass value="5"/>
      <inertia ixx="0.2" iyy="0.3" izz="0.4" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.4 0.3 0.2"/></geometry></visual>
  </link>
  <link name="arm">
    <inertial><mass value="1"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><cylinder radius="0.05" length="0.4"/></geometry></visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.2"/><axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" velocity="2" effort="10"/></joint>
</robot>
"""


@pytest.fixture
def rover() -> SadfDocument:
    """The anchor-scenario prospecting rover: three bodies on three frames, a continuous + a
    prismatic joint, a massless sensor mast, and no geometry at all — the shape every Phase-0
    reference asset has, and the "representative SADF asset (e.g. a library rover)" the
    acceptance criteria name."""
    return library.load_reference("prospecting_rover")


@pytest.fixture
def imported(tmp_path: Path) -> tuple[SadfDocument, Path]:
    """A SADF asset *with* geometry: imported from URDF, so it carries LOD tiers and hulls."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "rover.urdf").write_text(URDF_SOURCE, encoding="utf-8")
    doc = importers.import_urdf(source / "rover.urdf", assets_dir=source / "a", uri_prefix="a/")
    return doc, source


def rules(result: exporters.ExportResult) -> set[str]:
    return {loss.rule for loss in result.losses}


# --- the link-tree mapping (exporters._common.realize) ---------------------------


def test_realize_maps_frames_to_links_and_bodies_to_inertials(rover: SadfDocument) -> None:
    model = exporters.realize(rover)

    # one link per *frame* — including the massless mast, which URDF/SDF/USD can all hold
    assert [link.name for link in model.links] == [
        "body",
        "sensor_mast",
        "wheel_front_left",
        "drill",
    ]
    assert model.root == "body"
    assert {link.name for link in model.links if link.has_inertia} == {
        "body",
        "wheel_front_left",
        "drill",
    }


def test_realize_redraws_body_joints_between_frames(rover: SadfDocument) -> None:
    """SADF joints connect *bodies*; a link tree connects *links* — the mapping goes via frames."""
    joints = {j.name: j for j in exporters.realize(rover).joints}

    # `wheel_fl` is chassis -> wheel_front_left in SADF (bodies); here: body -> wheel_front_left
    assert (joints["wheel_fl"].parent, joints["wheel_fl"].child) == ("body", "wheel_front_left")
    assert np.allclose(joints["wheel_fl"].origin[:3, 3], (0.5, 0.4, 0.25))


def test_realize_synthesizes_a_fixed_joint_for_a_body_less_frame(rover: SadfDocument) -> None:
    """A frame's parenthood *is* a rigid attachment; a link tree can only say so with a joint."""
    mast = next(j for j in exporters.realize(rover).joints if j.child == "sensor_mast")
    assert mast.type == "fixed" and mast.synthesized and mast.parent == "body"
    assert np.allclose(mast.origin[:3, 3], (0.0, 0.0, 1.1))


def test_realize_composes_several_bodies_on_one_frame(rover: SadfDocument) -> None:
    """A link has one inertial. Two bodies on a frame compose exactly — and lose their names."""
    doc = rover.model_copy(deep=True)
    second = doc.asset.bodies[1].model_copy(update={"name": "ballast", "frame": "body"})
    doc.asset.bodies.append(second)

    model = exporters.realize(doc)
    assert "link.bodies_merged" in {loss.rule for loss in model.losses}

    mass, com, _ = exporters._common.composite_inertial(model.link("body").bodies)
    assert mass == pytest.approx(180.0 + 9.0)
    # the composite centre of mass is the mass-weighted mean of the two
    assert com.z == pytest.approx((180.0 * 0.35 + 9.0 * 0.0) / 189.0)


def test_realize_rejects_a_frame_cycle(rover: SadfDocument) -> None:
    doc = rover.model_copy(deep=True)
    frames = {f.name: f for f in doc.asset.frames}
    # sensor_mast -> drill -> sensor_mast: a closed loop that composes to no pose at all
    frames["sensor_mast"].parent = "drill"
    frames["drill"].parent = "sensor_mast"
    with pytest.raises(ExportError, match="frame cycle"):
        exporters.realize(doc)


def test_realize_rejects_two_joints_claiming_one_frame(rover: SadfDocument) -> None:
    doc = rover.model_copy(deep=True)
    doc.asset.joints[1].child_body = "wheel_front_left"  # both joints now drive the wheel
    with pytest.raises(ExportError, match="child of two joints"):
        exporters.realize(doc)


def test_realize_rejects_a_joint_that_reparents_the_root(rover: SadfDocument) -> None:
    doc = rover.model_copy(deep=True)
    doc.asset.joints[0].parent_body, doc.asset.joints[0].child_body = "wheel_front_left", "chassis"
    with pytest.raises(ExportError, match="root of a link tree has no parent"):
        exporters.realize(doc)


def test_realize_rejects_a_joint_between_bodies_on_one_frame(rover: SadfDocument) -> None:
    doc = rover.model_copy(deep=True)
    doc.asset.bodies[1].frame = "body"  # the wheel body now shares the chassis' frame
    with pytest.raises(ExportError, match="cannot be jointed to itself"):
        exporters.realize(doc)


# --- URDF ------------------------------------------------------------------------


def test_urdf_export_is_parseable_by_a_real_urdf_reader(
    rover: SadfDocument, tmp_path: Path
) -> None:
    """The acceptance criterion: a *valid* URDF from a library rover, via a third-party parser."""
    result = exporters.export_urdf(rover, tmp_path / "rover.urdf")
    robot = yourdfpy.URDF.load(str(result.path), load_meshes=False, build_scene_graph=False).robot

    assert {link.name for link in robot.links} == {
        "body",
        "sensor_mast",
        "wheel_front_left",
        "drill",
    }
    assert {j.name for j in robot.joints} == {"wheel_fl", "drill_z", "sensor_mast__fixed"}
    base = next(link for link in robot.links if link.name == "body")
    assert base.inertial.mass == pytest.approx(180.0)


def test_urdf_carries_joint_types_axes_and_limits(rover: SadfDocument, tmp_path: Path) -> None:
    result = exporters.export_urdf(rover, tmp_path / "r.urdf")
    joints = {j.get("name"): j for j in ET.parse(result.path).getroot().findall("joint")}

    assert joints["wheel_fl"].get("type") == "continuous"
    assert joints["drill_z"].get("type") == "prismatic"
    limit = joints["drill_z"].find("limit")
    assert limit is not None
    assert float(limit.get("lower")) == pytest.approx(-0.5)
    assert float(limit.get("upper")) == pytest.approx(0.0)
    assert joints["wheel_fl"].find("axis").get("xyz") == "0.0 1.0 0.0"


def test_urdf_writes_body_frame_obj_meshes_a_ros_consumer_can_read(
    imported: tuple[SadfDocument, Path], tmp_path: Path
) -> None:
    """A URDF consumer reads OBJ/STL/DAE — never the glTF and USD that SADF carries."""
    doc, source = imported
    result = exporters.export_urdf(doc, tmp_path / "out" / "r.urdf", base_dir=source)

    meshes = ET.parse(result.path).getroot().findall(".//mesh")
    assert meshes and all(m.get("filename").endswith(".obj") for m in meshes)
    # every referenced mesh resolves relative to the URDF, and none of them is an absolute path
    for mesh in meshes:
        uri = mesh.get("filename")
        assert not Path(uri).is_absolute()
        assert (result.path.parent / uri).is_file()

    # the mesh really is in the body frame: the base box was authored 0.4 x 0.3 x 0.2
    base = trimesh.load(result.path.parent / "r_meshes" / "base__visual0.obj", force="mesh")
    assert np.allclose(sorted(base.extents), sorted((0.4, 0.3, 0.2)))


def test_urdf_emits_one_lod_tier_and_says_which(
    imported: tuple[SadfDocument, Path], tmp_path: Path
) -> None:
    """URDF has no LOD concept. The tiers it cannot carry are reported, not silently dropped."""
    doc, source = imported
    coarse = exporters.export_urdf(
        doc, tmp_path / "c.urdf", base_dir=source, fidelity=FidelityTier.MASSMODEL
    )
    fine = exporters.export_urdf(
        doc, tmp_path / "f.urdf", base_dir=source, fidelity=FidelityTier.ARTICULATED
    )

    assert "urdf.lod_dropped" in rules(coarse)

    def names(result: exporters.ExportResult) -> set[str]:
        return {m.get("filename") for m in ET.parse(result.path).getroot().findall(".//mesh")}

    assert any(".lod2." in n for n in names(coarse))  # massmodel -> the coarsest tier
    assert not any(".lod" in n for n in names(fine))  # articulated -> the full-resolution mesh
    # and the coarse tier really is cheaper on disk
    coarse_mesh = next(p for p in coarse.mesh_paths if "visual" in p.name)
    fine_mesh = next(p for p in fine.mesh_paths if "visual" in p.name)
    assert coarse_mesh.stat().st_size <= fine_mesh.stat().st_size


def test_urdf_reports_an_unbounded_revolute_joint(rover: SadfDocument, tmp_path: Path) -> None:
    """URDF requires a bounded <limit> on a revolute joint; SADF does not. Say so."""
    doc = rover.model_copy(deep=True)
    doc.asset.joints[0].type = doc.asset.joints[0].type.__class__.REVOLUTE  # was continuous
    doc.asset.joints[0].limits = None

    result = exporters.export_urdf(doc, tmp_path / "r.urdf")
    assert "urdf.unbounded_joint" in rules(result)


def test_urdf_reports_the_spacecraft_blocks_no_robot_format_holds(
    rover: SadfDocument, tmp_path: Path
) -> None:
    """The whole reason SADF exists: URDF cannot state a power budget or a neutron spectrometer."""
    result = exporters.export_urdf(rover, tmp_path / "r.urdf")
    dropped = {loss.path for loss in result.losses if loss.rule == "asset.block_dropped"}
    assert {"asset.power", "asset.thermal", "asset.sensors", "asset.comms"} <= dropped
    assert "asset.capabilities" in dropped


def test_urdf_flags_the_prismatic_effort_unit(rover: SadfDocument, tmp_path: Path) -> None:
    """A prismatic joint's SADF ``effort_nm`` is a *force* in URDF. The number crosses; the unit
    is reinterpreted — which is exactly the kind of thing that must not be silent."""
    result = exporters.export_urdf(rover, tmp_path / "r.urdf")
    assert "joint.effort_unit" in rules(result)


# --- SDF -------------------------------------------------------------------------


def test_sdf_export_is_wellformed_and_reimportable(rover: SadfDocument, tmp_path: Path) -> None:
    """The acceptance criterion: a valid SDF from the library rover."""
    result = exporters.export_sdf(rover, tmp_path / "rover.sdf")
    model = ET.parse(result.path).getroot().find("model")

    assert model.get("name") == "astro-mine.fleet.prospecting-rover"
    assert {link.get("name") for link in model.findall("link")} == {
        "body",
        "sensor_mast",
        "wheel_front_left",
        "drill",
    }
    # Fleet's own SDF importer must be able to read what its SDF exporter writes.
    back = importers.import_sdf(result.path, assets_dir=tmp_path / "a")
    assert {b.name for b in back.asset.bodies} == {"body", "wheel_front_left", "drill"}


def test_sdf_writes_model_frame_poses_and_says_it_flattened(
    rover: SadfDocument, tmp_path: Path
) -> None:
    """SDF <=1.6 reads a link <pose> in the model frame. World poses survive; hierarchy does not."""
    doc = rover.model_copy(deep=True)
    # Nest the (body-less) mast under the drill, so the link tree is genuinely two deep and there
    # is a hierarchy to lose. The rover as authored is already flat — every frame hangs off the
    # root — so it would have nothing to flatten and would (rightly) report no loss.
    next(f for f in doc.asset.frames if f.name == "sensor_mast").parent = "drill"

    result = exporters.export_sdf(doc, tmp_path / "r.sdf")
    assert "sdf.frames_flattened" in rules(result)

    model = ET.parse(result.path).getroot().find("model")
    mast = next(link for link in model.findall("link") if link.get("name") == "sensor_mast")
    # the mast now sits at drill (0.4, 0, 0.2) + its own (0, 0, 1.1) => (0.4, 0, 1.3) in the model
    # frame — the *world* pose, which is what survives, even though the parenthood does not
    assert [float(v) for v in mast.find("pose").text.split()[:3]] == pytest.approx([0.4, 0.0, 1.3])


def test_sdf_omits_a_limit_it_does_not_have(rover: SadfDocument, tmp_path: Path) -> None:
    """Unlike URDF, a missing SDF <effort>/<velocity> means *unbounded* — so invent nothing."""
    result = exporters.export_sdf(rover, tmp_path / "r.sdf")
    model = ET.parse(result.path).getroot().find("model")
    drill = next(j for j in model.findall("joint") if j.get("name") == "drill_z")

    limit = drill.find("axis/limit")
    assert limit.find("effort") is not None  # the SADF joint declares one
    assert limit.find("velocity") is None  # it does not declare this one, so SDF says nothing


# --- USD -------------------------------------------------------------------------


def test_usd_export_is_a_valid_stage_with_physics(rover: SadfDocument, tmp_path: Path) -> None:
    """The acceptance criterion: a valid USD stage from the library rover."""
    from pxr import Usd, UsdGeom, UsdPhysics

    result = exporters.export_usd(rover, tmp_path / "rover.usda")
    stage = Usd.Stage.Open(str(result.path))

    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z  # the SADF body frame
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0  # SI, always
    assert stage.GetDefaultPrim().GetName() == "body"

    chassis = stage.GetPrimAtPath("/body")
    assert chassis.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(chassis).GetMassAttr().Get() == pytest.approx(180.0)

    # the frame tree nests natively — nothing is flattened, unlike SDF
    assert stage.GetPrimAtPath("/body/wheel_front_left").IsValid()
    assert stage.GetPrimAtPath("/body/wheel_front_left/wheel_fl").IsA(UsdPhysics.RevoluteJoint)
    assert stage.GetPrimAtPath("/body/drill/drill_z").IsA(UsdPhysics.PrismaticJoint)
    assert stage.GetPrimAtPath("/body/sensor_mast/sensor_mast__fixed").IsA(UsdPhysics.FixedJoint)


def test_usd_carries_every_lod_tier(imported: tuple[SadfDocument, Path], tmp_path: Path) -> None:
    """USD is the one target that can hold the whole ladder — so it must not throw tiers away."""
    from pxr import Usd

    doc, source = imported
    result = exporters.export_usd(doc, source / "rover.usda", base_dir=source)
    stage = Usd.Stage.Open(str(result.path))

    tiers = {
        int(prim.GetAttribute("astroMine:lod").Get())
        for prim in Usd.PrimRange(stage.GetDefaultPrim())
        if prim.GetAttribute("astroMine:role")
        and str(prim.GetAttribute("astroMine:role").Get()) == "visual"
    }
    assert tiers == {0, 1, 2}
    assert "usd.lod_dropped" not in rules(result)


def test_usd_diagonalizes_a_tensor_with_products_of_inertia(
    rover: SadfDocument, tmp_path: Path
) -> None:
    """UsdPhysics stores diag + principal axes. R.diag.R^T must recover the tensor exactly."""
    from pxr import Usd, UsdPhysics

    doc = rover.model_copy(deep=True)
    doc.asset.bodies[0].inertia_kg_m2.ixy = 3.0

    result = exporters.export_usd(doc, tmp_path / "r.usda")
    assert "usd.inertia_diagonalized" in rules(result)

    stage = Usd.Stage.Open(str(result.path))
    api = UsdPhysics.MassAPI(stage.GetPrimAtPath("/body"))
    moments = np.array(api.GetDiagonalInertiaAttr().Get())
    quat = api.GetPrincipalAxesAttr().Get()

    axes = importers.usd._rotation(quat)
    assert np.allclose(
        axes @ np.diag(moments) @ axes.T,
        [[28.0, 3.0, 0.0], [3.0, 34.0, 0.0], [0.0, 0.0, 22.0]],
        atol=1e-4,  # float32 storage (usd.float32_precision)
    )


def test_usd_reports_the_float32_narrowing(rover: SadfDocument, tmp_path: Path) -> None:
    """UsdPhysics mass attributes are single-precision; SADF carries doubles. Say so."""
    result = exporters.export_usd(rover, tmp_path / "r.usda")
    assert "usd.float32_precision" in rules(result)
    assert "usd.drive_limits_dropped" in rules(result)


def test_usd_encodes_a_non_basis_joint_axis_in_the_joint_frame(
    rover: SadfDocument, tmp_path: Path
) -> None:
    """UsdPhysics restricts an axis to the tokens X/Y/Z; an arbitrary axis rides on localRot."""
    from pxr import Usd

    doc = rover.model_copy(deep=True)
    axis = doc.asset.joints[0].axis
    axis.x, axis.y, axis.z = 0.0, 0.6, 0.8  # a unit axis that is no basis vector

    result = exporters.export_usd(doc, tmp_path / "r.usda")
    stage = Usd.Stage.Open(str(result.path))
    joint = stage.GetPrimAtPath("/body/wheel_front_left/wheel_fl")

    token = str(joint.GetAttribute("physics:axis").Get())
    align = importers.usd._rotation(joint.GetAttribute("physics:localRot1").Get())
    # the token's basis vector, rotated by the joint frame, is the SADF axis again
    assert np.allclose(align @ exporters.usd.AXIS_TOKENS[token], (0.0, 0.6, 0.8), atol=1e-6)


# --- render ----------------------------------------------------------------------


def test_render_previews_an_asset_with_geometry(
    imported: tuple[SadfDocument, Path], tmp_path: Path
) -> None:
    """The acceptance criterion: a preview, with no GPU, no renderer, and no network."""
    doc, source = imported
    result = exporters.render_preview(doc, tmp_path / "preview.glb", base_dir=source)

    scene = trimesh.load(result.path)
    assert len(scene.geometry) == 2  # one visual per link (base, arm)
    assert not result.losses  # the asset has real geometry: nothing was substituted


def test_render_uses_inertia_proxies_for_a_mesh_free_asset(
    rover: SadfDocument, tmp_path: Path
) -> None:
    """Every Phase-0 reference asset is mesh-free. A preview of one is boxes, and says so."""
    result = exporters.render_preview(rover, tmp_path / "preview.glb")

    assert "render.proxy_geometry" in rules(result)
    scene = trimesh.load(result.path)
    assert len(scene.geometry) == 3  # chassis, wheel, drill head — the three bodies


def test_render_poses_each_link_in_the_root_frame(rover: SadfDocument, tmp_path: Path) -> None:
    """A preview whose parts all sit at the origin is not a preview of a robot."""
    result = exporters.render_preview(rover, tmp_path / "p.glb")
    scene = trimesh.load(result.path)

    # the wheel proxy sits at the wheel frame (0.5, 0.4, 0.25); glTF is y-up, so body z -> gltf y
    centroids = np.array([g.centroid for g in scene.geometry.values()])
    assert centroids.std(axis=0).max() > 0.1  # the parts are genuinely spread out


def test_render_writes_a_usd_preview(rover: SadfDocument, tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    result = exporters.render_preview(rover, tmp_path / "p.usda", fmt="usd")
    stage = Usd.Stage.Open(str(result.path))
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 3
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z


def test_render_is_deterministic(rover: SadfDocument, tmp_path: Path) -> None:
    """Same asset, same bytes (conventions.md §11) — no timestamps, no dict-order drift."""
    first = exporters.render_preview(rover, tmp_path / "a.glb")
    second = exporters.render_preview(rover, tmp_path / "b.glb")
    assert first.path.read_bytes() == second.path.read_bytes()


def test_render_rejects_an_unknown_format(rover: SadfDocument, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="unknown preview format"):
        exporters.render_preview(rover, tmp_path / "p.png", fmt="png")


def test_render_refuses_an_asset_with_nothing_to_show(tmp_path: Path) -> None:
    """A mass-less, mesh-less asset has neither geometry nor a body to derive a proxy from."""
    doc = load_sadf(VALID_SADF)
    assert not doc.asset.geometry and not doc.asset.bodies
    with pytest.raises(ExportError, match="nothing to preview"):
        exporters.render_preview(doc, tmp_path / "p.glb")


# --- dispatch + the loss contract ------------------------------------------------


@pytest.mark.parametrize("fmt", exporters.FORMATS)
def test_export_description_dispatches(rover: SadfDocument, tmp_path: Path, fmt: str) -> None:
    result = exporters.export_description(rover, tmp_path / f"r.{fmt}", fmt=fmt)
    assert result.path.is_file() and result.path.stat().st_size > 0


def test_export_description_rejects_an_unknown_format(rover: SadfDocument, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="unknown export format"):
        exporters.export_description(rover, tmp_path / "r.dae", fmt="collada")


def test_every_reported_loss_rule_is_in_the_documented_contract(
    rover: SadfDocument, imported: tuple[SadfDocument, Path], tmp_path: Path
) -> None:
    """fleet.md §11's open question, answered and *enforced*.

    A converter may only lose what ``LOSS_CONTRACT`` says it loses. If a new lossy edge appears at
    runtime without an entry in the table, that is precisely the "silent loss" the contract exists
    to prevent — so the test fails until it is written down.
    """
    documented = " ".join(" ".join(v) for v in exporters.LOSS_CONTRACT.values())
    doc, source = imported

    reported: set[str] = set()
    for fmt in exporters.FORMATS:
        reported |= rules(exporters.export_description(rover, tmp_path / f"lib.{fmt}", fmt=fmt))
        reported |= rules(
            exporters.export_description(doc, source / f"imp.{fmt}", fmt=fmt, base_dir=source)
        )

    assert reported, "the exporters reported no losses at all, which cannot be right"
    undocumented = {rule for rule in reported if rule not in documented}
    assert not undocumented, f"lossy edges reported but not in LOSS_CONTRACT: {undocumented}"


def test_the_loss_contract_covers_every_converter_direction() -> None:
    assert set(exporters.LOSS_CONTRACT) == {
        "sadf->urdf",
        "sadf->sdf",
        "sadf->usd",
        "urdf->sadf",
        "sdf->sadf",
        "usd->sadf",
    }


def test_geometry_that_cannot_be_read_is_reported_not_crashed(
    rover: SadfDocument, tmp_path: Path
) -> None:
    """A ref pointing at a missing file must degrade to a mesh-free link, loudly."""
    from astro_mine.core.sadf.enums import GeometryFormat
    from astro_mine.core.sadf.model import GeometryRef

    doc = rover.model_copy(deep=True)
    doc.asset.geometry.append(
        GeometryRef(
            role=GeometryRole.VISUAL, format=GeometryFormat.GLTF, uri="gone.glb", frame="body"
        )
    )
    result = exporters.export_urdf(doc, tmp_path / "r.urdf")
    assert "geometry.unresolved" in rules(result)
    assert result.path.is_file()  # the export still succeeded
