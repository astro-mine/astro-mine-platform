"""URDF/SDF importers (RM-P0-FLEET-02): mapping, geometry, dispatch, and the CLI."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from astro_mine.core.sadf import validate_sadf
from astro_mine.core.sadf.enums import GeometryRole, JointType
from astro_mine.fleet import importers
from astro_mine.fleet.importers import _common, sdf
from astro_mine.fleet.importers._common import ImportError_


def write_mesh(path: Path) -> None:
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(path)


# a 5-link URDF exercising every joint type, geometry primitive, collision fallback,
# a rotated inertial frame, and a massless link.
URDF = """<?xml version="1.0"?>
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
    <collision><geometry><sphere radius="0.06"/></geometry></collision>
  </link>
  <link name="tool">
    <inertial><origin xyz="0 0 0" rpy="0 0 1.5707963"/><mass value="0.5"/>
      <inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><sphere radius="0.03"/></geometry></visual>
  </link>
  <link name="wheel">
    <inertial><mass value="0.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><mesh filename="wheel.stl" scale="1 1 1"/></geometry></visual>
    <collision><geometry><mesh filename="wheel.stl"/></geometry></collision>
  </link>
  <link name="sensor">
    <visual><geometry><box size="0.05 0.05 0.05"/></geometry></visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.2"/><axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" velocity="2" effort="10"/></joint>
  <joint name="slide" type="prismatic">
    <parent link="arm"/><child link="tool"/><origin xyz="0 0 0.4"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="0.1" velocity="0.5" effort="20"/></joint>
  <joint name="axle" type="continuous">
    <parent link="base"/><child link="wheel"/><origin xyz="0.2 0 -0.05"/><axis xyz="0 1 0"/></joint>
  <joint name="mount" type="fixed">
    <parent link="base"/><child link="sensor"/><origin xyz="0 0 0.25"/></joint>
</robot>
"""


@pytest.fixture
def urdf_file(tmp_path: Path) -> Path:
    write_mesh(tmp_path / "wheel.stl")
    p = tmp_path / "rover.urdf"
    p.write_text(URDF, encoding="utf-8")
    return p


# --- URDF integration ------------------------------------------------------------


def test_urdf_import_builds_valid_sadf(urdf_file: Path, tmp_path: Path) -> None:
    doc = importers.import_urdf(urdf_file, assets_dir=tmp_path / "assets", uri_prefix="assets/")
    validate_sadf(doc)  # belt-and-suspenders; build_sadf already validated
    a = doc.asset
    assert a.identity.id == "imported.rover"
    # massless 'sensor' has a frame but no body
    assert {b.name for b in a.bodies} == {"base", "arm", "tool", "wheel"}
    assert {f.name for f in a.frames} == {"base", "arm", "tool", "wheel", "sensor"}
    # base is the root frame (no parent); children carry a transform
    root = next(f for f in a.frames if f.name == "base")
    assert root.parent is None and a.root_frame == "base"


def test_urdf_joint_types_and_limits(urdf_file: Path, tmp_path: Path) -> None:
    a = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a").asset
    by_name = {j.name: j for j in a.joints}
    # Only joints between declared bodies survive; 'mount' (base -> massless 'sensor' link)
    # is dropped, since SADF joints connect bodies (Core referential-closure, RM-P1-CORE-05).
    assert set(by_name) == {"shoulder", "slide", "axle"}
    assert by_name["shoulder"].type is JointType.REVOLUTE
    assert by_name["slide"].type is JointType.PRISMATIC
    assert by_name["axle"].type is JointType.CONTINUOUS
    assert by_name["axle"].limits is None  # continuous: unbounded position
    shoulder = by_name["shoulder"].limits
    assert shoulder is not None and shoulder.position_rad is not None
    assert (shoulder.position_rad.min, shoulder.position_rad.max) == (-1.5, 1.5)
    assert shoulder.velocity_rad_s == pytest.approx(2.0)


def test_urdf_rotated_inertial_is_expressed_in_body_frame(urdf_file: Path, tmp_path: Path) -> None:
    a = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a").asset
    tool = next(b for b in a.bodies if b.name == "tool")
    # a 90° z-rotation swaps the ixx/iyy principal moments (1, 2) -> (2, 1)
    assert tool.inertia_kg_m2.ixx == pytest.approx(2.0, abs=1e-6)
    assert tool.inertia_kg_m2.iyy == pytest.approx(1.0, abs=1e-6)


def test_urdf_geometry_has_usd_and_gltf_with_collision_hull(
    urdf_file: Path, tmp_path: Path
) -> None:
    assets = tmp_path / "assets"
    a = importers.import_urdf(urdf_file, assets_dir=assets, uri_prefix="assets/").asset
    # 'base' declares only a visual -> a collision hull is still generated for it
    base_roles = {(g.role, g.format.value) for g in a.geometry if g.frame == "base"}
    assert (GeometryRole.VISUAL, "usd") in base_roles
    assert (GeometryRole.COLLISION, "gltf") in base_roles
    # every ref points at a written artifact
    for ref in a.geometry:
        assert (assets / Path(ref.uri).name).is_file()


def test_urdf_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ImportError_, match="not found"):
        importers.import_urdf(tmp_path / "nope.urdf", assets_dir=tmp_path)


def test_urdf_multiple_roots_rejected(tmp_path: Path) -> None:
    p = tmp_path / "two.urdf"
    p.write_text('<robot name="x"><link name="a"/><link name="b"/></robot>', encoding="utf-8")
    with pytest.raises(ImportError_, match="exactly one root"):
        importers.import_urdf(p, assets_dir=tmp_path)


def test_urdf_unsupported_joint_type_rejected(tmp_path: Path) -> None:
    p = tmp_path / "f.urdf"
    p.write_text(
        '<robot name="x"><link name="a"/><link name="b"/>'
        '<joint name="j" type="floating"><parent link="a"/><child link="b"/></joint></robot>',
        encoding="utf-8",
    )
    with pytest.raises(ImportError_, match="unsupported joint type"):
        importers.import_urdf(p, assets_dir=tmp_path)


# --- SDF integration -------------------------------------------------------------

SDF = """<sdf version="1.7"><world name="w"><model name="rover">
  <link name="base"><pose>0 0 0.1 0 0 0</pose>
    <inertial><mass>5</mass><inertia>
      <ixx>0.2</ixx><iyy>0.3</iyy><izz>0.4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
    <visual name="v"><geometry><box><size>0.4 0.3 0.2</size></box></geometry></visual>
    <collision name="c"><geometry>
      <cylinder><radius>0.1</radius><length>0.2</length></cylinder></geometry></collision>
  </link>
  <link name="wheel"><pose>0.2 0 0 0 0 0</pose>
    <inertial><mass>0.5</mass><inertia>
      <ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
    <visual name="v"><geometry>
      <mesh><uri>wheel.stl</uri><scale>1 1 1</scale></mesh></geometry></visual>
  </link>
  <joint name="axle" type="revolute"><parent>base</parent><child>wheel</child>
    <axis><xyz>0 1 0</xyz>
      <limit><lower>-1.5</lower><upper>1.5</upper><effort>10</effort><velocity>2</velocity></limit>
    </axis></joint>
</model></world></sdf>
"""


@pytest.fixture
def sdf_file(tmp_path: Path) -> Path:
    write_mesh(tmp_path / "wheel.stl")
    p = tmp_path / "rover.sdf"
    p.write_text(SDF, encoding="utf-8")
    return p


def test_sdf_import_builds_valid_sadf(sdf_file: Path, tmp_path: Path) -> None:
    a = importers.import_sdf(sdf_file, assets_dir=tmp_path / "a", uri_prefix="a/").asset
    assert a.identity.id == "imported.rover"
    # SDF link poses are relative to a synthetic model root frame
    assert a.root_frame == "rover"
    assert {f.parent for f in a.frames if f.name in {"base", "wheel"}} == {"rover"}
    axle = a.joints[0]
    assert axle.type is JointType.REVOLUTE and axle.limits is not None
    assert axle.limits.effort_nm == pytest.approx(10.0)


def test_sdf_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ImportError_, match="not found"):
        importers.import_sdf(tmp_path / "nope.sdf", assets_dir=tmp_path)


def test_sdf_no_model(tmp_path: Path) -> None:
    p = tmp_path / "empty.sdf"
    p.write_text('<sdf version="1.7"></sdf>', encoding="utf-8")
    with pytest.raises(ImportError_, match="no <model>"):
        importers.import_sdf(p, assets_dir=tmp_path)


def test_sdf_parse_error(tmp_path: Path) -> None:
    p = tmp_path / "broken.sdf"
    p.write_text("<sdf><model", encoding="utf-8")
    with pytest.raises(ImportError_, match="cannot parse"):
        importers.import_sdf(p, assets_dir=tmp_path)


def test_sdf_unsupported_geometry(tmp_path: Path) -> None:
    p = tmp_path / "p.sdf"
    p.write_text(
        '<sdf><model name="m"><link name="a"><visual name="v">'
        "<geometry><plane><normal>0 0 1</normal></plane></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )
    with pytest.raises(ImportError_, match="unsupported SDF geometry"):
        importers.import_sdf(p, assets_dir=tmp_path)


def test_sdf_joint_missing_parent(tmp_path: Path) -> None:
    p = tmp_path / "j.sdf"
    p.write_text(
        '<sdf><model name="m"><link name="a"/><link name="b"/>'
        '<joint name="j" type="fixed"><child>b</child></joint></model></sdf>',
        encoding="utf-8",
    )
    with pytest.raises(ImportError_, match="missing <parent>"):
        importers.import_sdf(p, assets_dir=tmp_path)


# --- dispatch --------------------------------------------------------------------


def test_import_description_infers_format(urdf_file: Path, sdf_file: Path, tmp_path: Path) -> None:
    assert importers.import_description(urdf_file, assets_dir=tmp_path / "u").asset.bodies
    assert importers.import_description(sdf_file, assets_dir=tmp_path / "s").asset.bodies


def test_import_description_explicit_format(urdf_file: Path, tmp_path: Path) -> None:
    doc = importers.import_description(urdf_file, assets_dir=tmp_path / "u", fmt="urdf")
    assert doc.asset.identity.id == "imported.rover"


def test_import_description_unknown_format(urdf_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ImportError_, match="unknown format"):
        importers.import_description(urdf_file, assets_dir=tmp_path, fmt="step")


def test_import_description_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "robot.xacro"
    p.write_text("<robot/>", encoding="utf-8")
    with pytest.raises(ImportError_, match="cannot infer format"):
        importers.import_description(p, assets_dir=tmp_path)


# --- math / helper branches ------------------------------------------------------


def test_rpy_to_quat_identity() -> None:
    q = _common.rpy_to_quat(0.0, 0.0, 0.0)
    assert (q.x, q.y, q.z, q.w) == pytest.approx((0.0, 0.0, 0.0, 1.0))


@pytest.mark.parametrize(
    "rot",
    [
        np.eye(3),  # trace > 0
        np.diag([1.0, -1.0, -1.0]),  # 180° about x  -> r[0,0] largest
        np.diag([-1.0, 1.0, -1.0]),  # 180° about y  -> r[1,1] largest
        np.diag([-1.0, -1.0, 1.0]),  # 180° about z  -> r[2,2] largest
    ],
)
def test_rotmat_to_quat_is_unit(rot: np.ndarray) -> None:
    q = _common._rotmat_to_quat(rot)
    assert math.isclose(q.x**2 + q.y**2 + q.z**2 + q.w**2, 1.0, abs_tol=1e-9)


def test_matrix_to_transform_identity_is_none() -> None:
    assert _common.matrix_to_transform(np.eye(4)) is None


def test_matrix_to_transform_carries_translation() -> None:
    m = np.eye(4)
    m[:3, 3] = (1.0, 2.0, 3.0)
    t = _common.matrix_to_transform(m)
    assert t is not None and (t.translation_m.x, t.translation_m.y, t.translation_m.z) == (
        1.0,
        2.0,
        3.0,
    )


def test_inertia_rotation_swaps_moments_under_90deg_z() -> None:
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    out = _common.inertia_in_body_frame(np.diag([1.0, 2.0, 3.0]), rz)
    assert (out.ixx, out.iyy, out.izz) == pytest.approx((2.0, 1.0, 3.0))


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("package://pkg/meshes/x.stl", "meshes/x.stl"),
        ("model://pkg/m/x.stl", "m/x.stl"),
        ("file://rel/x.stl", "rel/x.stl"),
        ("meshes/x.stl", "meshes/x.stl"),
    ],
)
def test_resolve_mesh_path_relative(uri: str, expected: str) -> None:
    base = Path("/base")
    assert _common.resolve_mesh_path(uri, base) == base / expected


def test_resolve_mesh_path_absolute() -> None:
    assert _common.resolve_mesh_path("/abs/x.stl", Path("/base")) == Path("/abs/x.stl")


def test_joint_type_maps_and_rejects() -> None:
    assert _common.joint_type("revolute") is JointType.REVOLUTE
    with pytest.raises(ImportError_, match="unsupported joint type"):
        _common.joint_type("ball")


def test_sdf_floats_arity_error() -> None:
    with pytest.raises(ImportError_, match="expected 3"):
        sdf._floats("1 2", 3)


def test_sdf_inertia_tensor_defaults_to_zero() -> None:
    assert np.allclose(sdf._inertia_tensor(None), np.zeros((3, 3)))


# --- CLI -------------------------------------------------------------------------








