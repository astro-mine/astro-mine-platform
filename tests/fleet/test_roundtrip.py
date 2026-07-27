"""Round-trip tests (fleet.md §10; RM-P0-FLEET-02).

  *"Round-trip tests: URDF→SADF→URDF and USD→SADF→USD preserve invariants (kinematic tree,
  masses, frames) within tolerance."*  — fleet.md §10

Each converter is exercised as a **closed loop** and judged on two things:

1. **the invariants** — the kinematic tree, link masses, centres of mass, inertia tensors, frame
   poses, joint types/axes/limits — must survive, within a *stated* tolerance;
2. **the fixed point** — a second lap must change nothing. A converter that drifts a little on
   every lap is not lossy, it is *wrong*, and it would erode an asset that ever crossed the
   boundary twice.

Where a format genuinely cannot hold something, the loss is asserted **explicitly** (below, and in
``exporters.LOSS_CONTRACT``) — a documented lossy edge, not a silent one.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yourdfpy

from astro_mine.core.sadf import SadfDocument
from astro_mine.core.sadf.enums import JointType
from astro_mine.fleet import exporters, importers
from astro_mine.fleet._core import canonical_json

# --- tolerances ------------------------------------------------------------------

#: **URDF/SDF round trip: exact, to a stated budget.** Both are text formats holding IEEE-754
#: doubles, and the exporter writes each float in its shortest round-tripping form, so masses,
#: inertias, and translations come back bit-identical.
XML_TOL = 1e-9

#: **Rotations cost one ulp.** URDF and SDF spell orientation as fixed-axis roll-pitch-yaw; SADF
#: stores a quaternion. quat→rpy→quat is not bit-exact, so a *rotated* frame lands ~1e-16 away.
#: The error is bounded and does **not** accumulate across laps (see the drift test), which is the
#: property that actually protects a long-lived asset.
ROTATION_ULP = 1e-12

#: **USD round trip: float32.** ``UsdPhysics`` declares mass, centre of mass, and inertia as
#: 32-bit floats (``physics:mass`` is a ``float``, ``physics:diagonalInertia`` a ``float3``), while
#: SADF carries doubles. Widening them with a private double attribute would be exactly the
#: side-channel conventions.md §1 forbids — and Isaac would not read it — so the narrowing is
#: accepted and bounded here instead. Frame poses are authored double-precision and are exact.
USD_MASS_RTOL = 1e-6

#: The full URDF fixture: every joint type, every geometry primitive, a collision fallback, a
#: **rotated** inertial frame (so the tensor rotation is exercised), and a **massless** link.
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
    <visual><geometry><mesh filename="wheel.stl"/></geometry></visual>
  </link>
  <link name="sensor">
    <visual><geometry><box size="0.05 0.05 0.05"/></geometry></visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.2" rpy="0 0 0.3"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" velocity="2" effort="10"/></joint>
  <joint name="slide" type="prismatic">
    <parent link="arm"/><child link="tool"/><origin xyz="0 0 0.4"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="0.1" velocity="0.5" effort="20"/></joint>
  <joint name="axle" type="continuous">
    <parent link="base"/><child link="wheel"/><origin xyz="0.2 0 -0.05"/>
    <axis xyz="0 1 0"/></joint>
  <joint name="mount" type="fixed">
    <parent link="base"/><child link="sensor"/><origin xyz="0 0 0.25"/></joint>
</robot>
"""


#: The same robot with every joint origin axis-aligned. A rotation is the only thing in the loop
#: that is not bit-exact (see :data:`ROTATION_ULP`), so this is the fixture the *byte*-level fixed
#: point is asserted on — and it is the shape every Phase-0 reference asset actually has.
URDF_NO_ROTATION = URDF.replace(' rpy="0 0 0.3"', "")


@pytest.fixture
def urdf_file(tmp_path: Path) -> Path:
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(tmp_path / "wheel.stl")
    path = tmp_path / "rover.urdf"
    path.write_text(URDF, encoding="utf-8")
    return path


# --- reading a URDF back as physics, not as XML ----------------------------------


def links_of(path: Path) -> dict[str, yourdfpy.Link]:
    robot = yourdfpy.URDF.load(str(path), load_meshes=False, build_scene_graph=False).robot
    return {link.name: link for link in robot.links}


def joints_of(path: Path) -> dict[str, yourdfpy.Joint]:
    robot = yourdfpy.URDF.load(str(path), load_meshes=False, build_scene_graph=False).robot
    return {joint.name: joint for joint in robot.joints}


def inertia_about_com(link: yourdfpy.Link) -> np.ndarray:
    """A URDF link's inertia tensor in **link-frame axes** — the frame-independent physics.

    URDF states the tensor in the frame its ``<inertial><origin>`` rotates to, so two URDFs can
    describe the same body with different numbers. Rotating into link axes (``I = R·I'·Rᵀ``) is
    what makes them comparable — and it is the whole point of the invariant: the *physics* must
    round-trip, not the spelling.
    """
    origin = link.inertial.origin if link.inertial.origin is not None else np.eye(4)
    rotation = np.asarray(origin)[:3, :3]
    return rotation @ np.asarray(link.inertial.inertia, dtype=float) @ rotation.T


def origin_of(joint: yourdfpy.Joint) -> np.ndarray:
    return np.eye(4) if joint.origin is None else np.asarray(joint.origin, dtype=float)


# --- URDF -> SADF -> URDF (the acceptance criterion) ------------------------------


def test_urdf_roundtrip_preserves_the_kinematic_tree(urdf_file: Path, tmp_path: Path) -> None:
    """The tree: same links, same joints, same parent/child edges, same types and axes."""
    doc = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a", uri_prefix="a/")
    out = exporters.export_urdf(doc, tmp_path / "out" / "rover.urdf", base_dir=tmp_path).path

    before, after = links_of(urdf_file), links_of(out)
    assert set(before) == set(after)

    source, result = joints_of(urdf_file), joints_of(out)
    # `mount` (base -> the massless `sensor` link) is dropped on import — SADF joints connect
    # bodies — and re-synthesized on export from the frame parenthood that carried it. Same edge,
    # same origin, new name: the physics is identical, the label is not.
    assert set(source) - {"mount"} <= set(result)
    assert "sensor__fixed" in result

    for name, original in source.items():
        echo = result[name if name != "mount" else "sensor__fixed"]
        assert (echo.parent, echo.child) == (original.parent, original.child)
        assert echo.type == original.type
        assert np.allclose(origin_of(echo), origin_of(original), atol=XML_TOL)
        if original.axis is not None:
            assert np.allclose(echo.axis, original.axis, atol=XML_TOL)


def test_urdf_roundtrip_preserves_masses_and_inertias(urdf_file: Path, tmp_path: Path) -> None:
    """Masses, centres of mass, and the *physical* inertia tensor — the fleet.md §10 invariant."""
    doc = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a", uri_prefix="a/")
    out = exporters.export_urdf(doc, tmp_path / "out" / "rover.urdf", base_dir=tmp_path).path

    before, after = links_of(urdf_file), links_of(out)
    for name, original in before.items():
        if original.inertial is None:
            assert after[name].inertial is None  # the massless link stays massless
            continue

        echo = after[name]
        assert echo.inertial.mass == pytest.approx(original.inertial.mass, abs=XML_TOL)
        assert np.allclose(
            np.asarray(origin_of_inertial(echo))[:3, 3],
            np.asarray(origin_of_inertial(original))[:3, 3],
            atol=XML_TOL,
        )
        assert np.allclose(inertia_about_com(echo), inertia_about_com(original), atol=XML_TOL)


def origin_of_inertial(link: yourdfpy.Link) -> np.ndarray:
    origin = link.inertial.origin
    return np.eye(4) if origin is None else np.asarray(origin, dtype=float)


def test_urdf_roundtrip_normalizes_a_rotated_inertial_frame(
    urdf_file: Path, tmp_path: Path
) -> None:
    """The documented lossy edge: the tensor's *spelling* is normalized, its physics is not.

    The `tool` link declares its inertia in a frame rotated 90° about z. SADF stores tensors in the
    body frame, so the rotation is baked in (``I' = R·I·Rᵀ``) and the re-export writes ``rpy=0``
    with the equivalent tensor. Same body; different XML. Asserting this keeps it a *decision*
    rather than a surprise.
    """
    doc = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a", uri_prefix="a/")
    out = exporters.export_urdf(doc, tmp_path / "out" / "rover.urdf", base_dir=tmp_path).path

    original, echo = links_of(urdf_file)["tool"], links_of(out)["tool"]
    source_rpy = np.asarray(origin_of_inertial(original))[:3, :3]
    assert not np.allclose(source_rpy, np.eye(3))  # the source really is rotated
    assert np.allclose(np.asarray(origin_of_inertial(echo))[:3, :3], np.eye(3))  # the echo is not

    # ...and the physics is untouched: the 90° z-rotation swapped ixx/iyy (1, 2) -> (2, 1)
    assert np.allclose(inertia_about_com(echo), inertia_about_com(original), atol=1e-6)
    assert echo.inertial.inertia[0][0] == pytest.approx(2.0, abs=1e-6)


def test_urdf_roundtrip_does_not_drift_over_many_laps(urdf_file: Path, tmp_path: Path) -> None:
    """The property that actually matters: the loop **converges**, it does not erode.

    A URDF spells orientation as fixed-axis roll-pitch-yaw; SADF stores a quaternion. The
    conversion quat→rpy→quat is not bit-exact, so an asset carrying a rotated joint (this one has
    ``rpy="0 0 0.3"``) comes back one ulp away — and a converter that lost *another* ulp on every
    lap would quietly grind an asset down over a long-lived pipeline.

    It does not: the per-lap difference is ~1e-16 and **constant**, so the loop is a fixed point in
    every sense that matters. Four laps, and the error never grows.
    """
    doc = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a0", uri_prefix="a0/")
    laps = []
    for lap in range(4):
        out = exporters.export_urdf(doc, tmp_path / f"lap{lap}" / "r.urdf", base_dir=tmp_path).path
        laps.append(out)
        doc = importers.import_urdf(out, assets_dir=tmp_path / f"a{lap}", uri_prefix="a0/")

    deltas = []
    for previous, current in itertools.pairwise(laps):
        before, after = joints_of(previous), joints_of(current)
        assert set(before) == set(after)  # the tree itself is a strict fixed point
        deltas.append(max(np.abs(origin_of(after[n]) - origin_of(before[n])).max() for n in before))

    assert max(deltas) <= ROTATION_ULP  # every lap lands within one ulp
    assert deltas[-1] <= deltas[0]  # ...and the error does not accumulate


def test_urdf_roundtrip_is_byte_exact_once_the_identity_settles(tmp_path: Path) -> None:
    """With no rotation to re-spell, the loop is exact **to the byte** from the second lap on.

    Two things settle on the first lap and never move again. The **identity**: a URDF carries one
    robot name, so a re-import mints an id from it — ``imported.<name>`` — and the exporter writes
    *that* id as the robot name next time. Because the prefix is idempotent, lap two names the robot
    exactly what lap one did. The **geometry**: SADF's meshes are re-materialized as OBJ, and an OBJ
    re-read and re-written is byte-stable. From there the loop is a literal identity function.

    Every Phase-0 reference asset has identity-rotation frames, so this is the case the library
    actually hits (conventions.md §11: same inputs => same bytes).
    """
    from astro_mine.fleet import library

    doc = library.load_reference("prospecting_rover")
    lap0 = exporters.export_urdf(doc, tmp_path / "l0" / "r.urdf").path
    settled = importers.import_urdf(lap0, assets_dir=tmp_path / "a0")

    lap1 = exporters.export_urdf(settled, tmp_path / "l1" / "r.urdf").path
    again = importers.import_urdf(lap1, assets_dir=tmp_path / "a1")
    lap2 = exporters.export_urdf(again, tmp_path / "l2" / "r.urdf").path

    assert lap1.read_text(encoding="utf-8") == lap2.read_text(encoding="utf-8")


def test_urdf_roundtrip_settles_the_sadf_after_one_lap(tmp_path: Path) -> None:
    """SADF → URDF → SADF changes only the *display name*, and nothing else at all.

    A URDF carries one robot name; SADF carries an identity block (id, display name, version, kind,
    description, labels). The id survives — the importer's namespace prefix is idempotent, which is
    what makes the loop close — but the display name is re-minted from the robot name. Everything
    else, down to the geometry refs and their LOD tiers, is byte-identical in canonical JSON.

    Uses a rotation-free asset so the comparison can be exact; the rotated case is covered by
    ``test_urdf_roundtrip_does_not_drift_over_many_laps``.
    """
    source = tmp_path / "src"
    source.mkdir()
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(source / "wheel.stl")
    (source / "r.urdf").write_text(URDF_NO_ROTATION, encoding="utf-8")

    first = importers.import_urdf(source / "r.urdf", assets_dir=source / "a1", uri_prefix="a1/")
    urdf_b = exporters.export_urdf(first, tmp_path / "b" / "r.urdf", base_dir=source).path
    second = importers.import_urdf(urdf_b, assets_dir=source / "a2", uri_prefix="a1/")

    assert second.asset.identity.id == first.asset.identity.id  # the id is a fixed point
    assert second.asset.identity.name != first.asset.identity.name  # the display name is not

    normalized = second.model_copy(deep=True)
    normalized.asset.identity.name = first.asset.identity.name
    assert canonical_json(normalized) == canonical_json(first)


def test_export_is_byte_deterministic(urdf_file: Path, tmp_path: Path) -> None:
    """Same asset in, same bytes out — for every format (conventions.md §11 determinism gate)."""
    doc = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a", uri_prefix="a/")
    for fmt in exporters.FORMATS:
        first = exporters.export_description(
            doc, tmp_path / "x" / f"r.{fmt}", fmt=fmt, base_dir=tmp_path
        ).path
        second = exporters.export_description(
            doc, tmp_path / "y" / f"r.{fmt}", fmt=fmt, base_dir=tmp_path
        ).path
        assert first.read_bytes() == second.read_bytes(), f"{fmt} export is not deterministic"


def test_urdf_roundtrip_preserves_geometry_and_its_lod_ladder(
    urdf_file: Path, tmp_path: Path
) -> None:
    """Geometry survives as *meshes*, and the LOD ladder is regenerated identically."""
    first = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a1", uri_prefix="a1/")
    urdf_b = exporters.export_urdf(first, tmp_path / "b" / "r.urdf", base_dir=tmp_path).path
    second = importers.import_urdf(urdf_b, assets_dir=tmp_path / "a2", uri_prefix="a1/")

    def refs(doc: SadfDocument) -> set[tuple[str, str, str, int]]:
        return {(r.frame, r.role.value, r.format.value, r.lod) for r in doc.asset.geometry}

    assert refs(second) == refs(first)
    assert {r.lod for r in first.asset.geometry} == {0, 1, 2}  # the ladder is there and survives

    # and the mesh a link actually carries is still the same shape (the base box: 0.4 x 0.3 x 0.2)
    for assets in (tmp_path / "a1", tmp_path / "a2"):
        mesh = trimesh.load(assets / "base__visual0.glb", force="mesh")
        assert np.allclose(sorted(mesh.extents), sorted((0.2, 0.3, 0.4)), atol=1e-6)


# --- USD -> SADF -> USD (fleet.md §10's second direction) -------------------------


def test_usd_roundtrip_preserves_the_tree_masses_and_frames(
    urdf_file: Path, tmp_path: Path
) -> None:
    """The same three invariants, through USD — the format Sim and Studio actually consume."""
    first = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a1", uri_prefix="a1/")
    stage = exporters.export_usd(first, tmp_path / "rover.usda", base_dir=tmp_path).path
    second = importers.import_usd(stage, assets_dir=tmp_path / "a2", uri_prefix="a1/")

    # frames: the tree nests natively in USD, so the hierarchy comes back exactly as it went in
    def frames(doc: SadfDocument) -> dict[str, str | None]:
        return {f.name: f.parent for f in doc.asset.frames}

    assert frames(second) == frames(first)
    assert second.asset.root_frame == first.asset.root_frame

    # masses + inertias, to float32 (see USD_MASS_RTOL)
    before = {b.name: b for b in first.asset.bodies}
    after = {b.name: b for b in second.asset.bodies}
    assert set(before) == set(after)
    for name, body in before.items():
        echo = after[name]
        assert echo.mass_kg == pytest.approx(body.mass_kg, rel=USD_MASS_RTOL)
        for axis in ("x", "y", "z"):
            assert getattr(echo.center_of_mass_m, axis) == pytest.approx(
                getattr(body.center_of_mass_m, axis), rel=USD_MASS_RTOL, abs=1e-7
            )
        for component in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
            assert getattr(echo.inertia_kg_m2, component) == pytest.approx(
                getattr(body.inertia_kg_m2, component), rel=USD_MASS_RTOL, abs=1e-7
            )


def test_usd_roundtrip_preserves_joints_including_the_continuous_marker(
    urdf_file: Path, tmp_path: Path
) -> None:
    """USD has no unbounded-revolute *type*; the distinction is recorded, so it must come back."""
    first = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a1", uri_prefix="a1/")
    stage = exporters.export_usd(first, tmp_path / "rover.usda", base_dir=tmp_path).path
    second = importers.import_usd(stage, assets_dir=tmp_path / "a2", uri_prefix="a1/")

    before = {j.name: j for j in first.asset.joints}
    after = {j.name: j for j in second.asset.joints}
    assert set(before) == set(after)

    for name, joint in before.items():
        echo = after[name]
        assert echo.type is joint.type
        assert (echo.parent_body, echo.child_body) == (joint.parent_body, joint.child_body)
        if joint.axis is not None:
            assert echo.axis is not None
            for axis in ("x", "y", "z"):
                assert getattr(echo.axis, axis) == pytest.approx(
                    getattr(joint.axis, axis), abs=1e-6
                )
        if joint.limits is not None and joint.limits.position_rad is not None:
            assert echo.limits is not None and echo.limits.position_rad is not None
            assert echo.limits.position_rad.min == pytest.approx(
                joint.limits.position_rad.min, abs=1e-5
            )

    # `axle` is continuous — a revolute joint with no bound. USD spells that the same way it spells
    # an unbounded revolute, so without the explicit marker it would come back as REVOLUTE.
    assert after["axle"].type is JointType.CONTINUOUS


def test_usd_roundtrip_loses_the_effort_and_velocity_limits(
    urdf_file: Path, tmp_path: Path
) -> None:
    """The documented lossy edge: a UsdPhysics joint holds position bounds and nothing else."""
    first = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a1", uri_prefix="a1/")
    result = exporters.export_usd(first, tmp_path / "rover.usda", base_dir=tmp_path)
    second = importers.import_usd(result.path, assets_dir=tmp_path / "a2", uri_prefix="a1/")

    assert "usd.drive_limits_dropped" in {loss.rule for loss in result.losses}
    shoulder_in = next(j for j in first.asset.joints if j.name == "shoulder")
    shoulder_out = next(j for j in second.asset.joints if j.name == "shoulder")

    assert shoulder_in.limits.effort_nm == pytest.approx(10.0)
    assert shoulder_out.limits.effort_nm is None  # gone, and said so
    assert shoulder_out.limits.position_rad is not None  # the bound survives


def test_usd_roundtrip_is_a_fixed_point(urdf_file: Path, tmp_path: Path) -> None:
    """A second lap through USD changes nothing — byte for byte."""
    first = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a1", uri_prefix="a1/")
    usd_a = exporters.export_usd(first, tmp_path / "a.usda", base_dir=tmp_path).path

    second = importers.import_usd(usd_a, assets_dir=tmp_path / "a2", uri_prefix="a1/")
    usd_b = exporters.export_usd(second, tmp_path / "b.usda", base_dir=tmp_path).path

    assert usd_a.read_text(encoding="utf-8") == usd_b.read_text(encoding="utf-8")


# --- SDF -> SADF -> SDF (the third converter; §11's "bidirectional URDF/SDF") ------


def test_sdf_roundtrip_preserves_masses_and_joints(urdf_file: Path, tmp_path: Path) -> None:
    first = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a1", uri_prefix="a1/")
    sdf = exporters.export_sdf(first, tmp_path / "rover.sdf", base_dir=tmp_path).path
    second = importers.import_sdf(sdf, assets_dir=tmp_path / "a2", uri_prefix="a1/")

    before = {b.name: b for b in first.asset.bodies}
    after = {b.name: b for b in second.asset.bodies}
    assert set(before) == set(after)
    for name, body in before.items():
        assert after[name].mass_kg == pytest.approx(body.mass_kg, abs=XML_TOL)
        assert after[name].inertia_kg_m2.ixx == pytest.approx(body.inertia_kg_m2.ixx, abs=XML_TOL)

    assert {j.name for j in second.asset.joints} == {j.name for j in first.asset.joints}


def test_sdf_roundtrip_flattens_the_frame_tree_once_then_settles(
    urdf_file: Path, tmp_path: Path
) -> None:
    """The documented SDF edge: the hierarchy flattens on the first lap, and never again.

    SDF ≤1.6 reads a link ``<pose>`` in the *model* frame, so a re-import parents every frame
    directly to the model root. World poses are untouched — the robot is in exactly the same shape
    — but ``arm`` is no longer a *child* of ``base``. Once flattened, it stays flattened: the
    second lap is a fixed point, so the tree does not keep collapsing.
    """
    first = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a1", uri_prefix="a1/")
    assert {f.parent for f in first.asset.frames if f.name == "tool"} == {"arm"}  # two deep

    sdf_a = exporters.export_sdf(
        first, tmp_path / "a.sdf", base_dir=tmp_path, assets_dir=tmp_path / "m"
    ).path
    second = importers.import_sdf(sdf_a, assets_dir=tmp_path / "a2", uri_prefix="a1/")

    root = second.asset.root_frame
    assert {f.parent for f in second.asset.frames if f.name != root} == {root}  # all flat now

    # the world pose survives the flattening: `tool` was base(0) -> arm(z .2, yaw .3) -> tool(z .4)
    tool = next(f for f in second.asset.frames if f.name == "tool")
    assert tool.transform.translation_m.z == pytest.approx(0.6, abs=XML_TOL)

    # ...and once flattened it stays flattened: a further lap moves nothing beyond the one-ulp
    # rpy<->quat noise, so the tree does not keep collapsing.
    sdf_b = exporters.export_sdf(
        second, tmp_path / "b.sdf", base_dir=tmp_path, assets_dir=tmp_path / "m"
    ).path
    third = importers.import_sdf(sdf_b, assets_dir=tmp_path / "a3", uri_prefix="a1/")

    assert {f.name: f.parent for f in third.asset.frames} == {
        f.name: f.parent for f in second.asset.frames
    }
    before = {link.name: link.world for link in exporters.realize(second).links}
    after = {link.name: link.world for link in exporters.realize(third).links}
    for name, pose in before.items():
        assert np.allclose(after[name], pose, atol=ROTATION_ULP), f"{name} drifted on lap 2"


def test_sdf_roundtrip_keeps_the_robot_in_one_piece(urdf_file: Path, tmp_path: Path) -> None:
    """Flattening the *hierarchy* must not move a single part: same world poses, same rotations."""
    first = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a1", uri_prefix="a1/")
    sdf = exporters.export_sdf(first, tmp_path / "a.sdf", base_dir=tmp_path).path
    second = importers.import_sdf(sdf, assets_dir=tmp_path / "a2", uri_prefix="a1/")

    before = {link.name: link.world for link in exporters.realize(first).links}
    after = {link.name: link.world for link in exporters.realize(second).links}
    for name, pose in before.items():
        assert np.allclose(after[name], pose, atol=XML_TOL), f"{name} moved"


# --- the library assets all survive every converter -------------------------------


@pytest.mark.parametrize("name", ["prospecting_rover", "excavator", "relay_orbiter", "isru_plant"])
@pytest.mark.parametrize("fmt", exporters.FORMATS)
def test_every_reference_asset_exports_and_reimports(name: str, fmt: str, tmp_path: Path) -> None:
    """The anchor roster is the contract. Each asset must survive each converter, both ways."""
    from astro_mine.fleet import library

    doc = library.load_reference(name)
    out = tmp_path / f"{name}.{fmt}"
    exporters.export_description(doc, out, fmt=fmt)

    back = importers.import_description(out, assets_dir=tmp_path / f"{name}_{fmt}")
    # masses are the invariant that must hold for every asset through every format
    tolerance = USD_MASS_RTOL if fmt == "usd" else XML_TOL
    assert sum(b.mass_kg for b in back.asset.bodies) == pytest.approx(
        sum(b.mass_kg for b in doc.asset.bodies), rel=tolerance
    )
    assert {j.name for j in back.asset.joints} == {j.name for j in doc.asset.joints}


def test_a_non_basis_joint_axis_survives_usd(urdf_file: Path, tmp_path: Path) -> None:
    """UsdPhysics only names X/Y/Z. An arbitrary axis rides on the joint frame's rotation, and the
    reader has to invert that exactly — otherwise every non-axis-aligned joint quietly re-aims."""
    doc = importers.import_urdf(urdf_file, assets_dir=tmp_path / "a1", uri_prefix="a1/")
    axis = next(j for j in doc.asset.joints if j.name == "shoulder").axis
    axis.x, axis.y, axis.z = 1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)

    stage = exporters.export_usd(doc, tmp_path / "r.usda", base_dir=tmp_path).path
    back = importers.import_usd(stage, assets_dir=tmp_path / "a2", uri_prefix="a1/")

    echo = next(j for j in back.asset.joints if j.name == "shoulder").axis
    assert (echo.x, echo.y, echo.z) == pytest.approx(
        (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)), abs=1e-6
    )
