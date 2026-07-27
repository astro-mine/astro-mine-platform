"""USD/glTF geometry handling (RM-P0-FLEET-02)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

from astro_mine.core.sadf.enums import FidelityTier, GeometryFormat, GeometryRole
from astro_mine.core.sadf.model import Inertia
from astro_mine.fleet import geometry
from astro_mine.fleet.geometry import GeometryError


def box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def read_glb_json(path: Path) -> dict:
    """The JSON chunk of a binary glTF, so a test can assert on the file rather than on a reload."""
    blob = path.read_bytes()
    json_length = struct.unpack("<I", blob[12:16])[0]
    return json.loads(blob[20 : 20 + json_length])


# --- load_mesh -------------------------------------------------------------------


def test_load_mesh_roundtrips(tmp_path: Path) -> None:
    p = tmp_path / "b.stl"
    box().export(p)
    assert geometry.load_mesh(p).faces.size > 0


def test_load_mesh_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GeometryError, match="not found"):
        geometry.load_mesh(tmp_path / "nope.stl")


def test_load_mesh_unreadable(tmp_path: Path) -> None:
    p = tmp_path / "bad.stl"
    p.write_text("definitely not a mesh", encoding="utf-8")
    with pytest.raises(GeometryError):
        geometry.load_mesh(p)


# --- normalize_mesh --------------------------------------------------------------


def test_normalize_applies_scale_and_transform() -> None:
    t = np.eye(4)
    t[:3, 3] = (10.0, 0.0, 0.0)  # translate +10 in x
    out = geometry.normalize_mesh(box(), scale=(2.0, 2.0, 2.0), transform=t)
    # unit box -> scaled to 2 m (extent 2), centroid shifted to x=10
    assert np.allclose(out.extents, (2.0, 2.0, 2.0))
    assert np.isclose(out.centroid[0], 10.0)


def test_normalize_is_a_noop_for_identity() -> None:
    out = geometry.normalize_mesh(box(), scale=(1.0, 1.0, 1.0), transform=None)
    assert np.allclose(out.extents, (1.0, 1.0, 1.0))


# --- convex_hull -----------------------------------------------------------------


def test_convex_hull_is_convex() -> None:
    hull = geometry.convex_hull(box())
    assert hull.is_convex and hull.faces.size > 0


# --- write_geometry --------------------------------------------------------------


def test_write_visual_emits_usd_and_gltf(tmp_path: Path) -> None:
    refs = geometry.write_geometry(
        box(),
        role=GeometryRole.VISUAL,
        stem="hull__visual0",
        frame="base",
        assets_dir=tmp_path,
        uri_prefix="assets/",
    )
    formats = {r.format for r in refs}
    assert formats == {GeometryFormat.USD, GeometryFormat.GLTF}
    assert all(r.role is GeometryRole.VISUAL and r.frame == "base" for r in refs)
    assert (tmp_path / "hull__visual0.glb").is_file()
    assert (tmp_path / "hull__visual0.usda").is_file()
    # Both canonical forms, at every LOD tier: lod 0 keeps the bare stem, the rest are suffixed.
    assert {r.uri for r in refs} == {
        "assets/hull__visual0.glb",
        "assets/hull__visual0.usda",
        "assets/hull__visual0.lod1.glb",
        "assets/hull__visual0.lod1.usda",
        "assets/hull__visual0.lod2.glb",
        "assets/hull__visual0.lod2.usda",
    }


# --- glTF is y-up, the spec's frame, not the body frame (issue #28) ---------------


def oriented_box() -> trimesh.Trimesh:
    """A box whose three extents differ, so a rotation cannot hide in a symmetry."""
    return trimesh.creation.box(extents=(1.0, 2.0, 3.0))  # x-forward, y-left, z-up


def write_visual(mesh: trimesh.Trimesh, tmp_path: Path) -> None:
    geometry.write_geometry(
        mesh,
        role=GeometryRole.VISUAL,
        stem="oriented",
        frame="base",
        assets_dir=tmp_path,
        uri_prefix="",
    )


def test_gltf_puts_the_body_up_axis_on_gltf_y(tmp_path: Path) -> None:
    """The asset stands up in a default glTF viewer.

    Authored z-up extent is 3.0; after the body→glTF rotation a consumer that honours the file's
    node transforms — as any conformant viewer does — must measure that 3.0 along glTF **+y**.
    Before issue #28 it measured along +z, and every viewer laid the asset on its side.
    """
    write_visual(oriented_box(), tmp_path)

    # `load_mesh` forces the scene to a single mesh, applying the graph's node transforms.
    loaded = geometry.load_mesh(tmp_path / "oriented.glb")
    assert np.allclose(loaded.extents, (2.0, 3.0, 1.0))  # (left, up, forward) -> (x, y, z)


def test_gltf_carries_the_rotation_on_the_node_not_the_vertices(tmp_path: Path) -> None:
    """The ``.glb`` and the ``.usda`` hold the same vertex arrays; only the node transform differs.

    Baking the rotation into the vertices would render identically but make the two canonical forms
    disagree numerically, and would express the mesh in a frame ``GeometryRef.frame`` does not name.
    """
    mesh = oriented_box()
    write_visual(mesh, tmp_path)
    gltf = read_glb_json(tmp_path / "oriented.glb")

    nodes_with_transform = [n for n in gltf["nodes"] if "matrix" in n or "rotation" in n]
    assert len(nodes_with_transform) == 1, "the rotation belongs on exactly one node"

    positions = next(a for a in gltf["accessors"] if a.get("type") == "VEC3" and "max" in a)
    assert np.allclose(positions["min"], mesh.bounds[0])
    assert np.allclose(positions["max"], mesh.bounds[1])


def test_gltf_rotation_is_proper_not_a_mirror(tmp_path: Path) -> None:
    """A reflection would render plausibly and silently flip a chiral asset's handedness."""
    write_visual(oriented_box(), tmp_path)
    gltf = read_glb_json(tmp_path / "oriented.glb")

    node = next(n for n in gltf["nodes"] if "matrix" in n)
    # glTF matrices are column-major; the upper-left 3x3 is the rotation.
    matrix = np.array(node["matrix"]).reshape(4, 4).T
    assert np.isclose(np.linalg.det(matrix[:3, :3]), 1.0)


def test_usd_stays_z_up_and_holds_the_same_vertices(tmp_path: Path) -> None:
    """The two forms may not drift apart: USD declares z-up, glTF rotates to y-up, same numbers."""
    from pxr import Usd, UsdGeom

    mesh = oriented_box()
    write_visual(mesh, tmp_path)

    stage = Usd.Stage.Open(str(tmp_path / "oriented.usda"))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0

    points = np.array(UsdGeom.Mesh.Get(stage, "/geometry").GetPointsAttr().Get())
    assert np.allclose(np.sort(points, axis=0), np.sort(mesh.vertices, axis=0))


def test_gltf_carries_normals_and_a_default_material(tmp_path: Path) -> None:
    """A primitive with no normals and no material renders as an unlit white blob everywhere."""
    write_visual(oriented_box(), tmp_path)
    gltf = read_glb_json(tmp_path / "oriented.glb")

    assert "NORMAL" in gltf["meshes"][0]["primitives"][0]["attributes"]

    materials = gltf.get("materials") or []
    assert len(materials) == 1
    assert materials[0]["name"] == "astro-mine-default"
    assert materials[0]["doubleSided"] is True


def test_default_material_never_overrides_the_source_mesh_appearance(tmp_path: Path) -> None:
    """A source mesh's colours are its author's intent; the default is only for bare geometry."""
    coloured = oriented_box()
    coloured.visual.face_colors = [200, 30, 30, 255]
    write_visual(coloured, tmp_path)
    gltf = read_glb_json(tmp_path / "oriented.glb")

    names = {m.get("name") for m in gltf.get("materials") or []}
    assert "astro-mine-default" not in names
    # trimesh carries face colours through as a vertex-colour attribute.
    assert "COLOR_0" in gltf["meshes"][0]["primitives"][0]["attributes"]


def test_glb_export_is_deterministic(tmp_path: Path) -> None:
    """The node matrix is exact 0.0/1.0, so re-exporting the same mesh yields the same bytes."""
    first, second = tmp_path / "a", tmp_path / "b"
    write_visual(oriented_box(), first)
    write_visual(oriented_box(), second)
    assert (first / "oriented.glb").read_bytes() == (second / "oriented.glb").read_bytes()


def test_write_collision_uses_the_hull(tmp_path: Path) -> None:
    # a concave L-shape: the written collision must be its (convex) hull
    concave = trimesh.util.concatenate(
        trimesh.creation.box(extents=(2, 1, 1)),
        trimesh.creation.box(extents=(1, 2, 1)),
    )
    refs = geometry.write_geometry(
        concave,
        role=GeometryRole.COLLISION,
        stem="x__collision0",
        frame="base",
        assets_dir=tmp_path,
        uri_prefix="",
    )
    assert all(r.role is GeometryRole.COLLISION for r in refs)
    written = geometry.load_mesh(tmp_path / "x__collision0.glb")
    assert written.is_convex


def test_collision_is_written_at_one_lod_only(tmp_path: Path) -> None:
    """A *cheaper* collision proxy would under-approximate the contact surface (a physics bug)."""
    refs = geometry.write_geometry(
        sphere(),
        role=GeometryRole.COLLISION,
        stem="x__collision1",
        frame="base",
        assets_dir=tmp_path,
        uri_prefix="",
    )
    assert {r.lod for r in refs} == {0}
    assert not list(tmp_path.glob("*.lod*"))


# --- LOD tiers (RM-P0-FLEET-02: "LOD, collision-hull, unit/frame normalization") --


def sphere() -> trimesh.Trimesh:
    """A mesh dense enough that decimation has something to remove (5 120 faces)."""
    return trimesh.creation.icosphere(subdivisions=4)


def test_decimate_meets_the_face_budget_of_each_tier() -> None:
    full = sphere()
    counts = [len(geometry.decimate(full, ratio).faces) for ratio in geometry.LOD_RATIOS]

    assert counts[0] == len(full.faces)  # lod 0 is the full-resolution mesh, untouched
    for lod, ratio in enumerate(geometry.LOD_RATIOS[1:], start=1):
        assert counts[lod] <= len(full.faces) * ratio
    assert counts[0] > counts[1] > counts[2]  # each tier is strictly cheaper than the last


def test_decimated_tiers_stay_closed_surfaces_that_bound_the_body() -> None:
    """A LOD tier a renderer cannot use — inside-out, or smaller than the asset — is not a tier."""
    full = sphere()
    for ratio in geometry.LOD_RATIOS[1:]:
        coarse = geometry.decimate(full, ratio)
        assert coarse.is_watertight
        # Clustering welds vertices inward, so a tier may shrink a little, but not collapse.
        assert np.allclose(coarse.extents, full.extents, atol=0.25 * float(max(full.extents)))


def test_decimation_is_deterministic() -> None:
    """Same mesh, same bytes — the reason LOD uses clustering, not float-ordered quadrics."""
    first = geometry.decimate(sphere(), 0.2)
    second = geometry.decimate(sphere(), 0.2)
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)


def test_an_already_coarse_mesh_yields_identical_tiers() -> None:
    """A 12-face box cannot be decimated into a closed surface; its tiers are the box itself."""
    assert [len(geometry.decimate(box(), r).faces) for r in geometry.LOD_RATIOS] == [12, 12, 12]


def test_write_geometry_emits_at_least_two_decimated_visual_tiers(tmp_path: Path) -> None:
    """The acceptance criterion: >= 2 visual LOD tiers per link, besides the collision hull."""
    refs = geometry.write_geometry(
        sphere(),
        role=GeometryRole.VISUAL,
        stem="link__visual0",
        frame="base",
        assets_dir=tmp_path,
        uri_prefix="",
    )
    tiers = sorted({r.lod for r in refs})
    assert len(tiers) >= 3 and tiers[0] == 0  # lod 0 + at least two decimated tiers

    # Each tier is a real, distinct artifact — not the same mesh under three names.
    faces = [
        len(geometry.load_geometry(tmp_path / f"link__visual0.lod{lod}.glb").faces)
        for lod in tiers[1:]
    ]
    assert len(geometry.load_geometry(tmp_path / "link__visual0.glb").faces) > faces[0] > faces[1]


@pytest.mark.parametrize(
    "tier,expected_lod",
    [
        (FidelityTier.MASSMODEL, 2),  # cheapest run -> coarsest mesh
        (FidelityTier.KINEMATIC, 1),
        (FidelityTier.ARTICULATED, 0),  # validation-grade run -> full resolution
        (FidelityTier.SURROGATE, 0),  # unmapped: fall back to full res, never guess coarse
    ],
)
def test_fidelity_tier_selects_a_lod(tier: FidelityTier, expected_lod: int) -> None:
    assert geometry.lod_for_tier(tier) == expected_lod


def test_select_geometry_picks_the_tier_a_fidelity_profile_asks_for(tmp_path: Path) -> None:
    refs = geometry.write_geometry(
        sphere(),
        role=GeometryRole.VISUAL,
        stem="v0",
        frame="base",
        assets_dir=tmp_path,
        uri_prefix="",
    ) + geometry.write_geometry(
        sphere(),
        role=GeometryRole.COLLISION,
        stem="c1",
        frame="base",
        assets_dir=tmp_path,
        uri_prefix="",
    )

    for tier, lod in [(FidelityTier.MASSMODEL, 2), (FidelityTier.ARTICULATED, 0)]:
        chosen = geometry.select_geometry(
            refs, fmt=GeometryFormat.GLTF, lod=geometry.lod_for_tier(tier)
        )
        visual = next(r for r in chosen if r.role is GeometryRole.VISUAL)
        collision = next(r for r in chosen if r.role is GeometryRole.COLLISION)
        assert visual.lod == lod
        # The collision proxy exists at lod 0 only, so a coarse tier must still resolve it —
        # a fidelity selection that left an asset with no collision geometry would be a trap.
        assert collision.lod == 0


# --- reading geometry back into the body frame (the exporters' side) --------------


def test_load_geometry_undoes_the_gltf_y_up_rotation(tmp_path: Path) -> None:
    """`load_geometry` is the exact inverse of `write_geometry`: what went in comes back out.

    `load_mesh` honours the glTF node transform and so returns *glTF-frame* vertices (that is what
    a viewer sees). An exporter re-materializing the mesh needs the **body** frame back, or every
    exported URDF would carry a rover lying on its side.
    """
    mesh = oriented_box()  # extents (1, 2, 3) in the body frame
    write_visual(mesh, tmp_path)

    assert np.allclose(geometry.load_mesh(tmp_path / "oriented.glb").extents, (2.0, 3.0, 1.0))
    assert np.allclose(geometry.load_geometry(tmp_path / "oriented.glb").extents, (1.0, 2.0, 3.0))
    assert np.allclose(geometry.load_geometry(tmp_path / "oriented.usda").extents, (1.0, 2.0, 3.0))


def test_load_geometry_reads_a_usd_stage(tmp_path: Path) -> None:
    mesh = oriented_box()
    write_visual(mesh, tmp_path)
    back = geometry.load_geometry(tmp_path / "oriented.usda")
    assert np.allclose(np.sort(back.vertices, axis=0), np.sort(mesh.vertices, axis=0))


def test_load_geometry_missing_usd(tmp_path: Path) -> None:
    with pytest.raises(GeometryError, match="not found"):
        geometry.load_geometry(tmp_path / "nope.usda")


def test_load_geometry_usd_without_a_mesh(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(tmp_path / "empty.usda"))
    UsdGeom.Xform.Define(stage, "/empty")
    stage.GetRootLayer().Save()
    with pytest.raises(GeometryError, match="holds no mesh"):
        geometry.load_geometry(tmp_path / "empty.usda")


def test_write_obj_round_trips_in_the_body_frame(tmp_path: Path) -> None:
    """OBJ is what a URDF/SDF consumer (MuJoCo, Gazebo, rviz) actually reads — and it is text,
    so the artifact is diffable and byte-reproducible."""
    mesh = oriented_box()
    geometry.write_obj(mesh, tmp_path / "m.obj")
    assert np.allclose(geometry.load_geometry(tmp_path / "m.obj").extents, mesh.extents)

    geometry.write_obj(mesh, tmp_path / "again.obj")
    assert (tmp_path / "m.obj").read_bytes() == (tmp_path / "again.obj").read_bytes()


# --- the inertia-equivalent proxy box (mesh-free assets) --------------------------


def test_mass_proxy_box_reproduces_the_bodys_mass_properties() -> None:
    """The proxy is not a cartoon: it is *the* uniform box with that mass and that inertia."""
    mass = 180.0
    inertia = Inertia(ixx=28.0, iyy=34.0, izz=22.0)
    proxy = geometry.mass_proxy_mesh(mass, inertia)

    # trimesh reports the inertia of a unit-density solid; scale it to the body's mass.
    scaled = np.asarray(proxy.moment_inertia) * (mass / proxy.volume)
    assert np.allclose(np.diag(scaled), (28.0, 34.0, 22.0), atol=1e-9)
    assert np.allclose(scaled - np.diag(np.diag(scaled)), 0.0, atol=1e-9)


def test_mass_proxy_box_honours_products_of_inertia() -> None:
    """A tensor with off-diagonal terms yields a box rotated onto the body's principal axes."""
    proxy = geometry.mass_proxy_mesh(10.0, Inertia(ixx=2.0, iyy=3.0, izz=4.0, ixy=0.5))
    scaled = np.asarray(proxy.moment_inertia) * (10.0 / proxy.volume)
    assert np.allclose(scaled, [[2.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 4.0]], atol=1e-9)


def test_mass_proxy_rejects_a_tensor_no_real_body_has() -> None:
    # izz > ixx + iyy violates the inertia triangle inequality
    with pytest.raises(GeometryError, match="triangle inequality"):
        geometry.mass_proxy_mesh(1.0, Inertia(ixx=1.0, iyy=1.0, izz=9.0))


def test_mass_proxy_rejects_a_massless_body() -> None:
    with pytest.raises(GeometryError, match="positive mass"):
        geometry.mass_proxy_mesh(0.0, Inertia(ixx=1.0, iyy=1.0, izz=1.0))
