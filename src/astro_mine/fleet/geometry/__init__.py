"""USD/glTF geometry handling (RM-P0-FLEET-02).

The engine-neutral mesh layer shared by the importers and the exporters: load a source
mesh, **normalize** it (bake unit scale + the source's local pose so geometry is expressed
in the body frame), generate a **convex collision hull** and **visual LOD tiers**, and write
the two canonical interchange forms — **USD** (preferred for [Sim]) and **glTF/GLB**
(web/View), per fleet.md §11 — returning the Core ``GeometryRef``s that point at them.

It owns no SADF schema (that is Core's) and no physics (that is Sim's); it only turns
meshes into normalized, content-addressable geometry artifacts and the refs to them.

**Frames.** A mesh is authored in the SADF body frame (x-forward, y-left, z-up). USD says so
explicitly (``SetStageUpAxis(z)``); glTF cannot, because glTF *defines* its own axes as +y up,
+z forward. The glTF writer therefore carries the body→glTF rotation on the mesh node, leaving
vertex data identical to the USD's. See :data:`_BODY_TO_GLTF`. :func:`load_geometry` is the
inverse: it reads any written artifact **back into the body frame**, so a geometry ref can be
re-materialized by the exporters without an axis flip.

**LOD.** ``write_geometry`` emits one visual artifact per tier in :data:`LOD_RATIOS`
(fleet.md §3 "LOD/collision-mesh handling"; RM-P0-FLEET-02), each carried on its own
``GeometryRef.lod``. A consumer dials the tier it needs; :func:`lod_for_tier` maps a SADF
fidelity tier onto one, so a coarse ``massmodel`` run never pays for the finest mesh
(fleet.md §8 "Fleet declares the tiers, Sim chooses them").

Backlog: RM-P0-FLEET-02 -- https://github.com/astro-mine/astro-mine-fleet/issues/2
Fixes: https://github.com/astro-mine/astro-mine-fleet/issues/28, /issues/31
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray

from astro_mine.core.sadf.enums import FidelityTier, GeometryFormat, GeometryRole
from astro_mine.core.sadf.model import GeometryRef, Inertia

__all__ = [
    "LOD_RATIOS",
    "GeometryError",
    "convex_hull",
    "decimate",
    "gltf_node_transform",
    "load_geometry",
    "load_mesh",
    "lod_for_tier",
    "mass_proxy_mesh",
    "normalize_mesh",
    "select_geometry",
    "write_geometry",
    "write_obj",
]


def gltf_node_transform() -> NDArray[np.float64]:
    """The body→glTF rotation a glTF writer carries on its root node (see :data:`_BODY_TO_GLTF`).

    Exposed so a *composed* glTF (an asset preview, not a single mesh) rotates into the spec's
    y-up frame the same way :func:`write_geometry` does — one convention, one place.
    """
    return _BODY_TO_GLTF.copy()


class GeometryError(Exception):
    """A mesh could not be loaded, normalized, or exported."""


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file into a single :class:`trimesh.Trimesh`.

    A multi-geometry source (e.g. a scene) is concatenated into one mesh. Raises
    :class:`GeometryError` if the file is missing or contains no triangles.
    """
    p = Path(path)
    if not p.is_file():
        raise GeometryError(f"mesh file not found: {p}")
    try:
        loaded = trimesh.load(p, force="mesh", process=False)
    except Exception as exc:  # trimesh raises a variety of loader-specific errors
        raise GeometryError(f"cannot load mesh {p}: {exc}") from exc
    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces.size == 0:
        raise GeometryError(f"mesh {p} has no triangulated geometry")
    return loaded


def normalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    scale: tuple[float, float, float] | None = None,
    transform: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` in the body frame: apply per-axis ``scale`` (SI metres),
    then the 4x4 ``transform`` (the source's local visual/collision pose). This is the
    unit/frame normalization that lets a body's geometry resolve in one consistent frame.
    """
    out = mesh.copy()
    if scale is not None and tuple(scale) != (1.0, 1.0, 1.0):
        out.apply_scale(np.asarray(scale, dtype=float))
    if transform is not None:
        out.apply_transform(np.asarray(transform, dtype=float))
    return out


def convex_hull(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """The convex hull of ``mesh`` — a usable, cheap collision proxy (fleet.md §3)."""
    hull = mesh.convex_hull
    if not isinstance(hull, trimesh.Trimesh) or hull.faces.size == 0:
        raise GeometryError("convex-hull generation produced an empty mesh")
    return hull


# --- LOD tiers -------------------------------------------------------------------

#: Target face-count fraction per visual LOD tier, indexed by ``GeometryRef.lod``:
#: ``lod=0`` is the full-resolution mesh, higher levels are progressively cheaper
#: (fleet.md §3/§4 "decimation, LOD, collision-hull generation"). Three tiers, so every
#: link carries **at least two** decimated visual tiers besides the full-resolution one.
LOD_RATIOS: tuple[float, ...] = (1.0, 0.5, 0.2)

#: A mesh at or below this face count is already as coarse as a closed surface usefully
#: gets; decimating it further collapses it. Such a mesh's LOD tiers are *identical* to
#: its full-resolution form (an explicit, documented degenerate case, not an error).
_MIN_FACES = 12

#: Bounds of the vertex-clustering grid search (see :func:`decimate`).
_MIN_CELLS, _MAX_CELLS = 2, 256

#: How a SADF fidelity tier picks a visual LOD tier: the finer the physics, the finer the
#: mesh. Fleet *declares* the tiers, Sim/Studio *choose* them (conventions.md §8).
_LOD_BY_TIER: dict[FidelityTier, int] = {
    FidelityTier.MASSMODEL: 2,
    FidelityTier.KINEMATIC: 1,
    FidelityTier.ARTICULATED: 0,
}


def lod_for_tier(tier: FidelityTier) -> int:
    """The visual LOD level a fidelity tier selects (``massmodel`` → coarsest).

    An unmapped tier (today only the deferred ``surrogate``) falls back to the
    full-resolution mesh rather than silently picking a coarse one.
    """
    return _LOD_BY_TIER.get(tier, 0)


def select_geometry(
    refs: Iterable[GeometryRef],
    *,
    role: GeometryRole | None = None,
    fmt: GeometryFormat | None = None,
    lod: int | None = None,
) -> list[GeometryRef]:
    """Filter geometry refs by role / format / LOD, preserving document order.

    When ``lod`` is given and a frame has no ref at exactly that level, its **closest
    available coarser-or-equal** tier is used instead (a ``collision`` proxy, which is only
    ever written at ``lod=0``, therefore always resolves). This keeps a fidelity-driven
    selection total: dialing a coarse tier never yields an asset with no geometry.
    """
    chosen = [
        ref
        for ref in refs
        if (role is None or ref.role is role) and (fmt is None or ref.format is fmt)
    ]
    if lod is None:
        return chosen
    picked: dict[tuple[str, GeometryRole, GeometryFormat, str], GeometryRef] = {}
    for ref in chosen:
        # one ref per (frame, role, format, artifact family); ``lod`` picks within it
        key = (ref.frame, ref.role, ref.format, _lod_family(ref.uri))
        best = picked.get(key)
        if best is None or _closer_lod(ref.lod, best.lod, lod):
            picked[key] = ref
    order = {id(ref): i for i, ref in enumerate(chosen)}
    return sorted(picked.values(), key=lambda ref: order[id(ref)])


def _lod_family(uri: str) -> str:
    """The uri with its ``.lodN`` marker stripped — the identity a LOD ladder shares."""
    name = Path(uri).name
    stem, _, suffix = name.rpartition(".")
    base, sep, tail = stem.rpartition(".")
    if sep and tail.startswith("lod") and tail[3:].isdigit():
        stem = base
    return f"{stem}.{suffix}"


def _closer_lod(candidate: int, current: int, want: int) -> bool:
    """Whether ``candidate`` is a better match for ``want`` than ``current``.

    Prefer the coarsest tier that is still at least as fine as requested; if every tier is
    finer than requested (the mesh has no such coarse form), take the coarsest one there is.
    """
    if (candidate <= want) != (current <= want):
        return candidate <= want
    return candidate > current if candidate <= want else candidate < current


def decimate(mesh: trimesh.Trimesh, ratio: float) -> trimesh.Trimesh:
    """A decimated copy of ``mesh`` with at most ``ratio`` of its faces.

    Uses **vertex clustering** (Rossignac-Borrel): snap vertices to a uniform grid over the
    mesh's bounding box, weld each cell to its members' centroid, and drop the faces that
    collapse. The grid resolution is found by a deterministic bisection over
    ``[_MIN_CELLS, _MAX_CELLS]`` — the finest grid that still meets the face budget.

    Why not quadric decimation (``trimesh.simplify_quadric_decimation``)? It needs the
    optional ``fast_simplification`` extension, and its greedy edge-collapse order depends on
    float summation order — the same mesh can decimate differently on another machine.
    Clustering is pure integer bucketing over sorted cells: **same mesh ⇒ same bytes**, which
    is what the determinism gate demands (conventions.md §11).

    A mesh at or below :data:`_MIN_FACES` faces, or a ``ratio >= 1``, is returned unchanged.
    """
    faces = len(mesh.faces)
    if ratio >= 1.0 or faces <= _MIN_FACES:
        return mesh.copy()
    target = max(_MIN_FACES, int(faces * ratio))

    lo, hi = _MIN_CELLS, _MAX_CELLS
    while lo < hi:  # the finest grid whose clustering still fits the budget
        mid = (lo + hi + 1) // 2
        if len(_cluster(mesh, mid).faces) <= target:
            lo = mid
        else:
            hi = mid - 1
    out = _cluster(mesh, lo)
    # A grid so coarse it annihilates the surface is not a LOD tier; fall back to the hull,
    # which is the cheapest mesh that still bounds the body.
    return out if len(out.faces) >= 4 else convex_hull(mesh)


def _cluster(mesh: trimesh.Trimesh, cells: int) -> trimesh.Trimesh:
    """Weld ``mesh``'s vertices onto a ``cells``-resolution grid over its bounding box."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    low, high = mesh.bounds
    longest = float(np.max(high - low))
    if longest <= 0.0:  # pragma: no cover - a zero-extent mesh has no faces to begin with
        return mesh.copy()

    grid = np.floor((vertices - low) / (longest / cells)).astype(np.int64)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()  # numpy 2 keeps the input's shape

    counts = np.bincount(inverse)
    welded = np.zeros((len(counts), 3), dtype=float)
    np.add.at(welded, inverse, vertices)
    welded /= counts[:, None]

    faces = inverse[np.asarray(mesh.faces)]
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    return trimesh.Trimesh(vertices=welded, faces=faces[keep], process=True)


def mass_proxy_mesh(mass_kg: float, inertia: Inertia) -> trimesh.Trimesh:
    """The uniform-density box with a body's mass and inertia — a mesh-free asset's stand-in.

    A Phase-0 reference asset declares mass properties and no meshes, so a preview would
    otherwise be empty. The **inertia-equivalent box** is the honest visual: diagonalize the
    tensor (``I = R·diag(Iₓ,I_y,I_z)·Rᵀ``), invert the box formula ``Iₓ = m(Y²+Z²)/12`` for the
    extents, and rotate the box back onto the body's principal axes. Same mass, same inertia —
    it is a *derived* proxy, never geometry the asset claims, and callers flag it as such.

    Raises :class:`GeometryError` if the tensor violates the inertia triangle inequality (no
    real body has such a tensor; ``fleet lint``'s ``inertia.positive_definite`` rule catches it).
    """
    if mass_kg <= 0.0:
        raise GeometryError(f"mass-proxy geometry needs positive mass, got {mass_kg}")
    tensor = np.array(
        [
            [inertia.ixx, inertia.ixy, inertia.ixz],
            [inertia.ixy, inertia.iyy, inertia.iyz],
            [inertia.ixz, inertia.iyz, inertia.izz],
        ],
        dtype=float,
    )
    moments, axes = np.linalg.eigh(tensor)  # ascending, orthonormal
    if np.linalg.det(axes) < 0.0:
        axes[:, 0] *= -1.0  # keep the proxy right-handed: a mirrored box is a different body

    ix, iy, iz = (float(m) for m in moments)
    squared = np.array(
        [
            6.0 / mass_kg * (iy + iz - ix),
            6.0 / mass_kg * (ix + iz - iy),
            6.0 / mass_kg * (ix + iy - iz),
        ]
    )
    if np.any(squared <= 0.0):
        raise GeometryError(
            "inertia tensor violates the triangle inequality "
            f"(principal moments {ix:g}, {iy:g}, {iz:g}); no uniform box has it"
        )
    box = trimesh.creation.box(extents=np.sqrt(squared))
    rotation = np.eye(4)
    rotation[:3, :3] = axes
    box.apply_transform(rotation)
    return box


# --- reading geometry back -------------------------------------------------------


def load_geometry(path: str | Path) -> trimesh.Trimesh:
    """Load a written geometry artifact **into the SADF body frame**.

    The inverse of :func:`write_geometry`, and the reader the exporters re-materialize a
    ``GeometryRef`` through. Dispatch is by suffix:

    - **glTF/GLB** — the file is y-up (either because Fleet put the body→glTF rotation on its
      node, or because its author obeyed the glTF spec), so the composed scene is rotated back
      through ``_BODY_TO_GLTF⁻¹``. A Fleet-written ``.glb`` therefore returns *exactly* the
      vertices that went in.
    - **USD** — read through ``pxr`` (trimesh has no USD reader), whose stages Fleet writes z-up
      and metre-scaled, i.e. already in the body frame.
    - anything else — :func:`load_mesh`, which treats the file as body-frame (the URDF/SDF mesh
      convention).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".usd", ".usda", ".usdc"}:
        return _load_usd(p)
    mesh = load_mesh(p)
    if suffix in {".glb", ".gltf"}:
        mesh.apply_transform(np.linalg.inv(_BODY_TO_GLTF))
    return mesh


def _load_usd(path: Path) -> trimesh.Trimesh:
    """The first ``UsdGeom.Mesh`` on a USD stage, as a triangulated :class:`trimesh.Trimesh`."""
    from pxr import Usd, UsdGeom

    if not path.is_file():
        raise GeometryError(f"mesh file not found: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise GeometryError(f"cannot open USD stage {path}")
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        usd_mesh = UsdGeom.Mesh(prim)
        points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        return trimesh.Trimesh(vertices=points, faces=_triangulate(counts, indices), process=False)
    raise GeometryError(f"USD stage {path} holds no mesh")


def _triangulate(counts: NDArray[np.int_], indices: NDArray[np.int_]) -> NDArray[np.int_]:
    """Fan-triangulate a USD face-vertex stream (Fleet writes triangles; others may not)."""
    faces: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts.tolist():
        fan = indices[offset : offset + count]
        faces += [(int(fan[0]), int(fan[i]), int(fan[i + 1])) for i in range(1, count - 1)]
        offset += count
    return np.asarray(faces, dtype=np.int64).reshape(-1, 3)


# --- writing geometry ------------------------------------------------------------


def write_obj(mesh: trimesh.Trimesh, path: Path) -> None:
    """Write ``mesh`` as a body-frame Wavefront OBJ (the URDF/SDF mesh interchange form).

    URDF and SDF consumers (MuJoCo, Gazebo, PyBullet, rviz) read OBJ/STL/DAE, not glTF or USD,
    and a URDF ``<mesh>`` is body-frame by convention — no axis conversion, no node transform.
    OBJ is text, so the artifact is diffable and byte-reproducible (conventions.md §11).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mesh.export(file_type="obj", include_texture=False), encoding="utf-8")


def write_geometry(
    mesh: trimesh.Trimesh,
    *,
    role: GeometryRole,
    stem: str,
    frame: str,
    assets_dir: Path,
    uri_prefix: str,
) -> list[GeometryRef]:
    """Write ``mesh`` as USD + glTF and return the ``GeometryRef``s pointing at them.

    A ``visual`` role emits one artifact pair **per LOD tier** in :data:`LOD_RATIOS` — the
    full-resolution mesh at ``lod=0`` (``<stem>.glb``) and a decimated mesh per further tier
    (``<stem>.lod<N>.glb``) — each ref tagged with its ``lod``. A ``collision`` role writes the
    convex hull instead, at ``lod=0`` only: a collision proxy is already the coarsest useful
    form of the body, and a *cheaper* one would under-approximate the contact surface — a
    silently smaller robot is a physics bug, not a saving.

    ``uri_prefix`` is prepended to each filename to make the ref relative to the SADF document
    that will embed it.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    write = partial(_write_tier, frame=frame, assets_dir=assets_dir, uri_prefix=uri_prefix)
    if role is GeometryRole.COLLISION:
        return write(convex_hull(mesh), role=role, stem=stem, lod=0)

    refs: list[GeometryRef] = []
    for lod, ratio in enumerate(LOD_RATIOS):
        tier_stem = stem if lod == 0 else f"{stem}.lod{lod}"
        refs += write(decimate(mesh, ratio), role=role, stem=tier_stem, lod=lod)
    return refs


def _write_tier(
    mesh: trimesh.Trimesh,
    *,
    role: GeometryRole,
    stem: str,
    frame: str,
    lod: int,
    assets_dir: Path,
    uri_prefix: str,
) -> list[GeometryRef]:
    """Write one mesh as both canonical forms and return its two refs."""
    names = {GeometryFormat.GLTF: f"{stem}.glb", GeometryFormat.USD: f"{stem}.usda"}
    _export_glb(mesh, assets_dir / names[GeometryFormat.GLTF])
    _export_usd(mesh, assets_dir / names[GeometryFormat.USD])
    return [
        GeometryRef(role=role, format=fmt, uri=f"{uri_prefix}{name}", frame=frame, lod=lod)
        for fmt, name in names.items()
    ]


#: The rotation from the SADF body frame into glTF's own coordinate system.
#:
#: A SADF asset is authored **x-forward, y-left, z-up** — the URDF/SDF convention SADF
#: inherits, and the frame ``GeometryRef.frame`` names. glTF 2.0 instead fixes its own
#: axes: **+y up, +z forward**. Writing body-frame vertices into a ``.glb`` unchanged
#: produces a file that *claims* to be y-up and is not, so every standard glTF consumer
#: renders the asset on its side.
#:
#: Mapping forward onto +z and up onto +y forces left onto +x. Both frames are
#: right-handed, so this is a proper rotation (determinant +1, the cyclic permutation
#: x→z, y→x, z→y) and nothing is mirrored: a chirally-handed manipulator stays handed
#: the way it was authored.
#:
#: The entries are exact 0.0/1.0, so the emitted matrix carries no float drift.
_BODY_TO_GLTF: np.ndarray = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],  # gltf +x  <- body +y (left)
        [0.0, 0.0, 1.0, 0.0],  # gltf +y  <- body +z (up)
        [1.0, 0.0, 0.0, 0.0],  # gltf +z  <- body +x (forward)
        [0.0, 0.0, 0.0, 1.0],
    ]
)


#: A neutral regolith-grey PBR material for a mesh whose source declared none.
#:
#: A glTF primitive with no material renders in the consumer's default — pure white, unlit
#: flat facets. Worlds' terrain exporter ships a default material for exactly this reason
#: (``worlds/spec/_tiles.py``); an asset preview deserves the same. ``doubleSided`` because
#: an imported hull's winding is not guaranteed, and a back-face-culled rover with an
#: inside-out triangle renders holes.
_DEFAULT_MATERIAL_NAME = "astro-mine-default"
_DEFAULT_BASE_COLOR = (180, 182, 188, 255)


def _with_default_material(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """``mesh``, given a default material if — and only if — it carries no appearance of its own.

    A source mesh's colours or textures are its author's intent and survive untouched; the
    default exists so a bare geometric primitive is not published as an unshaded white blob.
    """
    visual = mesh.visual
    if isinstance(visual, trimesh.visual.TextureVisuals) or getattr(visual, "kind", None):
        return mesh

    shaded = mesh.copy()
    shaded.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name=_DEFAULT_MATERIAL_NAME,
            baseColorFactor=_DEFAULT_BASE_COLOR,
            metallicFactor=0.0,
            roughnessFactor=0.85,
            doubleSided=True,
        )
    )
    return shaded


def _export_glb(mesh: trimesh.Trimesh, path: Path) -> None:
    """Write ``mesh`` as a spec-conformant binary glTF.

    The body→glTF rotation is carried by the mesh node's ``matrix``, **not** baked into
    the vertices. Two reasons: the ``.glb`` and the sibling ``.usda`` then hold the *same*
    vertex arrays, so the two canonical forms stay comparable and content-hash-diffable;
    and the numbers on disk remain expressed in the frame ``GeometryRef.frame`` names,
    which is what a SADF reader expects.

    Vertex normals are written explicitly. glTF lets a consumer flat-shade a primitive that
    omits them, so a normal-less mesh is legal but renders as unlit facets everywhere.
    """
    scene = trimesh.Scene()
    scene.add_geometry(_with_default_material(mesh), transform=_BODY_TO_GLTF)
    path.write_bytes(scene.export(file_type="glb", include_normals=True))


def _export_usd(mesh: trimesh.Trimesh, path: Path) -> None:
    """Author a minimal Z-up, metre-scaled USD mesh stage (ASCII ``.usda`` for
    diffable, reproducible output).

    The mesh prim is the stage's **default prim**, which is what makes the stage *referenceable*:
    a USD reference with no explicit prim path targets the referrer's default prim, and a stage
    that declares none resolves to nothing ("Unresolved reference prim path"). The asset stage
    :mod:`astro_mine.fleet.exporters.usd` writes references these meshes exactly that way.
    """
    from pxr import Gf, Usd, UsdGeom

    if path.exists():
        path.unlink()  # Usd.Stage.CreateNew refuses to clobber an existing layer
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Mesh.Define(stage, "/geometry")
    prim.CreatePointsAttr([Gf.Vec3f(*p) for p in mesh.vertices.tolist()])
    prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    prim.CreateFaceVertexIndicesAttr([int(i) for i in mesh.faces.flatten().tolist()])
    stage.SetDefaultPrim(prim.GetPrim())
    stage.GetRootLayer().Save()
