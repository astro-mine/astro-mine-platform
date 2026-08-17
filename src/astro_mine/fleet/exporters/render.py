# SPDX-License-Identifier: Apache-2.0
"""SADF -> a single preview artifact (``fleet render``; RM-P0-FLEET-01).

The asset preview [Studio](https://github.com/astro-mine/docs) shows in its robot menu and
[View](https://github.com/astro-mine/docs)'s ``<AssetPreview/>`` widget renders (fleet.md §6;
RM-P1-VIEW-03). Composes every link's visual geometry at one LOD tier, each posed by its frame's
place in the tree, into **one self-contained file** — glTF for the web, USD for Sim/Studio.

**No GPU, no renderer, no network.** fleet.md §7 puts a GPU behind an *optional* offline raster
thumbnail and requires the local tier to always work; a preview *artifact* is the form that does,
and it is what the consumers actually want — Cesium and Hydra rasterize it themselves, at the
viewport's resolution, rather than being handed someone else's PNG.

**Mesh-free assets.** Every Phase-0 reference asset declares mass properties and no meshes, so a
literal preview of one would be an empty scene. Such a link is previewed with its
**inertia-equivalent proxy box** (:func:`~astro_mine.fleet.geometry.mass_proxy_mesh`) — same mass,
same inertia tensor — and the substitution is reported as a ``render.proxy_geometry`` finding, so
a proxy is never mistaken for geometry the asset claims.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from astro_mine.core.sadf import SadfDocument
from astro_mine.core.sadf.enums import FidelityTier, GeometryFormat, GeometryRole
from astro_mine.fleet import geometry
from astro_mine.fleet.exporters import _common
from astro_mine.fleet.exporters._common import ExportError, ExportResult, LossFinding, RealizedLink

__all__ = ["PREVIEW_FORMATS", "render_preview"]

#: The preview forms, by file suffix — the two canonical geometry formats SADF already speaks.
PREVIEW_FORMATS: tuple[str, ...] = ("glb", "usd")


def render_preview(
    doc: SadfDocument,
    out: str | Path,
    *,
    base_dir: str | Path | None = None,
    fmt: str = "glb",
    fidelity: FidelityTier = FidelityTier.KINEMATIC,
) -> ExportResult:
    """Compose ``doc``'s visual geometry into one preview file at ``out``.

    ``fidelity`` picks the LOD tier (a preview defaults to the middle one — a thumbnail has no use
    for the finest mesh); ``fmt`` is ``glb`` or ``usd``. Raises :class:`ExportError` if the asset
    has neither geometry nor mass properties, which is the only case with genuinely nothing to
    show.
    """
    if fmt not in PREVIEW_FORMATS:
        raise ExportError(f"unknown preview format {fmt!r} (expected one of {PREVIEW_FORMATS})")

    out_path = Path(out)
    source = Path(base_dir) if base_dir is not None else out_path.parent
    model = _common.realize(doc, fidelity=fidelity)
    losses: list[LossFinding] = []

    scene = trimesh.Scene()
    for link in model.links:
        for index, mesh in enumerate(_visuals(link, source, losses)):
            posed = mesh.copy()
            posed.apply_transform(link.world)  # the link's pose in the root frame
            name = f"{link.name}_{index}"
            scene.add_geometry(posed, node_name=name, geom_name=name)

    if not scene.geometry:
        raise ExportError(
            f"asset {doc.asset.identity.id!r} has no visual geometry and no bodies to derive a "
            "proxy from — there is nothing to preview"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "glb":
        _write_glb(scene, out_path)
    else:
        _write_usd(scene, out_path)
    return ExportResult(path=out_path, losses=tuple(losses))


def _visuals(
    link: RealizedLink, base_dir: Path, losses: list[LossFinding]
) -> list[trimesh.Trimesh]:
    """A link's visual meshes at the selected tier — or its inertia-equivalent proxy boxes."""
    refs = [
        ref
        for ref in link.geometry
        if ref.role is GeometryRole.VISUAL and ref.format is GeometryFormat.GLTF
    ]
    meshes = []
    for ref in refs:
        try:
            meshes.append(_common.mesh_for(ref, base_dir))
        except geometry.GeometryError as exc:
            losses.append(
                LossFinding(
                    "geometry.unresolved",
                    f"asset.geometry[{ref.uri!r}]",
                    f"cannot read {ref.uri!r} ({exc}); the link is left out of the preview",
                )
            )
    if meshes or not link.has_inertia:
        return meshes

    losses.append(
        LossFinding(
            "render.proxy_geometry",
            f"asset.frames[{link.name!r}]",
            f"link {link.name!r} declares mass but no visual mesh; the preview shows its "
            "inertia-equivalent box (same mass, same inertia tensor) — a derived proxy, not "
            "geometry the asset claims",
        )
    )
    return _common.proxy_meshes(link)


def _write_glb(scene: trimesh.Scene, path: Path) -> None:
    """Write the preview as binary glTF, rotated into glTF's y-up frame.

    The scene is composed in the SADF body frame (z-up); glTF *defines* its axes as y-up, so the
    rotation goes on the scene's root node — the same convention
    :func:`~astro_mine.fleet.geometry.write_geometry` uses for a single mesh, so a preview and the
    meshes it was built from agree.
    """
    rotated = trimesh.Scene()
    to_gltf = geometry.gltf_node_transform()
    for name, mesh in scene.geometry.items():
        rotated.add_geometry(mesh, node_name=name, geom_name=name, transform=to_gltf)
    path.write_bytes(rotated.export(file_type="glb", include_normals=True))  # type: ignore[no-untyped-call]


def _write_usd(scene: trimesh.Scene, path: Path) -> None:
    """Write the preview as a z-up, metre-scaled USD stage — one mesh prim per link."""
    from pxr import Gf, Usd, UsdGeom

    if path.exists():
        path.unlink()  # Usd.Stage.CreateNew refuses to clobber
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/preview")
    stage.SetDefaultPrim(root.GetPrim())

    for name, mesh in scene.geometry.items():
        prim = UsdGeom.Mesh.Define(stage, f"/preview/{_prim_name(name)}")
        faces = np.asarray(mesh.faces)
        prim.CreatePointsAttr([Gf.Vec3f(*point) for point in np.asarray(mesh.vertices).tolist()])
        prim.CreateFaceVertexCountsAttr([3] * len(faces))
        prim.CreateFaceVertexIndicesAttr([int(i) for i in faces.flatten().tolist()])
    stage.GetRootLayer().Save()


def _prim_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    return cleaned if cleaned[:1].isalpha() or cleaned[:1] == "_" else f"_{cleaned}"
