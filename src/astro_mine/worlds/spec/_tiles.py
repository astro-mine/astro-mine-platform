"""A minimal, deterministic 3D-Tiles + glTF terrain export (RM-P0-WORLDS-07; worlds.md §3, §11).

The "early View reuse" thin-slice: a triangulated heightfield mesh of the terrain elevation,
written as a self-contained binary **glTF 2.0** (``.glb``) and referenced by a **3D Tiles 1.1**
``tileset.json`` (a single root tile with a box bounding volume; the LOD pyramid is deferred to
P1, worlds.md §11). Both are written by hand — no glTF/3D-Tiles library — so the byte output is
deterministic for the world-hash gate (conventions.md §1.5).

Geometry: the DEM is strided to a vertex cap, lifted to a mesh in a local **Y-up** frame
(glTF convention) centred on the patch centroid for float precision, with per-vertex normals and a
double-sided default material so it renders regardless of cull state.

**Georeferencing (RM-P1-WORLDS-16).** The tile's local frame is anchored to the body by the root
tile's ``transform``: an east-north-up basis at the patch centroid, expressed in the body-fixed
frame. Publishing it is what lets a consumer render the patch in the right place with no
georeferencing logic of its own. Before this, the transform was left identity and the centroid was
discarded, so every consumer had to reconstruct an approximate anchor from the grid — with a
sub-cell planimetric offset and, worse, no way at all to recover the patch's mean elevation, which
the mesh subtracts from every vertex.

Backlog: RM-P0-WORLDS-07, RM-P1-WORLDS-16 —
https://github.com/astro-mine/astro-mine-worlds/issues/33
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.units import FrameClass, PlanetaryCRS, ReferenceFrame
from astro_mine.worlds.crs import to_lonlat

__all__ = [
    "GLB_NAME",
    "TILESET_NAME",
    "TileAnchor",
    "TilesetExport",
    "enu_to_body_fixed",
    "export_3d_tiles",
    "heightfield_mesh",
]

GLB_NAME = "terrain.glb"
TILESET_NAME = "tileset.json"

# glTF enum constants.
_FLOAT = 5126
_UNSIGNED_INT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_TRIANGLES = 4

_GLB_MAGIC = 0x46546C67  # "glTF"
_JSON_CHUNK = 0x4E4F534A  # "JSON"
_BIN_CHUNK = 0x004E4942  # "BIN\0"


class HeightfieldMesh:
    """A triangulated heightfield: vertices, normals, indices, and a 3D-Tiles box volume."""

    def __init__(
        self,
        positions: NDArray[np.float32],
        normals: NDArray[np.float32],
        indices: NDArray[np.uint32],
        box: list[float],
        centroid: tuple[float, float, float],
    ) -> None:
        self.positions = positions
        self.normals = normals
        self.indices = indices
        self.box = box
        #: The point the mesh is centred on, as ``(map_x_m, map_y_m, elevation_m)`` in the
        #: source CRS. Taken over the **strided** subsample the mesh is built from, not the full
        #: grid — the two differ by up to a cell, which is why it must be reported from here
        #: rather than recomputed by a consumer.
        self.centroid = centroid

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0] // 3)


@dataclass(frozen=True)
class TileAnchor:
    """Where a tileset's local frame sits on the body (RM-P1-WORLDS-16, RM-P1-WORLDS-17).

    ``frame`` is the body-fixed :class:`~astro_mine.core.units.ReferenceFrame` the anchor origin and
    the tileset ``root.transform`` are expressed in — a typed Core frame, not a bare string, so a
    non-Python consumer can validate it against ``units.schema.json`` (RFC-0007). ``height_m`` is
    the elevation of the tile's local origin above the CRS reference sphere. It is exactly the patch
    mean elevation the mesh subtracts from every vertex, so a consumer adding a vertex height to it
    recovers an absolute elevation.
    """

    frame: ReferenceFrame
    longitude_deg: float
    latitude_deg: float
    height_m: float

    def to_manifest(self) -> dict[str, Any]:
        """The ``tiles_anchor`` object published in ``world.json``.

        ``frame`` is serialized per Core's canonical schema (a ``ReferenceFrame`` mapping); the
        ``origin`` sub-object — the lon/lat/height the ``root.transform`` places on the body — is
        unchanged (the View contract of RM-P1-WORLDS-16; view#3).
        """
        return {
            "frame": self.frame.model_dump(mode="json"),
            "origin": {
                "longitude_deg": self.longitude_deg,
                "latitude_deg": self.latitude_deg,
                "height_m": self.height_m,
            },
        }


@dataclass(frozen=True)
class TilesetExport:
    """The result of :func:`export_3d_tiles`: the written paths plus the published anchor."""

    tileset: Path
    glb: Path
    anchor: TileAnchor


def enu_to_body_fixed(
    longitude_deg: float, latitude_deg: float, height_m: float, radius_m: float
) -> list[float]:
    """The 3D-Tiles ``root.transform`` placing a local east-north-up frame on the body.

    Returns the 16 **column-major** floats of a 4x4 matrix whose columns are the east, north, and
    up basis vectors at ``(longitude, latitude)`` on the body's reference sphere, and whose
    translation is that point raised by ``height_m``.

    The basis is the closed form, not ``normalize(cross(z, up))``. The cross-product construction
    (the platform's only east-north-up code today, inline in ``astro_mine.spice.body_geometry``)
    degenerates as ``up`` approaches the pole — and the anchor world sits at latitude -89°, where it
    is already ill-conditioned. These expressions are exact everywhere except the pole itself, where
    longitude is meaningless anyway.

    A 3D Tiles tile's glTF content is Y-up and Cesium rotates it to Z-up, leaving the mesh's own
    axes as east/north/up — which is why this basis composes with the mesh directly.

    **The frame is a tangent plane, so it is exact only at the anchor.** A vertex ``d`` metres away
    sits above the sphere by the sagitta ``d^2 / 2R`` — about 3.7 m at the corners of a 5 km lunar
    patch. That residual is inherent to placing a flat tile with one transform (Cesium's own
    ``eastNorthUpToFixedFrame`` behaves identically); removing it would need per-vertex geodetic
    placement, i.e. a curved tile. It is three orders of magnitude below the ~680 m vertical error a
    consumer inherited when this transform was left identity.
    """
    lam = math.radians(longitude_deg)
    phi = math.radians(latitude_deg)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)

    east = (-sin_lam, cos_lam, 0.0)
    north = (-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi)
    up = (cos_phi * cos_lam, cos_phi * sin_lam, sin_phi)

    r = radius_m + height_m
    origin = (r * up[0], r * up[1], r * up[2])

    # 3D Tiles 1.1 §"transform": a column-major 4x4. Columns: east, north, up, translation.
    return [*east, 0.0, *north, 0.0, *up, 0.0, *origin, 1.0]


def heightfield_mesh(
    elevation: NDArray[np.float64],
    transform: tuple[float, float, float, float, float, float],
    *,
    max_dim: int = 64,
) -> HeightfieldMesh:
    """Build a centred, Y-up heightfield mesh from an elevation grid and its geotransform.

    The grid is strided so neither dimension exceeds ``max_dim`` (keeping the thin-slice tile
    small); NaN/void cells are filled with the valid-cell mean so the mesh stays finite. Returns
    the mesh, a 3D-Tiles ``box`` bounding volume in the mesh's local frame, and the map-frame
    ``centroid`` the mesh was centred on — which :func:`export_3d_tiles` turns into the tile's
    body-fixed anchor.
    """
    elev = np.asarray(elevation, dtype=np.float64)
    height, width = elev.shape
    valid = ~np.isnan(elev)
    fill = float(elev[valid].mean()) if bool(valid.any()) else 0.0
    elev = np.where(valid, elev, fill)

    row_step = max(1, -(-height // max_dim))  # ceil division
    col_step = max(1, -(-width // max_dim))
    rows = np.arange(0, height, row_step)
    cols = np.arange(0, width, col_step)
    sub = elev[np.ix_(rows, cols)]
    n_rows, n_cols = sub.shape

    a, b, c, d, e, f = transform
    col_grid, row_grid = np.meshgrid(cols.astype(np.float64), rows.astype(np.float64))
    world_x = a * col_grid + b * row_grid + c
    world_y = d * col_grid + e * row_grid + f
    world_z = sub

    cx, cy, cz = float(world_x.mean()), float(world_y.mean()), float(world_z.mean())
    # Terrain (east, north, up) -> glTF (X=east, Y=up, Z=-north), centred on the patch centroid.
    gx = world_x - cx
    gy = world_z - cz
    gz = -(world_y - cy)
    positions = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

    indices = _grid_indices(n_rows, n_cols)
    normals = _vertex_normals(positions, indices)

    half = np.abs(positions).max(axis=0)
    box = [
        0.0,
        0.0,
        0.0,
        float(half[0]),
        0.0,
        0.0,
        0.0,
        float(half[1]),
        0.0,
        0.0,
        0.0,
        float(half[2]),
    ]
    return HeightfieldMesh(positions, normals, indices, box, (cx, cy, cz))


def _grid_indices(n_rows: int, n_cols: int) -> NDArray[np.uint32]:
    """Two CCW triangles per grid quad, as a flat ``uint32`` index array."""
    r = np.arange(n_rows - 1)
    c = np.arange(n_cols - 1)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    v00 = (rr * n_cols + cc).ravel()
    v01 = v00 + 1
    v10 = v00 + n_cols
    v11 = v10 + 1
    tri1 = np.stack([v00, v10, v11], axis=1)
    tri2 = np.stack([v00, v11, v01], axis=1)
    return np.vstack([tri1, tri2]).reshape(-1).astype(np.uint32)


def _vertex_normals(
    positions: NDArray[np.float32], indices: NDArray[np.uint32]
) -> NDArray[np.float32]:
    """Area-weighted per-vertex normals accumulated from the triangle faces."""
    tris = indices.reshape(-1, 3)
    p = positions.astype(np.float64)
    face = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
    normals = np.zeros_like(p)
    for k in range(3):
        np.add.at(normals, tris[:, k], face)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return (normals / lengths).astype(np.float32)


def _write_glb(path: Path, mesh: HeightfieldMesh) -> None:
    """Write the mesh as a binary glTF 2.0 (``.glb``) — JSON chunk + BIN chunk."""
    positions = np.ascontiguousarray(mesh.positions, dtype=np.float32)
    normals = np.ascontiguousarray(mesh.normals, dtype=np.float32)
    indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
    pos_bytes, norm_bytes, idx_bytes = positions.tobytes(), normals.tobytes(), indices.tobytes()

    buffer = pos_bytes + norm_bytes + idx_bytes
    n_vertices = int(positions.shape[0])
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "astro-mine-worlds"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "mode": _TRIANGLES,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.6, 0.6, 0.6, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
            }
        ],
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": _ARRAY_BUFFER},
            {
                "buffer": 0,
                "byteOffset": len(pos_bytes),
                "byteLength": len(norm_bytes),
                "target": _ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": len(pos_bytes) + len(norm_bytes),
                "byteLength": len(idx_bytes),
                "target": _ELEMENT_ARRAY_BUFFER,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _FLOAT,
                "count": n_vertices,
                "type": "VEC3",
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            {"bufferView": 1, "componentType": _FLOAT, "count": n_vertices, "type": "VEC3"},
            {
                "bufferView": 2,
                "componentType": _UNSIGNED_INT,
                "count": int(indices.shape[0]),
                "type": "SCALAR",
            },
        ],
    }

    json_bytes = json.dumps(gltf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)  # pad to 4 bytes with spaces
    bin_bytes = buffer + b"\x00" * (-len(buffer) % 4)  # pad to 4 bytes with zeros

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        fh.write(struct.pack("<II", len(json_bytes), _JSON_CHUNK))
        fh.write(json_bytes)
        fh.write(struct.pack("<II", len(bin_bytes), _BIN_CHUNK))
        fh.write(bin_bytes)


def export_3d_tiles(
    tiles_dir: str | Path,
    elevation: NDArray[np.float64],
    transform: tuple[float, float, float, float, float, float],
    *,
    crs: PlanetaryCRS,
    max_dim: int = 64,
) -> TilesetExport:
    """Export ``elevation`` as a georeferenced 3D-Tiles tileset (``tileset.json`` + glTF).

    The tileset is a single root tile whose box bounding volume and geometric error come from the
    mesh extent, and whose ``transform`` anchors the mesh's local east-north-up frame at the patch
    centroid in ``crs``'s body-fixed frame. Both files are byte-deterministic for the world-hash
    gate. Returns the written paths and the published :class:`TileAnchor`.
    """
    out = Path(tiles_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh = heightfield_mesh(elevation, transform, max_dim=max_dim)
    glb_path = out / GLB_NAME
    _write_glb(glb_path, mesh)

    centre_x, centre_y, mean_elevation_m = mesh.centroid
    longitude_deg, latitude_deg = to_lonlat(crs, centre_x, centre_y)
    # The CRS's body-fixed frame, typed (RM-P1-WORLDS-17): its ``name`` is the same string the
    # anchor carried before; ``frame_class``/``center`` make it a schema-valid Core frame so View
    # (RM-P1-VIEW-06) validates rather than mirrors it. For the lunar anchor this is exactly
    # ``MOON_BODY_FIXED``.
    anchor = TileAnchor(
        frame=ReferenceFrame(
            name=crs.body_fixed_frame, frame_class=FrameClass.BODY_FIXED, center=crs.body
        ),
        longitude_deg=longitude_deg,
        latitude_deg=latitude_deg,
        height_m=mean_elevation_m,
    )
    root_transform = enu_to_body_fixed(
        longitude_deg, latitude_deg, mean_elevation_m, crs.reference_radius_m
    )

    geometric_error = 2.0 * max(mesh.box[3], mesh.box[7], mesh.box[11])
    tileset = {
        "asset": {"version": "1.1"},
        "geometricError": geometric_error,
        "root": {
            "boundingVolume": {"box": mesh.box},
            "geometricError": geometric_error,
            "refine": "REPLACE",
            "transform": root_transform,
            "content": {"uri": GLB_NAME},
        },
    }
    tileset_path = out / TILESET_NAME
    tileset_path.write_text(
        json.dumps(tileset, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return TilesetExport(tileset=tileset_path, glb=glb_path, anchor=anchor)
