# SPDX-License-Identifier: Apache-2.0
"""URDF/SDF/USD importers + USD/glTF geometry handling (RM-P0-FLEET-02).

Bring existing robot descriptions into SADF: parse URDF (``yourdfpy``), SDF (stdlib XML), or a USD
stage (``pxr``) into a shared intermediate model, then build a validated SADF document whose
geometry is written as normalized USD + glTF artifacts with a generated collision hull and visual
LOD tiers.

``import_description`` dispatches on file extension; ``import_urdf`` / ``import_sdf`` /
``import_usd`` are the explicit entry points. Each returns a
:class:`~astro_mine.core.sadf.SadfDocument` and writes geometry under ``assets_dir`` (refs are
prefixed with ``uri_prefix`` so they resolve relative to wherever the document is written).

The mirror direction — SADF → URDF/SDF/USD — lives in :mod:`astro_mine.fleet.exporters`, whose
``LOSS_CONTRACT`` documents what each direction preserves (fleet.md §11).

Backlog: RM-P0-FLEET-02 -- astro-mine-fleet#2
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from astro_mine.core.sadf import SadfDocument
from astro_mine.fleet.importers._common import ImportError_, build_sadf
from astro_mine.fleet.importers.sdf import parse_sdf
from astro_mine.fleet.importers.urdf import parse_urdf
from astro_mine.fleet.importers.usd import parse_usd

__all__ = [
    "ImportError_",
    "import_description",
    "import_sdf",
    "import_urdf",
    "import_usd",
]

_Importer = Callable[..., SadfDocument]


def import_urdf(path: str | Path, *, assets_dir: str | Path, uri_prefix: str = "") -> SadfDocument:
    """Import a URDF description into a validated SADF document."""
    return build_sadf(parse_urdf(path), assets_dir=Path(assets_dir), uri_prefix=uri_prefix)


def import_sdf(path: str | Path, *, assets_dir: str | Path, uri_prefix: str = "") -> SadfDocument:
    """Import an SDF description into a validated SADF document."""
    return build_sadf(parse_sdf(path), assets_dir=Path(assets_dir), uri_prefix=uri_prefix)


def import_usd(path: str | Path, *, assets_dir: str | Path, uri_prefix: str = "") -> SadfDocument:
    """Import a USD stage into a validated SADF document."""
    return build_sadf(parse_usd(path), assets_dir=Path(assets_dir), uri_prefix=uri_prefix)


_BY_NAME: dict[str, _Importer] = {"urdf": import_urdf, "sdf": import_sdf, "usd": import_usd}
_BY_SUFFIX: dict[str, _Importer] = {
    ".urdf": import_urdf,
    ".sdf": import_sdf,
    ".world": import_sdf,
    ".usd": import_usd,
    ".usda": import_usd,
    ".usdc": import_usd,
}


def import_description(
    path: str | Path,
    *,
    assets_dir: str | Path,
    uri_prefix: str = "",
    fmt: str | None = None,
) -> SadfDocument:
    """Import a URDF, SDF, or USD description, dispatching on ``fmt`` or the file extension."""
    p = Path(path)
    if fmt is not None:
        importer = _BY_NAME.get(fmt)
        if importer is None:
            raise ImportError_(f"unknown format {fmt!r} (expected one of {sorted(_BY_NAME)})")
    else:
        importer = _BY_SUFFIX.get(p.suffix.lower())
        if importer is None:
            raise ImportError_(
                f"cannot infer format from {p.suffix!r}; pass an explicit format "
                f"(supported extensions: {sorted(_BY_SUFFIX)})"
            )
    return importer(p, assets_dir=assets_dir, uri_prefix=uri_prefix)
