# SPDX-License-Identifier: Apache-2.0
"""Ground-station catalogs beyond DSN — data, not code (RM-P1-LINK-13).

The P0 MVP treated DSN antennas as caller-supplied :class:`~astro_mine.link.windows.GroundStation`
objects. This adds a **catalog** abstraction so a whole ground network — DSN, ESA's **ESTRACK**,
or a user-defined set of antennas — is loaded from **content-addressed YAML** (link.md §11
"ground-station catalog: DSN + ESTRACK + user-defined"). New stations arrive as YAML entries with
**no Link code change** (conventions.md §1.7 "data, not code"); each catalog carries a content
digest so a run's ground segment is reproducible and provenance-tracked (link.md §5).

Built-in catalogs (`dsn`, `estrack`) ship as package data under ``catalogs/``; an arbitrary
catalog loads from any YAML path/text with the same schema. Only public station geodety is used
-- the sensitive live-mission-prediction path is gated elsewhere (see
:mod:`~astro_mine.link.products`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

from astro_mine.link.windows import GroundStation
from astro_mine.link.windows._errors import LinkWindowError

__all__ = [
    "GroundStationCatalog",
    "builtin_catalog",
    "default_ground_catalog",
    "load_ground_catalog",
]

_CATALOG_PACKAGE = "astro_mine.link.constellation.catalogs"
_BUILTINS = ("dsn", "estrack")


@dataclass(frozen=True)
class GroundStationCatalog:
    """A named, content-addressed set of Earth ground stations.

    ``stations`` are ready-to-use :class:`~astro_mine.link.windows.GroundStation` objects;
    ``networks`` is the parallel per-station network label (``dsn`` / ``estrack`` / a custom
    tag). ``digest`` is the SHA-256 of the catalog's canonical bytes — the content address that
    makes a run's ground segment reproducible; ``source`` records where it came from. Station
    names are unique within a catalog (a contact graph keyed by name cannot collide)."""

    stations: tuple[GroundStation, ...]
    networks: tuple[str, ...]
    digest: str
    source: str

    def __post_init__(self) -> None:
        if len(self.stations) != len(self.networks):
            raise LinkWindowError("catalog stations/networks length mismatch")
        names = [s.name for s in self.stations]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise LinkWindowError(f"duplicate ground-station name(s) {dupes} in catalog")

    @property
    def names(self) -> tuple[str, ...]:
        """Every station id, in catalog order."""
        return tuple(s.name for s in self.stations)

    def by_network(self, network: str) -> tuple[GroundStation, ...]:
        """The stations belonging to ``network`` (e.g. ``"estrack"``)."""
        return tuple(
            s for s, net in zip(self.stations, self.networks, strict=True) if net == network
        )

    def merge(
        self, other: GroundStationCatalog, *, source: str | None = None
    ) -> GroundStationCatalog:
        """A combined catalog of ``self`` then ``other`` (e.g. DSN + ESTRACK + custom).

        The merged digest is derived from both inputs' digests, so the union is itself
        content-addressed. Raises :class:`LinkWindowError` on a name collision across the two."""
        digest = hashlib.sha256(f"{self.digest}+{other.digest}".encode()).hexdigest()
        return GroundStationCatalog(
            stations=self.stations + other.stations,
            networks=self.networks + other.networks,
            digest=digest,
            source=source or f"merge({self.source},{other.source})",
        )


def _parse(text: str, *, source: str) -> GroundStationCatalog:
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "stations" not in doc:
        raise LinkWindowError(f"ground catalog {source!r} must be a mapping with a 'stations' list")
    network = str(doc.get("network", "custom"))
    stations: list[GroundStation] = []
    networks: list[str] = []
    for entry in doc["stations"]:
        try:
            station = GroundStation.from_latlon(
                str(entry["name"]),
                float(entry["lat_deg"]),
                float(entry["lon_deg"]),
                min_elevation_deg=float(entry.get("min_elevation_deg", 10.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LinkWindowError(f"malformed station entry in catalog {source!r}: {exc}") from exc
        stations.append(station)
        networks.append(str(entry.get("network", network)))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return GroundStationCatalog(tuple(stations), tuple(networks), digest, source)


def load_ground_catalog(source: str | Path) -> GroundStationCatalog:
    """Load a ground-station catalog from a YAML **file path** or a raw YAML **string**.

    The YAML is a mapping with an optional top-level ``network`` and a ``stations`` list, each
    entry ``{name, lat_deg, lon_deg, min_elevation_deg?, network?}``. The catalog is
    content-addressed by the SHA-256 of the exact YAML bytes. A malformed catalog raises
    :class:`~astro_mine.link.windows.LinkWindowError` loudly rather than yielding a partial set.
    """
    candidate = Path(source) if not isinstance(source, Path) else source
    if candidate.exists():
        return _parse(candidate.read_text(encoding="utf-8"), source=str(candidate))
    if isinstance(source, Path):
        raise LinkWindowError(f"ground catalog file not found: {source}")
    return _parse(source, source="<string>")


def builtin_catalog(name: str) -> GroundStationCatalog:
    """A shipped catalog by name — ``"dsn"`` or ``"estrack"`` (RM-P1-LINK-13 §11)."""
    if name not in _BUILTINS:
        raise LinkWindowError(f"unknown built-in catalog {name!r}; known: {list(_BUILTINS)}")
    text = (resources.files(_CATALOG_PACKAGE) / f"{name}.yaml").read_text(encoding="utf-8")
    return _parse(text, source=f"builtin:{name}")


def default_ground_catalog() -> GroundStationCatalog:
    """The default open catalog: **DSN + ESTRACK** merged, content-addressed."""
    return builtin_catalog("dsn").merge(builtin_catalog("estrack"), source="builtin:dsn+estrack")
