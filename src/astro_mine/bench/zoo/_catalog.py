# SPDX-License-Identifier: Apache-2.0
"""The scenario zoo catalog — discovery behind one interface, two backends (bench.md §1, §3, §5).

The zoo is the platform's curated benchmark catalog; the anchor "Lunar Polar Water-Ice Prospecting
v1" is its first entry. Scenarios are **authored as documents, not code** (bench.md §3): each lives
in its own ``<slug>/scenario.json``. Scenarios are immutable once published; a fix is a new version
(a new ``scenario_id`` + directory), so historical leaderboards never need recomputation
(bench.md §5).

:class:`ScenarioCatalog` is the discovery interface, and there are two implementations of it
(bench#33):

- :class:`FilesystemCatalog` — the packaged zoo, discovered by scanning this package. It is the
  **tier-1 default** and stays so: the local tier must run on one workstation with no database, no
  cloud, and no account (CX-LOCAL; conventions.md §7). Nothing about the offline
  ``clone → run → score`` path changes.
- :class:`~astro_mine.bench.zoo.SqlCatalog` — the **PostgreSQL** catalog bench.md §5's data
  architecture specifies ("Indexes specs, versions, lineage; **pgvector** for similarity/search"),
  which the filesystem scan cannot grow into: it indexes every spec, derives each entry's version
  and **lineage** (v2 descends from v1 — the immutable-versioning rule made queryable), and answers
  **similarity search** over a pgvector ``vector`` column. It is selected only when a deployment
  configures ``ASTRO_MINE_BENCH_CATALOG_DSN``.

:class:`CatalogEntry` is the indexed row: the spec itself plus the derived family/version/lineage
and its embedding. :class:`WritableCatalog` adds the authoring surface
(:meth:`~WritableCatalog.upsert`,
:meth:`~WritableCatalog.seed_from`) — the migration path that populates Postgres from the packaged
zoo, and the write path behind the leaderboard's authenticated ``POST /scenarios``
(``scenario:author``, bench#29).

Backlog: RM-P0-BENCH-02 — astro-mine-bench#2;
bench#33 — astro-mine-bench#33
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.bench.scenario import ScenarioSpec

__all__ = [
    "CATALOG_DSN_ENV",
    "SCENARIO_DOC",
    "CatalogEntry",
    "FilesystemCatalog",
    "ScenarioCatalog",
    "SearchHit",
    "WritableCatalog",
    "catalog_entry",
    "default_catalog",
    "list_scenarios",
    "load_scenario",
]

#: The per-scenario document filename discovered under each zoo entry directory.
SCENARIO_DOC = "scenario.json"

#: Env var selecting the Postgres/pgvector catalog. **Unset is the supported default** — the local
#: tier reads the packaged filesystem zoo and needs no database (CX-LOCAL).
CATALOG_DSN_ENV = "ASTRO_MINE_BENCH_CATALOG_DSN"

_ZOO_PACKAGE = "astro_mine.bench.zoo"

#: A zoo scenario id ends in its version: ``lunar-polar-ice-prospecting-v1`` → family
#: ``lunar-polar-ice-prospecting``, version ``1``. That convention *is* the lineage: bench.md §5's
#: "a fix is a new version" means ``…-v2`` descends from ``…-v1`` — what the catalog indexes.
_VERSIONED_ID = re.compile(r"^(?P<family>.+)-v(?P<version>\d+)$")


class CatalogEntry(BaseModel):
    """One indexed zoo scenario: the spec, its identity, and its place in the version lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    name: str
    #: The version family the scenario belongs to (its id with the trailing ``-vN`` removed).
    family: str
    #: The integer version parsed from the id; ``1`` for an unversioned id.
    version: int = Field(ge=1)
    #: The previous version in the same family, when the catalog holds it — the lineage edge.
    parent_id: str | None = None
    spec_version: str
    description: str | None = None
    spec_hash: str
    #: The full ScenarioSpec document, so the catalog can *serve* a scenario, not just index it.
    spec: dict[str, object]
    #: The similarity-search embedding (pgvector column on Postgres) — see :mod:`._embed`.
    embedding: tuple[float, ...] = ()

    def to_spec(self) -> ScenarioSpec:
        """Rehydrate the indexed document into a validated :class:`ScenarioSpec`."""
        return ScenarioSpec.model_validate(self.spec)


class SearchHit(BaseModel):
    """One similarity-search result: the matched entry and its distance from the query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entry: CatalogEntry
    #: Cosine distance in ``[0, 2]`` — 0 is identical. pgvector's ``<=>`` operator on Postgres.
    distance: float


@runtime_checkable
class ScenarioCatalog(Protocol):
    """Discovery over the zoo: what scenarios exist, and what each one is."""

    def list_scenarios(self) -> tuple[str, ...]:
        """Every scenario id in the catalog, in stable sorted order."""
        ...

    def load_scenario(self, scenario_id: str) -> ScenarioSpec:
        """Load and validate a scenario by id; raise :class:`KeyError` if it is not catalogued."""
        ...

    def entries(self) -> tuple[CatalogEntry, ...]:
        """Every indexed entry — the spec plus its version/lineage identity."""
        ...


@runtime_checkable
class WritableCatalog(ScenarioCatalog, Protocol):
    """A catalog that can be *authored into* — the hosted (Postgres) tier.

    The packaged filesystem zoo is deliberately **not** writable: it ships in the wheel, and
    mutating it at runtime would break the immutability the whole reproducibility story rests on
    (bench.md §5). Publishing is a hosted, authenticated, ``scenario:author`` act (bench#29).
    """

    def upsert(self, spec: ScenarioSpec) -> CatalogEntry:
        """Index ``spec`` into the catalog (idempotent on its ``scenario_id``)."""
        ...

    def seed_from(self, source: ScenarioCatalog) -> tuple[CatalogEntry, ...]:
        """Populate this catalog from ``source`` — the migration path off the filesystem scan."""
        ...

    def search(self, query: str | Sequence[float], *, limit: int = 5) -> list[SearchHit]:
        """Rank the catalog by similarity to ``query`` (a text query, or an embedding)."""
        ...

    def lineage(self, scenario_id: str) -> tuple[CatalogEntry, ...]:
        """The scenario's version lineage, oldest first — every version of its family up to it."""
        ...


def _parse_id(scenario_id: str) -> tuple[str, int]:
    """Split a zoo id into its (family, version); an unversioned id is version 1 of itself."""
    match = _VERSIONED_ID.match(scenario_id)
    if match is None:
        return scenario_id, 1
    return match["family"], int(match["version"])


def catalog_entry(spec: ScenarioSpec, *, parent_id: str | None = None) -> CatalogEntry:
    """Index one :class:`ScenarioSpec` into a :class:`CatalogEntry`, embedding included."""
    from astro_mine.bench.zoo._embed import embed_scenario

    family, version = _parse_id(spec.scenario_id)
    return CatalogEntry(
        scenario_id=spec.scenario_id,
        name=spec.name,
        family=family,
        version=version,
        parent_id=parent_id,
        spec_version=spec.spec_version,
        description=spec.description,
        spec_hash=spec.spec_hash,
        spec=spec.model_dump(mode="json"),
        embedding=embed_scenario(spec),
    )


def link_lineage(entries: Sequence[CatalogEntry]) -> tuple[CatalogEntry, ...]:
    """Fill in each entry's ``parent_id``: the previous version present in its family.

    Lineage is *derived*, not authored — bench.md §5's immutability rule ("a fix is a new version")
    already encodes it in the ids, so the catalog reads it back rather than asking authors to
    restate it and get it wrong.
    """
    by_family: dict[str, list[CatalogEntry]] = {}
    for entry in entries:
        by_family.setdefault(entry.family, []).append(entry)

    linked: list[CatalogEntry] = []
    for family_entries in by_family.values():
        ordered = sorted(family_entries, key=lambda e: e.version)
        previous: str | None = None
        for entry in ordered:
            linked.append(entry.model_copy(update={"parent_id": previous}))
            previous = entry.scenario_id
    return tuple(sorted(linked, key=lambda e: e.scenario_id))


class FilesystemCatalog:
    """The packaged zoo, discovered by scanning this package — the tier-1 offline default.

    Adding a scenario is a new *document*, not a loader edit (bench.md §3). This is what
    ``astro-mine bench score`` and ``run(spec, policy)`` read, with no database anywhere in sight.
    """

    def _discover(self) -> dict[str, Traversable]:
        """Map ``scenario_id`` → its ``scenario.json`` resource, by scanning the zoo package."""
        catalog: dict[str, Traversable] = {}
        for entry in files(_ZOO_PACKAGE).iterdir():
            if not entry.is_dir():
                continue
            doc = entry.joinpath(SCENARIO_DOC)
            if not doc.is_file():
                continue
            scenario_id = str(json.loads(doc.read_text(encoding="utf-8"))["scenario_id"])
            if scenario_id in catalog:
                raise ValueError(f"duplicate scenario_id in zoo: {scenario_id!r}")
            catalog[scenario_id] = doc
        return catalog

    def list_scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(self._discover()))

    def load_scenario(self, scenario_id: str) -> ScenarioSpec:
        catalog = self._discover()
        doc = catalog.get(scenario_id)
        if doc is None:
            available = ", ".join(sorted(catalog)) or "(none)"
            raise KeyError(f"no zoo scenario {scenario_id!r}; available: {available}")
        return ScenarioSpec.model_validate_json(doc.read_text(encoding="utf-8"))

    def entries(self) -> tuple[CatalogEntry, ...]:
        specs = [self.load_scenario(scenario_id) for scenario_id in self.list_scenarios()]
        return link_lineage([catalog_entry(spec) for spec in specs])


@lru_cache(maxsize=4)
def _sql_catalog(dsn: str) -> ScenarioCatalog:
    """Open (and cache) the hosted catalog for ``dsn`` — one engine per process, per DSN."""
    from astro_mine.bench.zoo._sql import SqlCatalog

    return SqlCatalog(dsn)


def default_catalog(env: Mapping[str, str] | None = None) -> ScenarioCatalog:
    """The catalog this process reads: the packaged filesystem zoo, unless a DSN is configured.

    The default is deliberately the **offline** one: a workstation with no Postgres must be able to
    list and score every zoo scenario (CX-LOCAL; bench#33 AC3).
    """
    environment = os.environ if env is None else env
    dsn = environment.get(CATALOG_DSN_ENV)
    return _sql_catalog(dsn) if dsn else FilesystemCatalog()


def list_scenarios() -> tuple[str, ...]:
    """Every scenario id registered in the zoo, in stable sorted order."""
    return default_catalog().list_scenarios()


def load_scenario(scenario_id: str) -> ScenarioSpec:
    """Load and validate a zoo scenario by id.

    Raises :class:`KeyError` if no scenario with ``scenario_id`` is registered in the zoo.
    """
    return default_catalog().load_scenario(scenario_id)
