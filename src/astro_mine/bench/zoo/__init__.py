"""The scenario zoo — the curated benchmark catalog (bench.md §1, §3, §5).

The anchor "Lunar Polar Water-Ice Prospecting v1" scenario lives here. :func:`load_scenario` loads a
scenario by id, :func:`list_scenarios` lists the catalog, and :func:`resolve_scenario` (re-exported
from :mod:`astro_mine.bench.scenario`) materializes a spec into its content-addressed
:class:`ResolvedScenario` identity. :func:`resolve_anchor` is the one-call load-and-resolve of the
anchor. New scenarios are authored as documents and discovered automatically (bench.md §3).

Discovery sits behind the :class:`ScenarioCatalog` interface, with two backends (bench#33):
:class:`FilesystemCatalog` (the packaged zoo — the **tier-1 offline default**, no database)
and :class:`SqlCatalog` (the **PostgreSQL + pgvector** catalog bench.md §5 specifies: spec, version
and lineage index, plus similarity search). :func:`default_catalog` picks between them from the
environment, so the offline ``clone → run → score`` path is untouched by the hosted one.

Backlog: RM-P0-BENCH-02 — https://github.com/astro-mine/astro-mine-bench/issues/2;
bench#33 — https://github.com/astro-mine/astro-mine-bench/issues/33
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.bench.scenario import ResolvedScenario, resolve_scenario
from astro_mine.bench.zoo._catalog import (
    CATALOG_DSN_ENV,
    CatalogEntry,
    FilesystemCatalog,
    ScenarioCatalog,
    SearchHit,
    WritableCatalog,
    catalog_entry,
    default_catalog,
    list_scenarios,
    load_scenario,
)
from astro_mine.bench.zoo._embed import EMBEDDING_DIM, Embedder, embed_scenario, embed_text
from astro_mine.bench.zoo._integrity import (
    ZooIntegrityError,
    check_scenario_immutable,
    verify_zoo,
)

if TYPE_CHECKING:
    from astro_mine.bench.zoo._sql import SqlCatalog

#: The anchor scenario id — the Phase-0 reference benchmark the whole MVP is built to run.
ANCHOR_SCENARIO_ID = "lunar-polar-ice-prospecting-v1"

__all__ = [
    "ANCHOR_SCENARIO_ID",
    "CATALOG_DSN_ENV",
    "EMBEDDING_DIM",
    "CatalogEntry",
    "Embedder",
    "FilesystemCatalog",
    "ResolvedScenario",
    "ScenarioCatalog",
    "SearchHit",
    "WritableCatalog",
    "ZooIntegrityError",
    "catalog_entry",
    "check_scenario_immutable",
    "default_catalog",
    "embed_scenario",
    "embed_text",
    "list_scenarios",
    "load_scenario",
    "open_sql_catalog",
    "resolve_anchor",
    "resolve_scenario",
    "verify_zoo",
]


def open_sql_catalog(url: str) -> SqlCatalog:
    """Open the Postgres/pgvector zoo catalog at ``url`` (requires the ``[leaderboard]`` extra).

    Thin lazy wrapper so the base package imports without SQLAlchemy — the local tier reads the
    packaged filesystem zoo and needs no database (CX-LOCAL).
    """
    from astro_mine.bench.zoo._sql import SqlCatalog as _SqlCatalog

    return _SqlCatalog(url)


def resolve_anchor() -> ResolvedScenario:
    """Load and resolve the anchor scenario end-to-end into its content-addressed identity."""
    return resolve_scenario(load_scenario(ANCHOR_SCENARIO_ID))
