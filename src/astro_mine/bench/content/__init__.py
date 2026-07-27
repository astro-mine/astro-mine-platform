"""Obtaining the content a scenario pins — the local half of the commons' content distribution.

A :class:`~astro_mine.bench.scenario.ScenarioSpec` pins its world, fleet, resource prior and contact
plan **by content hash**, which makes a benchmark reproducible but does not make its inputs
*obtainable*. :func:`fetch_scenario_content` is the missing verb: it mirrors those pins by digest
from the published registry into a local OCI-layout store, fail-closed, after which everything runs
offline (CX-LOCAL, CX-REPRO).

Bench populates the store and reports a **path**; reading it back into a live bundle store belongs
to the engine that consumes it (Sim's ``open_bundle_store``). Bench never imports Sim
(conventions.md §1.1; bench.md §2.2). Needs the ``[fetch]`` extra for the Hub client.

Backlog: bench#56 — https://github.com/astro-mine/astro-mine-bench/issues/56
"""

from __future__ import annotations

from astro_mine.bench.content._fetch import (
    DEFAULT_CONTENT_SOURCE,
    STORE_ENV,
    DigestMismatch,
    FetchedPin,
    FetchError,
    default_store_path,
    fetch_scenario_content,
    resolve_store_path,
)

__all__ = [
    "DEFAULT_CONTENT_SOURCE",
    "STORE_ENV",
    "DigestMismatch",
    "FetchError",
    "FetchedPin",
    "default_store_path",
    "fetch_scenario_content",
    "resolve_store_path",
]
