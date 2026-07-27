"""Zoo integrity — the immutable, content-addressed invariant (bench.md §5, §8; RM-P1-BENCH-12).

The zoo grows by **adding** immutable, content-addressed ScenarioSpecs, never by mutating existing
ones, so historical leaderboards stay valid for their pinned spec (bench.md §8). This module turns
that from a convention into an enforced property: every zoo scenario must (1) validate, (2) pin each
content reference by a well-formed ``sha256:`` digest, and (3) have a ``spec_hash`` that is stable
across a canonical-JSON round trip — the load-bearing content-addressing invariant.

:func:`verify_zoo` runs the check across the whole catalog and returns each scenario's ``spec_hash``
so a test pins known hashes and fails loudly if an "immutable" spec is ever edited in place.
"""

from __future__ import annotations

import re

from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.zoo._catalog import list_scenarios, load_scenario

__all__ = ["ZooIntegrityError", "check_scenario_immutable", "verify_zoo"]

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class ZooIntegrityError(Exception):
    """Raised when a zoo scenario is not immutable/content-addressed as the zoo requires."""


def check_scenario_immutable(spec: ScenarioSpec) -> str:
    """Assert one scenario is content-addressed and hash-stable; return its ``spec_hash``.

    Checks that (1) every pinned content reference is a well-formed ``sha256:`` digest, and (2) the
    ``spec_hash`` recomputes identically after a canonical-JSON round trip (so the content address
    is independent of authoring/serialization). Raises :class:`ZooIntegrityError` on any violation.
    """
    for ref in spec.content_refs():
        if not _SHA256.match(ref.content_hash):
            raise ZooIntegrityError(
                f"scenario {spec.scenario_id!r} content {ref.id!r} is not sha256-pinned: "
                f"{ref.content_hash!r}"
            )
    roundtrip = ScenarioSpec.model_validate_json(spec.model_dump_json())
    if roundtrip.spec_hash != spec.spec_hash:
        raise ZooIntegrityError(
            f"scenario {spec.scenario_id!r} spec_hash is not stable across a JSON round trip"
        )
    return spec.spec_hash


def verify_zoo() -> dict[str, str]:
    """Check every zoo scenario's immutability/content-addressing; return ``{id: spec_hash}``.

    The catalog-wide guard: a test pins historical scenarios' ``spec_hash`` so an accidental
    in-place edit to an "immutable" spec fails CI (bench.md §8 — historical results stay valid).
    """
    return {sid: check_scenario_immutable(load_scenario(sid)) for sid in list_scenarios()}
