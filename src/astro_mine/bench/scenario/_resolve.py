"""The content-hash resolver — materialize a ScenarioSpec into a resolved scenario identity.

:func:`resolve_scenario` turns a :class:`~astro_mine.bench.scenario._spec.ScenarioSpec` into a
content-addressed :class:`ResolvedScenario`: it (1) validates the spec's pinned Core interface
versions against the installed Core (``astro_mine.core.compat``), (2) collects every referenced
content hash into a resolved map, and (3) computes a deterministic ``scenario_hash`` over the
canonical spec plus the pinned toolchain — mirroring how ``astro_mine.worlds.spec`` folds a
``WorldSpec`` + resolved component hashes + toolchain into a content-addressed world id.

Determinism is the point: no wall-clock, no RNG, no environment lookups, so two clean checkouts
resolve the byte-identical scenario (bench.md §2; conventions.md §5). The acceptance criterion —
*changing any input changes the hash* — follows from the spec hash being folded into the identity.

Backlog: RM-P0-BENCH-01 — https://github.com/astro-mine/astro-mine-bench/issues/1
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from astro_mine.bench.scenario._hash import content_hash, normalize_sha256
from astro_mine.bench.scenario._spec import ScenarioSpec
from astro_mine.core import SCHEMA_DIGEST
from astro_mine.core.compat import assert_core_compatible

__all__ = ["IncompatibleCoreSchema", "ResolvedScenario", "resolve_scenario"]


class IncompatibleCoreSchema(RuntimeError):
    """The scenario pins a Core schema set that the installed Core does not provide.

    The counterpart to :class:`astro_mine.core.compat.IncompatibleCoreInterface` for the *contract*
    rather than its version number. ``CORE_INTERFACE_VERSIONS`` is frozen at ``0.1.0`` for every
    interface through Phase 3, so ``assert_core_compatible`` returns *compatible* for every Core
    revision and cannot detect a changed schema. The schema digest can, and does (``VERSIONING.md``
    §4.1; CX-REPRO).
    """


@dataclass(frozen=True)
class ResolvedScenario:
    """A ScenarioSpec resolved to its content-addressed identity — the frozen task instance.

    :attr:`scenario_hash` is the deterministic identity a leaderboard result binds to;
    :attr:`content_hashes` is the resolved ``id → sha256:`` map of every pinned input.
    """

    scenario_id: str
    scenario_hash: str
    spec: ScenarioSpec
    spec_hash: str
    content_hashes: dict[str, str]
    core_interface: dict[str, str]
    #: The Core schema digest this scenario resolved against — the pinned one when the spec pins it
    #: (verified equal to the installed Core's), else the installed Core's, recorded for provenance.
    core_schema_digest: str
    toolchain: dict[str, str]


def resolve_scenario(
    spec: ScenarioSpec,
    *,
    provided_core: Mapping[str, str] | None = None,
    provided_schema_digest: str | None = None,
    toolchain: Mapping[str, str] | None = None,
) -> ResolvedScenario:
    """Resolve a :class:`ScenarioSpec` to its content-addressed :class:`ResolvedScenario`.

    Validates the spec's pinned Core interface versions against the installed Core (or the
    ``provided_core`` map, for testing), resolves all referenced content by hash, and computes a
    deterministic ``scenario_hash`` over the canonical spec + pinned ``toolchain``. The full
    toolchain pinning (lockfile, engine versions) is supplied by the reproducibility harness
    (BENCH-04); it defaults to empty here so the resolver stays pure.

    Raises :class:`astro_mine.core.compat.IncompatibleCoreInterface` if the installed Core cannot
    satisfy the pinned interface versions, :class:`IncompatibleCoreSchema` if the spec pins a Core
    schema digest the installed Core does not provide (``provided_schema_digest`` overrides the
    installed digest, for testing), and ``ValueError`` if a content id is pinned to two different
    hashes (an ambiguous, non-reproducible reference).
    """
    assert_core_compatible(spec.core_interface, provided=provided_core)

    installed_digest = (
        SCHEMA_DIGEST
        if provided_schema_digest is None
        else normalize_sha256(provided_schema_digest)
    )
    if spec.core_schema_digest is not None and spec.core_schema_digest != installed_digest:
        raise IncompatibleCoreSchema(
            f"scenario {spec.scenario_id!r} pins Core schema digest {spec.core_schema_digest}, "
            f"but the installed Core provides {installed_digest}. The Core schema set changed; "
            f"this scenario cannot reproduce byte-for-byte against it. Re-author the scenario "
            f"against the new schemas (a new spec_version) or install the pinned Core."
        )

    content_hashes: dict[str, str] = {}
    for ref in spec.content_refs():
        existing = content_hashes.get(ref.id)
        if existing is not None and existing != ref.content_hash:
            raise ValueError(
                f"content id {ref.id!r} pinned to conflicting hashes: "
                f"{existing} != {ref.content_hash}"
            )
        content_hashes[ref.id] = ref.content_hash

    resolved_toolchain = dict(sorted((toolchain or {}).items()))
    scenario_hash = content_hash({"spec_hash": spec.spec_hash, "toolchain": resolved_toolchain})

    return ResolvedScenario(
        scenario_id=spec.scenario_id,
        scenario_hash=scenario_hash,
        spec=spec,
        spec_hash=spec.spec_hash,
        content_hashes=dict(sorted(content_hashes.items())),
        core_interface=dict(sorted(spec.core_interface.items())),
        core_schema_digest=installed_digest,
        toolchain=resolved_toolchain,
    )
