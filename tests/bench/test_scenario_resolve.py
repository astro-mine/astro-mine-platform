"""The content-hash resolver: Core-compat validation, resolved identity, determinism.

Covers the acceptance criterion's second half — *two clean checkouts resolve the identical
scenario* (modeled as resolve determinism) — plus content-by-hash resolution and the fail-loud
Core interface check.
"""

from __future__ import annotations

import pytest

from astro_mine.bench.scenario import (
    ContentPins,
    ContentRef,
    IncompatibleCoreSchema,
    resolve_scenario,
)
from astro_mine.core import SCHEMA_DIGEST
from astro_mine.core.compat import IncompatibleCoreInterface, assert_core_compatible
from tests.bench._factories import make_scenario_spec, sha256_of


def test_resolve_returns_content_addressed_identity() -> None:
    spec = make_scenario_spec()
    resolved = resolve_scenario(spec)
    assert resolved.scenario_id == spec.scenario_id
    assert resolved.scenario_hash.startswith("sha256:")
    assert resolved.spec_hash == spec.spec_hash


def test_resolve_collects_all_referenced_content_by_hash() -> None:
    spec = make_scenario_spec()
    resolved = resolve_scenario(spec)
    assert resolved.content_hashes == {
        "shackleton-v1": sha256_of("a"),
        "astro-mine.fleet.prospecting-rover": sha256_of("b"),
        "ice-prior-v1": sha256_of("c"),
    }


def test_resolve_is_deterministic() -> None:
    # Same spec resolved twice (stand-in for two clean checkouts) → identical identity.
    assert resolve_scenario(make_scenario_spec()).scenario_hash == (
        resolve_scenario(make_scenario_spec()).scenario_hash
    )


def test_changing_a_content_hash_changes_the_scenario_hash() -> None:
    base = resolve_scenario(make_scenario_spec()).scenario_hash
    changed = resolve_scenario(
        make_scenario_spec(
            content=ContentPins(
                world=ContentRef(id="shackleton-v1", content_hash=sha256_of("f")),
                fleet=(ContentRef(id="rover", content_hash=sha256_of("b")),),
            )
        )
    ).scenario_hash
    assert base != changed


def test_toolchain_is_folded_into_the_identity() -> None:
    spec = make_scenario_spec()
    bare = resolve_scenario(spec).scenario_hash
    pinned = resolve_scenario(spec, toolchain={"astro-mine-sim": "0.1.0"})
    assert pinned.scenario_hash != bare
    assert pinned.toolchain == {"astro-mine-sim": "0.1.0"}


def test_incompatible_core_interface_fails_loudly() -> None:
    # 0.y requires an exact minor; the installed Core provides env 0.1.0.
    spec = make_scenario_spec(core_interface={"env": "0.2.0"})
    with pytest.raises(IncompatibleCoreInterface):
        resolve_scenario(spec)


def test_unknown_core_interface_fails_loudly() -> None:
    spec = make_scenario_spec(core_interface={"telepathy": "0.1.0"})
    with pytest.raises(IncompatibleCoreInterface):
        resolve_scenario(spec)


def test_provided_core_override_lets_a_future_pin_resolve() -> None:
    # A scenario pinned to a hypothetical future Core resolves when that Core is the provider.
    spec = make_scenario_spec(core_interface={"env": "1.2.0"})
    resolved = resolve_scenario(spec, provided_core={"env": "1.5.0"})
    assert resolved.core_interface == {"env": "1.2.0"}


# --- the Core schema-digest pin (VERSIONING.md §4.1; CX-REPRO) --------------------------------
#
# The mechanism `assert_core_compatible` cannot provide: CORE_INTERFACE_VERSIONS is frozen at 0.1.0
# through Phase 3, so the interface check returns *compatible* for every Core revision. Only the
# schema digest distinguishes one Core schema set from another.

#: A well-formed digest standing in for a *different* Core schema set (a changed Core revision).
_OTHER_DIGEST = sha256_of("d")


def test_a_scenario_pinning_the_installed_schema_digest_resolves() -> None:
    spec = make_scenario_spec(core_schema_digest=SCHEMA_DIGEST)
    assert resolve_scenario(spec).core_schema_digest == SCHEMA_DIGEST


def test_a_mismatched_schema_digest_fails_loudly() -> None:
    # The heart of the issue: a Core whose schemas differ from the pin must FAIL, not silently
    # resolve to a different scenario_hash under the same (frozen) core_interface 0.1.0.
    spec = make_scenario_spec(core_schema_digest=_OTHER_DIGEST)
    with pytest.raises(IncompatibleCoreSchema) as excinfo:
        resolve_scenario(spec)
    assert _OTHER_DIGEST in str(excinfo.value)
    assert SCHEMA_DIGEST in str(excinfo.value)


def test_a_mismatch_is_caught_even_though_the_interface_check_passes() -> None:
    # Both specs pin a satisfiable core_interface, so assert_core_compatible() accepts both. Only
    # the digest tells them apart — proving the pin adds the guarantee the frozen version cannot.
    pinned = make_scenario_spec(core_schema_digest=SCHEMA_DIGEST)
    stale = make_scenario_spec(core_schema_digest=_OTHER_DIGEST)
    assert_core_compatible(pinned.core_interface)
    assert_core_compatible(stale.core_interface)  # the frozen check cannot distinguish them
    resolve_scenario(pinned)
    with pytest.raises(IncompatibleCoreSchema):
        resolve_scenario(stale)


def test_provided_schema_digest_override_drives_both_branches() -> None:
    # Mirrors provided_core: a scenario pinned to another Core's schemas resolves against it...
    spec = make_scenario_spec(core_schema_digest=_OTHER_DIGEST)
    assert resolve_scenario(spec, provided_schema_digest=_OTHER_DIGEST).core_schema_digest == (
        _OTHER_DIGEST
    )
    # ...and the installed Core, which no longer matches, is rejected.
    with pytest.raises(IncompatibleCoreSchema):
        resolve_scenario(spec, provided_schema_digest=SCHEMA_DIGEST)


def test_an_unpinned_scenario_still_resolves_and_records_the_installed_digest() -> None:
    # The pin is optional (an older spec, or a non-Python binding). Resolution records the digest
    # the run actually validated under, so provenance is complete even without a pin.
    spec = make_scenario_spec()
    assert spec.core_schema_digest is None
    assert resolve_scenario(spec).core_schema_digest == SCHEMA_DIGEST


def test_the_schema_digest_pin_changes_the_scenario_hash() -> None:
    # The pin is part of the task identity, not a side note: it flows into spec_hash and therefore
    # into scenario_hash. Two scenarios differing only by their pinned contract are different tasks.
    unpinned = resolve_scenario(make_scenario_spec()).scenario_hash
    pinned = resolve_scenario(make_scenario_spec(core_schema_digest=SCHEMA_DIGEST)).scenario_hash
    assert unpinned != pinned


def test_consistent_duplicate_pin_is_deduplicated() -> None:
    spec = make_scenario_spec(
        content=ContentPins(
            world=ContentRef(id="shared", content_hash=sha256_of("a")),
            fleet=(ContentRef(id="shared", content_hash=sha256_of("a")),),
        )
    )
    resolved = resolve_scenario(spec)
    assert resolved.content_hashes == {"shared": sha256_of("a")}


def test_conflicting_duplicate_pin_is_rejected() -> None:
    spec = make_scenario_spec(
        content=ContentPins(
            world=ContentRef(id="shared", content_hash=sha256_of("a")),
            fleet=(ContentRef(id="shared", content_hash=sha256_of("b")),),
        )
    )
    with pytest.raises(ValueError, match="conflicting hashes"):
        resolve_scenario(spec)
