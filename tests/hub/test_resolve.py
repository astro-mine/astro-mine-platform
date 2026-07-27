"""Resolver tests (RM-P1-HUB-04): version ranges, interface compat, determinism (Hypothesis)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from packaging.version import Version

from astro_mine.hub._content import content_hash
from astro_mine.hub.index import InMemoryCatalog, ingest
from astro_mine.hub.resolve import Resolution, ResolutionError, ResolutionRequest, resolve

from .conftest import make_manifest


def _catalog(*versions: str, name: str = "art", **kwargs: object) -> InMemoryCatalog:
    cat = InMemoryCatalog()
    for version in versions:
        manifest = make_manifest(name, version, **kwargs)  # type: ignore[arg-type]
        ingest(cat, manifest, digest=content_hash(f"{name}:{version}".encode()), publisher="p")
    return cat


# --- basic resolution -----------------------------------------------------------------------


def test_resolve_picks_highest_and_pins_digest() -> None:
    cat = _catalog("1.0.0", "1.2.0", "1.1.0")
    resolution = resolve(cat, ResolutionRequest(name="art"))
    assert isinstance(resolution, Resolution)
    assert resolution.primary.version == "1.2.0"
    assert resolution.primary.reference == "art:1.2.0"
    assert resolution.primary.digest == content_hash(b"art:1.2.0")


def test_resolve_within_version_range() -> None:
    cat = _catalog("1.0.0", "1.5.0", "2.0.0")
    resolution = resolve(cat, ResolutionRequest(name="art", version_spec=">=1.0.0,<2.0.0"))
    assert resolution.primary.version == "1.5.0"


def test_resolve_no_match_raises() -> None:
    cat = _catalog("1.0.0")
    with pytest.raises(ResolutionError):
        resolve(cat, ResolutionRequest(name="art", version_spec=">=3.0.0"))
    with pytest.raises(ResolutionError):
        resolve(cat, ResolutionRequest(name="does-not-exist"))


def test_invalid_specifier_raises() -> None:
    cat = _catalog("1.0.0")
    with pytest.raises(ResolutionError):
        resolve(cat, ResolutionRequest(name="art", version_spec="not-a-spec"))


def test_non_pep440_versions_are_skipped() -> None:
    cat = _catalog("1.0.0", "not-a-version")
    assert resolve(cat, ResolutionRequest(name="art")).primary.version == "1.0.0"


# --- interface + capability compatibility ---------------------------------------------------


def test_unknown_requested_interface_raises() -> None:
    cat = _catalog("1.0.0")
    with pytest.raises(ResolutionError):
        resolve(cat, ResolutionRequest(name="art", interfaces={"poilcy": "0.1.0"}))  # misspelled


def test_refuses_incompatible_core_major() -> None:
    # the only artifact declares an incompatible Core interface major → unresolvable
    cat = _catalog("1.0.0", interfaces={"policy": "2.0.0"})
    with pytest.raises(ResolutionError):
        resolve(cat, ResolutionRequest(name="art"))


def test_resolves_requested_interface_and_tags() -> None:
    cat = InMemoryCatalog()
    ingest(
        cat,
        make_manifest("art", "1.0.0", interfaces={"policy": "0.1.0"}, tags=["mobility.wheeled"]),
        digest=content_hash(b"a"),
        publisher="p",
    )
    ok = resolve(
        cat,
        ResolutionRequest(
            name="art", interfaces={"policy": "0.1.0"}, capability_tags=["mobility.wheeled"]
        ),
    )
    assert ok.primary.version == "1.0.0"
    with pytest.raises(ResolutionError):  # a tag the artifact lacks
        resolve(cat, ResolutionRequest(name="art", capability_tags=["propulsion.chemical"]))


def test_yanked_is_refused_by_default() -> None:
    cat = _catalog("1.0.0", "1.1.0")
    entry = cat.get("art:1.1.0")
    assert entry is not None
    entry.yanked = True
    assert resolve(cat, ResolutionRequest(name="art")).primary.version == "1.0.0"


def test_provided_interfaces_override() -> None:
    # a runtime that provides policy 2.0.0 resolves an artifact built against it
    cat = _catalog("1.0.0", interfaces={"policy": "2.0.0"})
    resolution = resolve(
        cat, ResolutionRequest(name="art", provided_interfaces={"policy": "2.0.0"})
    )
    assert resolution.primary.version == "1.0.0"


# --- properties (Hypothesis) ----------------------------------------------------------------

_VERSIONS = st.lists(
    st.tuples(st.integers(0, 9), st.integers(0, 30), st.integers(0, 30)),
    min_size=1,
    max_size=8,
    unique=True,
)


@given(triples=_VERSIONS)
def test_property_resolves_highest_version(triples: list[tuple[int, int, int]]) -> None:
    versions = [f"{a}.{b}.{c}" for a, b, c in triples]
    cat = _catalog(*versions)
    resolution = resolve(cat, ResolutionRequest(name="art"))
    assert Version(resolution.primary.version) == max(Version(v) for v in versions)


@given(triples=_VERSIONS)
def test_property_resolution_is_deterministic(triples: list[tuple[int, int, int]]) -> None:
    versions = [f"{a}.{b}.{c}" for a, b, c in triples]
    cat = _catalog(*versions)
    request = ResolutionRequest(name="art")
    assert resolve(cat, request) == resolve(cat, request)
