"""Determinism + content-addressed caching + oracle cross-checks (RM-P0-LINK-05).

Unit-level: the content-address digests, the pinned-input cache key, the ContactPlan cache
(including on-disk reproduction across a fresh instance — a "re-run"), and the dependency-free
pass-time oracle comparator. The *live* GMAT cross-check that produces a real reference lives
in ``test_oracle_gmat.py`` and skips wherever GMAT is unavailable; here the comparator itself
is exercised with synthetic references.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pytest

from astro_mine.core.messages import ContactInterval, ContactNode, ContactPlan
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import Epoch, TimeScale
from astro_mine.link.cache import (
    CacheKey,
    LinkCacheError,
    PassTimeBudgetError,
    PassTimeReport,
    PlanCache,
    assert_within_budget,
    build_cache_key,
    cache_key,
    canonical_digest,
    cross_check_pass_times,
    hash_file,
    plan_digest,
)
from astro_mine.link.geometry import SurfaceNode
from astro_mine.link.windows import ContactWindow


def _epoch(t: float) -> Epoch:
    return Epoch(tdb_seconds=t, scale=TimeScale.TDB)


def _plan(*, rate: float | None = 1.0e6) -> ContactPlan:
    return ContactPlan(
        nodes=[ContactNode(id="rover", role=NodeRole.SPACE)],
        intervals=[
            ContactInterval(
                node_a="rover", node_b="relay", start_tdb_s=30.0, end_tdb_s=60.0, max_rate_bps=rate
            )
        ],
        epoch_start_tdb_s=0.0,
        epoch_end_tdb_s=100.0,
    )


# --- canonical_digest -------------------------------------------------------------------


def test_canonical_digest_is_key_order_independent_and_value_sensitive() -> None:
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})
    assert canonical_digest({"a": 1, "b": 2}) != canonical_digest({"a": 1, "b": 3})
    assert canonical_digest(frozenset({"x", "y"})) == canonical_digest({"y", "x"})


class _Band(Enum):
    """A bare (non-str/int) enum — the one shape that reaches the enum canonicalizer."""

    UHF = 1
    S = 2


def test_canonical_digest_handles_pydantic_dataclass_enum_path_bytes() -> None:
    # Pydantic (Epoch), a frozen dataclass (SurfaceNode), a bare enum, Path, and bytes all
    # canonicalize and stay sensitive to their contents.
    assert canonical_digest(_epoch(0.0)) != canonical_digest(_epoch(1.0))
    node = SurfaceNode(name="rover", position_m=(0.0, 0.0, 1.0))
    assert canonical_digest(node) == canonical_digest(SurfaceNode("rover", (0.0, 0.0, 1.0)))
    assert canonical_digest(node) != canonical_digest(SurfaceNode("rover", (0.0, 0.0, 2.0)))
    assert canonical_digest(_Band.UHF) != canonical_digest(_Band.S)
    assert canonical_digest(NodeRole.SPACE) != canonical_digest(NodeRole.GROUND)
    assert len(canonical_digest(Path("/kernels/moon.bsp"))) == 64
    assert canonical_digest(b"\x00\x01") != canonical_digest(b"\x00\x02")


def test_canonical_digest_rejects_an_unserializable_object() -> None:
    with pytest.raises(TypeError, match="cannot canonicalize"):
        canonical_digest(object())


# --- hash_file --------------------------------------------------------------------------


def test_hash_file_is_content_addressed(tmp_path: Path) -> None:
    a = tmp_path / "kernel_a.bsp"
    b = tmp_path / "kernel_b.bsp"
    a.write_bytes(b"ephemeris-bytes")
    b.write_bytes(b"ephemeris-bytes")
    assert hash_file(a) == hash_file(b)  # identical content -> identical hash
    b.write_bytes(b"different-bytes")
    assert hash_file(a) != hash_file(b)


def test_hash_file_missing_raises_loudly(tmp_path: Path) -> None:
    with pytest.raises(LinkCacheError, match="missing cache input"):
        hash_file(tmp_path / "absent.bsp")


# --- plan_digest ------------------------------------------------------------------------


def test_plan_digest_is_stable_and_sensitive() -> None:
    assert plan_digest(_plan()) == plan_digest(_plan())
    assert plan_digest(_plan(rate=1.0e6)) != plan_digest(_plan(rate=2.0e6))


# --- CacheKey / cache_key / build_cache_key ---------------------------------------------


def test_cache_key_returns_the_key_digest() -> None:
    key = CacheKey(kernels="k", terrain="t", nodes="n", epoch="e", config="c")
    assert cache_key(key) == key.digest == cache_key(key)  # deterministic


def test_cache_key_changes_with_link_version() -> None:
    base = dict(kernels="k", terrain="t", nodes="n", epoch="e", config="c")
    v0 = CacheKey(**base, link_version="0.0.0").digest
    v9 = CacheKey(**base, link_version="9.9.9").digest
    assert v0 != v9


def test_build_cache_key_hashes_kernel_content_and_is_order_independent(tmp_path: Path) -> None:
    k1 = tmp_path / "de.bsp"
    k2 = tmp_path / "lsk.tls"
    k1.write_bytes(b"de-bytes")
    k2.write_bytes(b"lsk-bytes")
    epoch = {"start": 0.0, "end": 100.0, "step_s": 60.0}
    key_ab = build_cache_key(kernels=[k1, k2], nodes=["rover"], epoch=epoch, config={"fidelity": 1})
    key_ba = build_cache_key(kernels=[k2, k1], nodes=["rover"], epoch=epoch, config={"fidelity": 1})
    assert cache_key(key_ab) == cache_key(key_ba)  # kernel order does not matter

    k1.write_bytes(b"de-bytes-TRUNCATED")  # a changed kernel misses cache
    key_changed = build_cache_key(
        kernels=[k1, k2], nodes=["rover"], epoch=epoch, config={"fidelity": 1}
    )
    assert cache_key(key_changed) != cache_key(key_ab)


def test_build_cache_key_is_sensitive_to_nodes_epoch_and_config() -> None:
    epoch = {"step_s": 60.0}
    base = cache_key(build_cache_key(nodes=["rover"], epoch=epoch, config={"fidelity": 1}))
    other_nodes = build_cache_key(nodes=["hopper"], epoch=epoch, config={"fidelity": 1})
    other_epoch = build_cache_key(nodes=["rover"], epoch={"step_s": 30.0}, config={"fidelity": 1})
    other_config = build_cache_key(nodes=["rover"], epoch=epoch, config={"fidelity": 2})
    assert cache_key(other_nodes) != base
    assert cache_key(other_epoch) != base
    assert cache_key(other_config) != base


def test_build_cache_key_with_all_defaults_is_a_valid_digest() -> None:
    assert len(cache_key(build_cache_key())) == 64  # canonicalizes empty files + None inputs


# --- PlanCache --------------------------------------------------------------------------


def _key(tag: str = "scenario-1") -> CacheKey:
    return build_cache_key(nodes=[tag])


def test_plan_cache_miss_returns_none() -> None:
    cache = PlanCache()
    key = _key()
    assert key not in cache
    assert cache.get(key) is None


def test_plan_cache_put_then_get_reproduces_the_plan_in_memory() -> None:
    cache = PlanCache()
    key = _key()
    digest = cache.put(key, _plan())
    assert digest == key.digest
    assert key in cache
    assert plan_digest(cache.get(key)) == plan_digest(_plan())  # byte-for-byte reproduction


def test_plan_cache_resolve_computes_once_then_hits() -> None:
    cache = PlanCache()
    key = _key()
    calls = {"n": 0}

    def compute() -> ContactPlan:
        calls["n"] += 1
        return _plan()

    first = cache.resolve(key, compute)
    second = cache.resolve(key, compute)
    assert calls["n"] == 1  # the second call was served from cache
    assert plan_digest(first) == plan_digest(second)


def test_plan_cache_persists_across_instances_a_rerun_hits_disk(tmp_path: Path) -> None:
    key = _key()
    PlanCache(tmp_path).put(key, _plan())
    # A fresh process/instance pointed at the same root reproduces the plan from disk.
    rerun = PlanCache(tmp_path)
    assert key in rerun
    reproduced = rerun.get(key)
    assert reproduced is not None
    assert plan_digest(reproduced) == plan_digest(_plan())


def test_plan_cache_resolve_persisted_is_a_hit_on_rerun(tmp_path: Path) -> None:
    key = _key()
    calls = {"n": 0}

    def compute() -> ContactPlan:
        calls["n"] += 1
        return _plan()

    PlanCache(tmp_path).resolve(key, compute)
    PlanCache(tmp_path).resolve(key, compute)  # fresh instance, same root
    assert calls["n"] == 1


# --- oracle cross-check -----------------------------------------------------------------


def _passes() -> list[ContactWindow]:
    return [
        ContactWindow("dss", "relay", _epoch(100.0), _epoch(160.0)),
        ContactWindow("dss", "relay", _epoch(400.0), _epoch(470.0)),
    ]


def test_cross_check_identical_passes_are_within_budget() -> None:
    report = cross_check_pass_times(_passes(), _passes(), tolerance_s=1.0)
    assert isinstance(report, PassTimeReport)
    assert report.within_budget
    assert report.max_error_s == 0.0
    assert report.counts_match


def test_cross_check_reports_worst_rise_set_delta() -> None:
    shifted = [(100.0, 162.0), (403.0, 470.0)]  # set +2 on pass 1, rise +3 on pass 2
    report = cross_check_pass_times(_passes(), shifted, tolerance_s=5.0)
    assert report.max_rise_error_s == pytest.approx(3.0)
    assert report.max_set_error_s == pytest.approx(2.0)
    assert report.within_budget
    assert not cross_check_pass_times(_passes(), shifted, tolerance_s=1.0).within_budget


def test_cross_check_count_mismatch_fails_even_with_small_deltas() -> None:
    report = cross_check_pass_times(_passes(), _passes()[:1], tolerance_s=1.0)
    assert not report.counts_match
    assert not report.within_budget
    assert report.computed_passes == 2 and report.reference_passes == 1


def test_cross_check_accepts_windows_intervals_and_tuples() -> None:
    interval_ref = [
        ContactInterval(node_a="dss", node_b="relay", start_tdb_s=100.0, end_tdb_s=160.0),
        ContactInterval(node_a="dss", node_b="relay", start_tdb_s=400.0, end_tdb_s=470.0),
    ]
    assert cross_check_pass_times(_passes(), interval_ref, tolerance_s=0.0).within_budget
    assert cross_check_pass_times([(0.0, 10.0)], [(0.0, 10.0)], tolerance_s=0.0).within_budget


def test_cross_check_empty_inputs_trivially_match() -> None:
    report = cross_check_pass_times([], [], tolerance_s=0.0)
    assert report.within_budget and report.max_error_s == 0.0


def test_cross_check_negative_tolerance_raises() -> None:
    with pytest.raises(LinkCacheError, match="tolerance_s must be non-negative"):
        cross_check_pass_times(_passes(), _passes(), tolerance_s=-1.0)


def test_cross_check_unreadable_interval_raises() -> None:
    with pytest.raises(LinkCacheError, match="cannot read a pass interval"):
        cross_check_pass_times([object()], _passes(), tolerance_s=1.0)


def test_assert_within_budget_returns_or_raises() -> None:
    ok = cross_check_pass_times(_passes(), _passes(), tolerance_s=1.0)
    assert assert_within_budget(ok) is ok
    over = cross_check_pass_times(_passes(), [(100.0, 200.0), (400.0, 470.0)], tolerance_s=1.0)
    with pytest.raises(PassTimeBudgetError, match="set Δ"):
        assert_within_budget(over)
