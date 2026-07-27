"""SPICE kernel management — furnish / clear / fail-loud (RFC-0002)."""

from __future__ import annotations

from pathlib import Path

import pytest
import spiceypy as sp

from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.spice import (
    SpiceKernelError,
    clear_kernels,
    kernel_pool,
    load_metakernel,
)

# A trivial but valid furnishable text kernel.
_TRIVIAL = "\\begindata\nTEST_SPICE_RFC0002 = ( 42 )\n\\begintext\n"


def test_missing_kernel_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(SpiceKernelError):
        load_metakernel(tmp_path / "does_not_exist.tm")


def test_load_then_clear(tmp_path: Path) -> None:
    kernel = tmp_path / "trivial.tk"
    kernel.write_text(_TRIVIAL, encoding="utf-8")
    load_metakernel(kernel)
    assert sp.gdpool("TEST_SPICE_RFC0002", 0, 1)[0] == 42.0
    clear_kernels()
    # After clearing, the pool variable is gone — the lookup fails loudly.
    with pytest.raises(Exception, match=r"(?i)kernel|pool|variable"):
        sp.gdpool("TEST_SPICE_RFC0002", 0, 1)


def test_kernel_pool_clears_on_exit(tmp_path: Path) -> None:
    kernel = tmp_path / "trivial.tk"
    kernel.write_text(_TRIVIAL, encoding="utf-8")
    with kernel_pool(kernel):
        assert sp.gdpool("TEST_SPICE_RFC0002", 0, 1)[0] == 42.0
    with pytest.raises(Exception, match=r"(?i)kernel|pool|variable"):
        sp.gdpool("TEST_SPICE_RFC0002", 0, 1)


def test_coverage_validation_accepts_covered_window(synthetic_spice) -> None:
    # The synthetic SPK covers its 24 h window → up-front validation passes.
    spk = synthetic_spice.directory / "synth.bsp"
    load_metakernel(spk, coverage=synthetic_spice.window)


def test_coverage_validation_rejects_uncovered_window(synthetic_spice) -> None:
    spk = synthetic_spice.directory / "synth.bsp"
    end = synthetic_spice.window.end.tdb_seconds
    far = EpochWindow(
        start=Epoch(tdb_seconds=end + 1.0e9, scale=TimeScale.TDB),
        end=Epoch(tdb_seconds=end + 2.0e9, scale=TimeScale.TDB),
    )
    with pytest.raises(SpiceKernelError, match="does not cover"):
        load_metakernel(spk, coverage=far)


def test_coverage_validation_requires_an_spk(tmp_path: Path) -> None:
    # A furnished text kernel is not an SPK: validating a window finds no coverage.
    clear_kernels()
    kernel = tmp_path / "trivial.tk"
    kernel.write_text(_TRIVIAL, encoding="utf-8")
    window = EpochWindow(
        start=Epoch(tdb_seconds=0.0, scale=TimeScale.TDB),
        end=Epoch(tdb_seconds=1000.0, scale=TimeScale.TDB),
    )
    try:
        with pytest.raises(SpiceKernelError, match="no SPK kernel furnished"):
            load_metakernel(kernel, coverage=window)
    finally:
        clear_kernels()
