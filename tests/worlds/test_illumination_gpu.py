"""GPU ray-cast illumination (RM-P1-WORLDS-10; worlds.md §7 GPU illumination service).

Two concerns:

- The **GPU-vs-CPU tolerance gate** (``@pytest.mark.gpu``) proves the device path — the same kernel
  dispatched to CuPy — matches the CPU ray-cast reference. It ``importorskip``\\s CuPy and needs a
  CUDA device, so CI (no GPU) skips it; run it on a CuPy/CUDA host with ``uv run pytest -m gpu``.
- The **graceful-fallback test** runs everywhere (no GPU needed): it forces CuPy's import to fail,
  asserts the field-model factory degrades to the portable CPU ray-cast model — worlds.md §11 "CPU
  ray casting as the portable fallback" — rather than raising.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from astro_mine.worlds.illumination import (
    RAYCAST_GPU_BACKEND,
    RayCastGpuIlluminationModel,
    RayCastIlluminationModel,
    build_illumination_model,
)
from astro_mine.worlds.terrain import ingest_dem


@pytest.mark.gpu
def test_gpu_ray_cast_matches_cpu_reference(synthetic_dem, synthetic_spice, tmp_path) -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:  # pragma: no cover - host-dependent
        pytest.skip("no CUDA device available")
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    gpu = RayCastGpuIlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    cpu = RayCastIlluminationModel(
        product, backend=RAYCAST_GPU_BACKEND, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE"
    )
    # The GPU path produces fine illumination matching the ray-cast reference exactly (same kernel,
    # float64), so within any stated tolerance — here bit-for-bit on the lit mask.
    np.testing.assert_array_equal(
        gpu.illuminated_mask(synthetic_spice.epoch), cpu.illuminated_mask(synthetic_spice.epoch)
    )
    assert (
        gpu.illumination_at(*_centre(gpu), synthetic_spice.epoch)[0]
        == cpu.illumination_at(*_centre(cpu), synthetic_spice.epoch)[0]
    )
    # Both are labelled `raycast_gpu`, so the world hash matches whether device or fallback ran.
    assert gpu.illumination_hash == cpu.illumination_hash


def test_factory_falls_back_to_cpu_when_cupy_absent(
    monkeypatch, synthetic_dem, synthetic_spice, tmp_path
) -> None:
    # Force `import cupy` to fail regardless of whether CuPy is installed on the host.
    monkeypatch.setitem(sys.modules, "cupy", None)
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    model = build_illumination_model(
        product, backend=RAYCAST_GPU_BACKEND, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE"
    )
    # Degrades to the portable CPU ray-cast — not the GPU subclass, and it does not raise.
    assert isinstance(model, RayCastIlluminationModel)
    assert not isinstance(model, RayCastGpuIlluminationModel)
    assert model.backend == RAYCAST_GPU_BACKEND
    mask = model.illuminated_mask(synthetic_spice.epoch)
    assert mask.shape == (model.height, model.width) and mask.dtype == np.bool_


def _centre(model) -> tuple[float, float]:
    import rasterio.transform

    x, y = rasterio.transform.xy(model.transform, model.height // 2, model.width // 2)
    return float(x), float(y)
