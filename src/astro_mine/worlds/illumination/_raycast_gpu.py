"""GPU-dispatched fine ray-cast illumination (RM-P1-WORLDS-10; worlds.md §7, §8, §11).

worlds.md §7 puts illumination on a **GPU illumination service** (NVIDIA GPU Operator / MIG) and §8
names *illumination over large regions / long epoch windows* the dominant cost, mitigated by "GPU
ray casting for the fine on-demand path". This backend is exactly that: it reuses the **same**
engine-neutral kernel as :mod:`~astro_mine.worlds.illumination._raycast` (so the device path cannot
diverge from the CPU reference) and only redirects the mask array onto the device via a **lazily
imported** CuPy, keeping CuPy an optional, guarded dependency (the ``[gpu]`` extra).

CuPy needs a CUDA runtime that CI does not have, so the device glue is untestable here and carries
``# pragma: no cover``; its behaviour is gated by ``@pytest.mark.gpu`` tests that ``importorskip``
CuPy. When CuPy (or a GPU) is absent the field-model factory falls back to the CPU ray-cast model —
worlds.md §11 "CPU ray casting as the portable fallback" — so a swarm query is served either way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.worlds.illumination._raycast import RayCastIlluminationModel

__all__ = ["RAYCAST_GPU_BACKEND", "RayCastGpuIlluminationModel"]

#: The backend selector for the GPU ray-cast field model.
RAYCAST_GPU_BACKEND = "raycast_gpu"


class RayCastGpuIlluminationModel(RayCastIlluminationModel):
    """Fine ray-cast illumination with the mask kernel dispatched to a CuPy device.

    Identical to :class:`~astro_mine.worlds.illumination._raycast.RayCastIlluminationModel` except
    the void-filled DEM is moved to the GPU and the shared kernel is run with ``xp = cupy``; the
    result is copied back to the host. CuPy is imported in the constructor so its absence fails
    fast (the factory then falls back to the CPU model). Because CuPy and the ``raycast_gpu`` CPU
    fallback produce the *same* numbers, the ``raycast_gpu`` backend label — and therefore the world
    hash — is identical whether the device or the fallback served the query.
    """

    def __init__(self, terrain: Any, *, backend: str = RAYCAST_GPU_BACKEND, **kwargs: Any) -> None:
        import cupy

        self._cupy = cupy  # pragma: no cover - requires a CUDA device unavailable in CI
        super().__init__(terrain, backend=backend, **kwargs)  # pragma: no cover

    def _xp(self) -> Any:  # pragma: no cover - device-only glue
        return self._cupy

    def _to_device(self, array: NDArray[np.float64]) -> Any:  # pragma: no cover - device-only glue
        return self._cupy.asarray(array)

    def _to_host(self, mask: Any) -> NDArray[np.bool_]:  # pragma: no cover - device-only glue
        return np.asarray(self._cupy.asnumpy(mask), dtype=np.bool_)
