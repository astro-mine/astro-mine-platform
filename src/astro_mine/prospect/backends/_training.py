"""Training-observation validation shared by the resource-field backends (RM-P0-PROSPECT-02).

The GP and grid backends both condition on the same primitive: scattered ``(position, value)``
observations. This helper normalizes and validates that input into a ``(K, 2)`` array of planar
coordinates (the field is 2-D; depth/3-D is deferred to P1) and a ``(K,)`` value array — failing
loudly on a length mismatch or a half-specified pair. It is deliberately *not* the ordered,
replayable observation log of a belief field — that, and the sealed ground-truth split, are
RM-P0-PROSPECT-04. Here an observation set is just the data a single-shot posterior conditions on.

Backlog: RM-P0-PROSPECT-02 — astro-mine-prospect#2
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.resource import Position

__all__ = ["validate_training"]


def validate_training(
    train_points: Sequence[Position] | None,
    train_values: Sequence[float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate paired training observations into ``(xy, values)`` float64 arrays.

    Returns the planar coordinates ``xy`` of shape ``(K, 2)`` (the ``z`` of each
    :data:`~astro_mine.core.resource.Position` is dropped — 2-D fields only) and the values of
    shape ``(K,)``. Both ``None`` yields empty ``(0, 2)`` / ``(0,)`` arrays — the prior, no
    conditioning. ``train_points`` and ``train_values`` MUST be supplied together and be the
    same length.
    """
    if (train_points is None) != (train_values is None):
        raise ValueError("train_points and train_values must be provided together, or neither")
    if train_points is None or train_values is None:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    points = list(train_points)
    values = list(train_values)
    if len(points) != len(values):
        raise ValueError(
            f"train_points and train_values length mismatch: {len(points)} vs {len(values)}"
        )
    xy = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float64).reshape(-1, 2)
    y = np.array([float(v) for v in values], dtype=np.float64)
    return xy, y
