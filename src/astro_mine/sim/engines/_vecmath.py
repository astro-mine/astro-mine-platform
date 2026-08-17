# SPDX-License-Identifier: Apache-2.0
"""Pure-Python 3-vector helpers for the reduced-order engine set (RM-P0-SIM-03).

Dependency-light vector math (no numpy) so the concrete engines stay in the always-works
local tier (CX-LOCAL) alongside the reference engine, and same-process determinism is
trivial. A vector is a plain ``(x, y, z)`` float triple in SI units; the frame it resolves
in is the caller's concern (carried on the Core
:class:`~astro_mine.core.messages.model.StateSample`).
"""

from __future__ import annotations

import math

__all__ = [
    "Vec",
    "add",
    "axis_angle_rotate",
    "cross",
    "dot",
    "norm",
    "normalize",
    "scale",
    "sub",
]

#: An SI 3-vector as a plain float triple (metres, m/s, …) in a caller-named frame.
Vec = tuple[float, float, float]


def add(a: Vec, b: Vec) -> Vec:
    """Component-wise ``a + b``."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec, b: Vec) -> Vec:
    """Component-wise ``a - b``."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec, s: float) -> Vec:
    """The vector ``a`` scaled by scalar ``s``."""
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec, b: Vec) -> float:
    """The dot product ``a · b``."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec, b: Vec) -> Vec:
    """The cross product ``a * b``."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec) -> float:
    """The Euclidean length ``‖a‖``."""
    return math.sqrt(dot(a, a))


def normalize(a: Vec) -> Vec:
    """The unit vector along ``a``, or ``(0, 0, 0)`` if ``a`` is the zero vector."""
    length = norm(a)
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def axis_angle_rotate(v: Vec, axis: Vec, angle_rad: float) -> Vec:
    """Rotate ``v`` about a unit ``axis`` by ``angle_rad`` (Rodrigues' rotation)."""
    k = normalize(axis)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # v·cosθ + (k*v)·sinθ + k·(k·v)·(1-cosθ)
    return add(
        add(scale(v, cos_a), scale(cross(k, v), sin_a)),
        scale(k, dot(k, v) * (1.0 - cos_a)),
    )
