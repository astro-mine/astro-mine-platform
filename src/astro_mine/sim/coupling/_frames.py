# SPDX-License-Identifier: Apache-2.0
"""The coupling boundary's frame bridge — inertial ↔ body-fixed via SPICE (RFC-0002).

The transform the multi-domain coupler (RM-P0-SIM-04) could not do. A coupling boundary hands a
producer sub-engine's pose to a consumer sub-engine; when the two engines resolve their agents in
*different* reference frames — the relay orbiter propagates in the inertial ``J2000`` frame, the
surface swarm lives in the rotating body-fixed ``MOON_ME`` frame — the handoff needs the rotating-
frame rotation between them. Core deliberately cannot host that resolution (``core.md`` §2.3: the
name→geometry step needs heavy deps), so the coupler used to refuse the exchange outright and an
orbital+surface co-simulation was simply not expressible.

:class:`SpiceFrameBridge` closes that gap by resolving the rotation through **``astro-mine-spice``**
— the *one* SPICE realization of Core's frame/time vocabulary the whole platform shares (RFC-0002),
rather than a Sim-local re-derivation. It is the default bridge; a host may inject any
:class:`FrameBridge` (a test double, a higher-fidelity kernel set) through the same seam.

**Degrade, don't lie.** SPICE needs a furnished kernel pool to know a body's orientation, and a
missing kernel is a *fault*, never a silent identity: when the pool cannot answer, the bridge raises
:class:`FrameBridgeError` naming both frames. That is exactly the loud failure the coupler shipped
before, so a host that never furnishes kernels sees the behaviour it always saw — it just now has a
way to *succeed* (spice.md §2.5; conventions.md §5).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.spice import SpiceGeometryError, SpiceKernelError, frame_transform

if TYPE_CHECKING:
    from astro_mine.core.units import Epoch, ReferenceFrame

__all__ = ["FrameBridge", "FrameBridgeError", "SpiceFrameBridge"]

#: A 3x3 row-major rotation.
_Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


class FrameBridgeError(RuntimeError):
    """A coupling boundary cannot bridge a producer sample into the consumer's frame.

    Raised when the rotation between the two frames cannot be resolved — in practice, when the SPICE
    kernel pool carries no orientation for the body at the requested epoch. The coupler refuses to
    silently mix frames (conventions.md §5): a boundary either exchanges a correctly rotated state
    or
    it fails loudly."""


@runtime_checkable
class FrameBridge(Protocol):
    """Resolves a frame-explicit state sample into a target reference frame at an epoch.

    The seam the coupler's boundary exchange routes through, so the SPICE realization is swappable
    (a test double, a mission-specific kernel set) without touching the co-simulation machinery."""

    def bridge(
        self, sample: StateSample, target_frame: ReferenceFrame, epoch: Epoch
    ) -> StateSample:
        """``sample`` re-expressed in ``target_frame`` at ``epoch``.

        Identity when the frames already match. Raises :class:`FrameBridgeError` when the rotation
        cannot be resolved — never a silent identity."""
        ...


class SpiceFrameBridge:
    """The default :class:`FrameBridge`: the rotation comes from ``astro-mine-spice`` (RFC-0002).

    Rotates the sample's **pose** (translation *and* orientation) and its **linear velocity** into
    the target frame with the 3x3 rotation ``astro_mine.spice.frame_transform`` resolves at the
    epoch — the same SPICE implementation Worlds and Link resolve their geometry through, so a
    boundary handoff and an illumination query can never disagree about where ``MOON_ME`` points.

    SI is invariant across the platform (conventions.md §5), so no unit bridging is needed — only
    the rotation. This is a *kinematic* frame bridge: it does not add the rotating frame's transport
    velocity (``omega cross r``), because the coupling boundary exchanges **pose**, and the
    consumer's own engine owns its velocity state (see :meth:`bridge`)."""

    def bridge(
        self, sample: StateSample, target_frame: ReferenceFrame, epoch: Epoch
    ) -> StateSample:
        """``sample`` re-expressed in ``target_frame`` at ``epoch``."""
        if sample.frame == target_frame:
            return sample
        rotation = self._rotation(sample.frame, target_frame, epoch)
        translation = sample.pose.translation_m
        x, y, z = _apply(rotation, (translation.x, translation.y, translation.z))
        pose = Transform(
            translation_m=Vec3(x=x, y=y, z=z),
            rotation_quat_xyzw=_compose(rotation, sample.pose.rotation_quat_xyzw),
        )
        velocity = sample.linear_velocity_mps
        rotated_velocity = None
        if velocity is not None:
            vx, vy, vz = _apply(rotation, (velocity.x, velocity.y, velocity.z))
            rotated_velocity = Vec3(x=vx, y=vy, z=vz)
        return sample.model_copy(
            update={
                "frame": target_frame,
                "pose": pose,
                "linear_velocity_mps": rotated_velocity,
            }
        )

    @staticmethod
    def _rotation(source: ReferenceFrame, target: ReferenceFrame, epoch: Epoch) -> _Matrix:
        """The 3x3 rotation ``source -> target`` at ``epoch``, from the shared SPICE foundation.

        A SPICE kernel/geometry fault is re-raised as :class:`FrameBridgeError` naming both frames:
        the coupler's callers contract on *that* error, and a missing kernel must fail loudly rather
        than degrade into an identity rotation (spice.md §2.5)."""
        try:
            matrix = frame_transform(source, target, epoch)
        except (SpiceKernelError, SpiceGeometryError) as exc:
            raise FrameBridgeError(
                f"cannot bridge {source.name!r} -> {target.name!r} at TDB "
                f"{epoch.tdb_seconds}: the SPICE kernel pool cannot resolve the rotation "
                f"({exc}). Furnish the body's orientation kernels (a text PCK, plus the frame "
                "kernel for a non-IAU frame) — a missing kernel is a fault, never an identity "
                "rotation (RFC-0002)."
            ) from exc
        rows = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
        return (
            (rows[0][0], rows[0][1], rows[0][2]),
            (rows[1][0], rows[1][1], rows[1][2]),
            (rows[2][0], rows[2][1], rows[2][2]),
        )


def _apply(rotation: _Matrix, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """``rotation @ vector``."""
    return (
        rotation[0][0] * vector[0] + rotation[0][1] * vector[1] + rotation[0][2] * vector[2],
        rotation[1][0] * vector[0] + rotation[1][1] * vector[1] + rotation[1][2] * vector[2],
        rotation[2][0] * vector[0] + rotation[2][1] * vector[1] + rotation[2][2] * vector[2],
    )


def _compose(rotation: _Matrix, orientation: Quat) -> Quat:
    """The sample's orientation, re-expressed in the target frame: ``q(rotation) * orientation``.

    The pose's quaternion maps body → source frame; left-multiplying by the source → target rotation
    yields body → target, so the attitude travels with the pose rather than being silently reused in
    a frame it was never expressed in."""
    return _quat_multiply(_matrix_to_quat(rotation), orientation)


def _matrix_to_quat(rotation: _Matrix) -> Quat:
    """The unit quaternion (x, y, z, w) of a proper rotation matrix (Shepperd's method: pivot on the
    largest denominator so the square root never loses precision near a 180° rotation)."""
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return Quat(
            x=(rotation[2][1] - rotation[1][2]) / s,
            y=(rotation[0][2] - rotation[2][0]) / s,
            z=(rotation[1][0] - rotation[0][1]) / s,
            w=0.25 * s,
        )
    if rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        s = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        return Quat(
            x=0.25 * s,
            y=(rotation[0][1] + rotation[1][0]) / s,
            z=(rotation[0][2] + rotation[2][0]) / s,
            w=(rotation[2][1] - rotation[1][2]) / s,
        )
    if rotation[1][1] > rotation[2][2]:
        s = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        return Quat(
            x=(rotation[0][1] + rotation[1][0]) / s,
            y=0.25 * s,
            z=(rotation[1][2] + rotation[2][1]) / s,
            w=(rotation[0][2] - rotation[2][0]) / s,
        )
    s = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
    return Quat(
        x=(rotation[0][2] + rotation[2][0]) / s,
        y=(rotation[1][2] + rotation[2][1]) / s,
        z=0.25 * s,
        w=(rotation[1][0] - rotation[0][1]) / s,
    )


def _quat_multiply(a: Quat, b: Quat) -> Quat:
    """The Hamilton product ``a * b`` of two scalar-last unit quaternions."""
    return Quat(
        x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )
