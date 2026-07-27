"""2D soft-sphere DEM kernel — the ground-truth granular contact physics (RM-P1-SIM-06).

A dependency-heavy (numpy) discrete-element solver for a vertical trench cross-section: a
bed of mono-disperse discs settling under gravity with Hookean normal + viscous-Coulomb
tangential contacts, box walls, and a rigid blade that sweeps horizontally to excavate. This
is the **high-fidelity ground truth** the reduced-order ``GranularEngine`` (``MASSMODEL``)
approximates and the [Surrogate](surrogate.md) learns (sim.md §4, §11) — Project Chrono /
Taichi-MPM class methods at production scale; this CPU reference tier is the always-works
local realization.

numpy lives *here*, never at package import: :mod:`astro_mine.sim.engines.dem` imports this
only inside its factory, so the base wheel (and ``builtins.py``) stay numpy-free (the
``[dem]`` extra). Determinism is ``TOLERANCE`` (float contact sums are not bit-portable);
same-seed runs reproduce in-process because all randomness flows through a seeded
:class:`numpy.random.Generator` and the integrator is fixed-step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["DemBed", "DemParams", "build_params", "make_bed", "substep"]

_F64 = np.float64
FloatArray = npt.NDArray[np.float64]

#: Fraction of the contact oscillator period used as the explicit substep. Symplectic Euler
#: is stable to ~0.3·T for a damped contact; 0.05 keeps a wide margin for stacked contacts.
_DT_FRACTION = 0.05

#: Displacement (in particle radii) beyond which a particle counts as excavated/disturbed.
_DISPLACEMENT_RADII = 0.75


@dataclass(frozen=True, slots=True)
class DemParams:
    """Immutable physical + numerical parameters of one excavation bed.

    Contact stiffness ``contact_stiffness_n_m`` and the derived per-particle ``mass_kg``
    (disc area * ``regolith_density_kg_m3``, unit out-of-plane depth) set the stable substep
    ``dt_internal_s``. The blade is a vertical segment at ``tool_x0_m`` of height
    ``tool_height_m`` advancing at ``tool_speed_mps`` once digging.
    """

    n_particles: int
    particle_radius_m: float
    regolith_density_kg_m3: float
    contact_stiffness_n_m: float
    normal_damping: float
    tangential_damping: float
    friction_coeff: float
    gravity_m_s2: float
    bed_width_m: float
    wall_stiffness_n_m: float
    tool_x0_m: float
    tool_height_m: float
    tool_speed_mps: float
    dt_internal_s: float

    @property
    def mass_kg(self) -> float:
        """Per-disc mass — area (pi*r^2) * bulk density, unit out-of-plane depth."""
        return self.regolith_density_kg_m3 * math.pi * self.particle_radius_m**2


def build_params(
    *,
    n_particles: int,
    particle_radius_m: float,
    regolith_density_kg_m3: float,
    contact_stiffness_n_m: float,
    restitution: float,
    friction_coeff: float,
    gravity_m_s2: float,
    bed_width_m: float,
    tool_x0_m: float,
    tool_height_m: float,
    tool_speed_mps: float,
) -> DemParams:
    """Resolve scenario inputs into a :class:`DemParams`, deriving damping and the substep.

    ``restitution`` (→0 fully damped, →1 elastic) maps to the normal damping via the standard
    log-restitution relation for a linear spring-dashpot contact; tangential damping tracks it.
    """
    mass = regolith_density_kg_m3 * math.pi * particle_radius_m**2
    log_e = math.log(min(max(restitution, 1e-3), 0.999))
    normal_damping = (
        -2.0 * log_e * math.sqrt(mass * contact_stiffness_n_m) / math.sqrt(math.pi**2 + log_e**2)
    )
    dt_internal = _DT_FRACTION * 2.0 * math.pi * math.sqrt(mass / contact_stiffness_n_m)
    return DemParams(
        n_particles=n_particles,
        particle_radius_m=particle_radius_m,
        regolith_density_kg_m3=regolith_density_kg_m3,
        contact_stiffness_n_m=contact_stiffness_n_m,
        normal_damping=normal_damping,
        tangential_damping=normal_damping,
        friction_coeff=friction_coeff,
        gravity_m_s2=gravity_m_s2,
        bed_width_m=bed_width_m,
        wall_stiffness_n_m=contact_stiffness_n_m,
        tool_x0_m=tool_x0_m,
        tool_height_m=tool_height_m,
        tool_speed_mps=tool_speed_mps,
        dt_internal_s=dt_internal,
    )


@dataclass(slots=True)
class DemBed:
    """Mutable DEM state advanced in place: particle kinematics + tool + accumulators."""

    pos: FloatArray  # (N, 2) positions (x, z), metres
    vel: FloatArray  # (N, 2) velocities, m/s
    pos0: FloatArray  # (N, 2) initial positions, for displacement/excavation tracking
    tool_x_m: float  # current blade x, metres
    floor_reaction_n: float = 0.0  # last-substep total upward floor force
    tool_reaction_n: float = 0.0  # last-substep horizontal draft force on the blade

    def kinetic_energy_j(self, mass_kg: float) -> float:
        """Total kinetic energy — the settling monitor (→ 0 as the bed comes to rest)."""
        return float(0.5 * mass_kg * np.square(self.vel).sum())

    def total_momentum(self, mass_kg: float) -> FloatArray:
        """Total linear momentum (2,) — conserved under internal contacts alone."""
        return np.asarray(mass_kg * self.vel.sum(axis=0), dtype=_F64)

    def displaced_mass_kg(self, params: DemParams) -> float:
        """Excavated/disturbed mass — particles moved > ~0.75 r from their initial spot."""
        moved = np.sqrt(np.square(self.pos - self.pos0).sum(axis=1))
        threshold = _DISPLACEMENT_RADII * params.particle_radius_m
        return float(np.count_nonzero(moved > threshold)) * params.mass_kg


def make_bed(params: DemParams, seed: int) -> DemBed:
    """Seed a close-packed, near-equilibrium shallow bed — the deterministic initial packing.

    Discs are laid in a hexagonally-staggered lattice from the floor up (touching, with a hair
    of clearance so nothing starts overlapping), so the bed starts near static equilibrium and
    settles in a few hundred substeps rather than dropping from height. A small seeded jitter
    breaks the crystalline symmetry.
    """
    rng = np.random.default_rng(seed)
    r = params.particle_radius_m
    dx = 2.05 * r  # horizontal spacing (touching + clearance)
    dz = dx * math.sqrt(3.0) / 2.0  # hexagonal row spacing
    usable = params.bed_width_m - 2.0 * r
    per_row = max(1, int(usable // dx) if usable > dx else 1)
    pos = np.zeros((params.n_particles, 2), dtype=_F64)
    for i in range(params.n_particles):
        row, col = divmod(i, per_row)
        stagger = 0.5 * dx if row % 2 else 0.0
        pos[i, 0] = r + dx * col + stagger
        pos[i, 1] = r + dz * row
    pos[:, 0] = np.clip(pos[:, 0], r, params.bed_width_m - r)
    pos += rng.uniform(-0.05 * r, 0.05 * r, size=pos.shape)
    vel = np.zeros_like(pos)
    return DemBed(pos=pos, vel=vel, pos0=pos.copy(), tool_x_m=params.tool_x0_m)


def _pair_forces(bed: DemBed, params: DemParams) -> FloatArray:
    """Vectorized pairwise soft-sphere contact forces — Hookean normal + capped Coulomb.

    O(N²) over the (small) reference-scale bed. Pairwise forces are equal-and-opposite, so the
    sum conserves momentum exactly (a validated invariant).
    """
    pos, vel = bed.pos, bed.vel
    diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 2): i minus j
    dist = np.sqrt(np.square(diff).sum(axis=-1))  # (N, N)
    np.fill_diagonal(dist, np.inf)
    overlap = 2.0 * params.particle_radius_m - dist
    contact = overlap > 0.0
    if not contact.any():
        return np.zeros_like(pos)
    safe_dist = np.where(dist > 0.0, dist, 1.0)
    normal = diff / safe_dist[..., None]  # unit i←j
    v_rel = vel[:, None, :] - vel[None, :, :]
    v_rel_n = np.sum(v_rel * normal, axis=-1)  # approach is negative
    fn = params.contact_stiffness_n_m * overlap - params.normal_damping * v_rel_n
    fn = np.where(contact, np.maximum(fn, 0.0), 0.0)  # contacts push, never pull
    v_rel_t = v_rel - v_rel_n[..., None] * normal
    vt_mag = np.sqrt(np.square(v_rel_t).sum(axis=-1))
    safe_vt = np.where(vt_mag > 0.0, vt_mag, 1.0)
    t_hat = v_rel_t / safe_vt[..., None]
    ft = np.minimum(params.tangential_damping * vt_mag, params.friction_coeff * fn)
    force_ij = fn[..., None] * normal - ft[..., None] * t_hat  # on i from j
    return np.asarray(force_ij.sum(axis=1), dtype=_F64)


def _boundary_forces(bed: DemBed, params: DemParams, *, tool_active: bool) -> FloatArray:
    """Wall + floor + blade contact forces; records the floor and tool reactions."""
    pos, vel = bed.pos, bed.vel
    r = params.particle_radius_m
    k = params.wall_stiffness_n_m
    force = np.zeros_like(pos)

    # Floor (z = 0): upward normal on any disc dipping below r, damped on approach. The
    # damping is gated on actual contact — otherwise it would drag every downward-moving
    # disc anywhere, injecting spurious external momentum.
    in_floor = pos[:, 1] < r
    floor_overlap = np.where(in_floor, r - pos[:, 1], 0.0)
    floor_f = np.where(
        in_floor,
        np.maximum(k * floor_overlap - params.normal_damping * np.minimum(vel[:, 1], 0.0), 0.0),
        0.0,
    )
    force[:, 1] += floor_f
    bed.floor_reaction_n = float(floor_f.sum())

    # Side walls (x = 0, x = W).
    force[:, 0] += k * np.maximum(r - pos[:, 0], 0.0)
    force[:, 0] -= k * np.maximum(r - (params.bed_width_m - pos[:, 0]), 0.0)

    # Blade: a vertical segment at tool_x pushing discs ahead of it (+x) while digging.
    tool_reaction = 0.0
    if tool_active:
        hit = (
            (pos[:, 0] > bed.tool_x_m)
            & (pos[:, 0] < bed.tool_x_m + r)
            & (pos[:, 1] < params.tool_height_m)
        )
        blade_overlap = np.where(hit, r - (pos[:, 0] - bed.tool_x_m), 0.0)
        rel_v = np.where(hit, vel[:, 0] - params.tool_speed_mps, 0.0)
        blade_f = np.maximum(k * blade_overlap - params.normal_damping * rel_v, 0.0)
        force[:, 0] += blade_f
        tool_reaction = float(blade_f.sum())
    bed.tool_reaction_n = tool_reaction
    return force


def substep(bed: DemBed, params: DemParams, dt_s: float, *, tool_active: bool) -> None:
    """Advance the bed one internal substep of ``dt_s`` (semi-implicit Euler); move the blade.

    ``dt_s`` should be ``params.dt_internal_s`` (the caller may divide a macro-step into an
    integer number of these); a larger step risks the explicit integrator going unstable.
    """
    mass = params.mass_kg
    force = _pair_forces(bed, params) + _boundary_forces(bed, params, tool_active=tool_active)
    force[:, 1] -= mass * params.gravity_m_s2  # gravity
    bed.vel += (force / mass) * dt_s  # semi-implicit: velocity first
    bed.pos += bed.vel * dt_s
    if tool_active:
        bed.tool_x_m += params.tool_speed_mps * dt_s
