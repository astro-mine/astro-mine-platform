"""Deriving a falsification start and attack from **the spec itself** (issue #35).

The search was anchor-only, and not because the plant is: `WorstCaseAdversary` already aims at
whatever keep-out regions the compiled model carries. What was anchor-shaped was the *state* — a
hardcoded start position and a hardcoded dict of the anchor's six signal keys, which any other spec
would either miss entirely or `KeyError` on. So `falsify` could not take a spec, and the authoring
loop the guide taught (`validate → compile → falsify → sign`) stopped one step short of the step
that was supposed to justify trusting what you wrote.

**There is no scenario, and that is the answer to "which scenario?".** The plant is a synthetic
double integrator carrying a scalar-signal vector (:mod:`astro_mine.guard.falsify.rollout`) — no
terrain, no mission, no world. What the search is *relative to* is the spec's own safe set: the
keep-out geometry it declares and the scalar envelope its bounds carve out. So an arbitrary spec
needs no scenario binding at all; it needs its **own** safe set read out of its compiled IR,
which is what this module does:

- :func:`signal_envelopes` — the ``[floor, ceiling]`` interval each signal's own bounds imply;
- :func:`safe_signals` — one point comfortably inside every envelope, so a violation is the
  adversary's doing rather than the initial condition's;
- :func:`safe_position` — a start clear of every keep-out region by a stated clearance;
- :func:`initial_state` — the two together.

Everything here is a **pure function of the compiled model**, so a falsification run stays
reproducible from `(spec, seed)` alone (conventions §11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from astro_mine.guard.falsify.oracle import keepout_barrier
from astro_mine.guard.falsify.rollout import PlantState
from astro_mine.guard.spec.enums import PredicateOp
from astro_mine.guard.spec.ir import CompiledSafetyModel

__all__ = [
    "DEFAULT_CLEARANCE_M",
    "FalsifyDeriveError",
    "SignalEnvelope",
    "initial_state",
    "safe_position",
    "safe_signals",
    "signal_envelopes",
]

#: How far outside every keep-out region :func:`safe_position` insists on starting, in metres.
DEFAULT_CLEARANCE_M = 10.0

#: A signal a spec bounds on one side only has no midpoint, so the safe start is offset from the
#: bound by this fraction of its magnitude — scale-free, so it works for joules and for m/s alike.
_ONE_SIDED_SLACK = 0.5
#: ...unless the bound is zero, where a fraction of it is still zero.
_ZERO_BOUND_SLACK = 1.0
#: A signal no scalar bound mentions (the anchor's `charging_window_active` is temporal-only) has no
#: envelope to sit inside. It gets a nominal "present and true" value.
_UNBOUNDED_SIGNAL_VALUE = 1.0


class FalsifyDeriveError(Exception):
    """Raised when a spec admits no falsifiable start — a fact about the spec, not a crash."""


@dataclass(frozen=True, slots=True)
class SignalEnvelope:
    """The safe interval one signal's scalar bounds imply. ``None`` means unbounded that side.

    Strict and non-strict operators collapse to the same interval: the difference is a measure-zero
    boundary, and the safe start is chosen in the interior either way.
    """

    key: str
    floor: float | None = None
    ceiling: float | None = None

    @property
    def width(self) -> float | None:
        """The interval's width, or ``None`` when it is unbounded on either side."""
        if self.floor is None or self.ceiling is None:
            return None
        return self.ceiling - self.floor

    def safe_value(self) -> float:
        """A value comfortably inside this envelope."""
        if self.floor is not None and self.ceiling is not None:
            return (self.floor + self.ceiling) / 2.0
        if self.floor is not None:
            return self.floor + _slack(self.floor)
        if self.ceiling is not None:
            return self.ceiling - _slack(self.ceiling)
        return _UNBOUNDED_SIGNAL_VALUE

    @property
    def scale(self) -> float:
        """The natural step size for this signal — its own width, or its bound's magnitude.

        Scale-free by construction, so one `fraction` drives a battery measured in hundreds of
        thousands of joules and a speed measured in tenths of a metre per second alike.
        """
        width = self.width
        if width is not None and width > 0.0:
            return width
        bound = self.floor if self.floor is not None else self.ceiling
        return _ZERO_BOUND_SLACK if bound is None else _slack(bound)

    def toward_violation(self, value: float, *, fraction: float) -> float:
        """Step ``value`` toward the side this envelope forbids, by ``fraction`` of its scale.

        The direction is the envelope's, so the same call drains a battery floor and heats a thermal
        ceiling. A signal bounded on both sides is driven toward its **floor**: an attack has to
        pick one, and picking deterministically keeps the run reproducible.
        """
        step = fraction * self.scale
        if self.floor is not None:
            return value - step
        if self.ceiling is not None:
            return value + step
        return value


def _slack(bound: float) -> float:
    magnitude = abs(bound) * _ONE_SIDED_SLACK
    return magnitude if magnitude > 0.0 else _ZERO_BOUND_SLACK


def signal_envelopes(compiled: CompiledSafetyModel) -> dict[str, SignalEnvelope]:
    """The safe interval each signal in ``compiled`` is held to by its own scalar bounds.

    Every signal in the predicate table appears, including ones only a temporal monitor reads — the
    plant has to supply a value for those too, and their envelope is simply unbounded.
    """
    table = compiled.predicate_table
    floors: dict[str, float] = {}
    ceilings: dict[str, float] = {}
    for bound in compiled.scalar_bounds:
        atom = table.atoms[bound.atom_index]
        key = table.signals[atom.signal_index]
        if atom.op in (PredicateOp.GE, PredicateOp.GT):
            # The tightest floor wins: satisfying it satisfies the looser ones.
            floors[key] = max(floors.get(key, atom.threshold), atom.threshold)
        else:
            ceilings[key] = min(ceilings.get(key, atom.threshold), atom.threshold)

    envelopes: dict[str, SignalEnvelope] = {}
    for key in table.signals:
        floor, ceiling = floors.get(key), ceilings.get(key)
        if floor is not None and ceiling is not None and floor >= ceiling:
            raise FalsifyDeriveError(
                f"signal {key!r} is bounded to an empty interval "
                f"(floor {floor} >= ceiling {ceiling}) — no state satisfies this spec, so there is "
                "nothing to falsify. Fix the spec's scalar bounds."
            )
        envelopes[key] = SignalEnvelope(key=key, floor=floor, ceiling=ceiling)
    return envelopes


def safe_signals(compiled: CompiledSafetyModel) -> dict[str, float]:
    """A signal vector comfortably inside every envelope the spec declares."""
    return {key: env.safe_value() for key, env in signal_envelopes(compiled).items()}


#: The directions :func:`safe_position` probes, in order — axes first (a half-space spec is escaped
#: on an axis), then diagonals. Deterministic, so the derived start is reproducible.
_PROBE_DIRECTIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0),
)
#: Radii probed along each direction, in metres. Geometric, so a spec whose keep-out regions are
#: kilometres across is escaped in a handful of steps.
_PROBE_RADII_M: tuple[float, ...] = (0.0, 50.0, 200.0, 1_000.0, 5_000.0, 25_000.0, 100_000.0)


def safe_position(
    compiled: CompiledSafetyModel,
    *,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    spatial_dim: int = 3,
) -> tuple[float, ...]:
    """A start position clear of every keep-out region in ``compiled`` by ``clearance_m``.

    A deterministic outward probe rather than a solver: the requirement is only *some* interior
    point, the regions are convex barriers with closed-form clearance
    (:func:`~astro_mine.guard.falsify.oracle.keepout_barrier`), and a reproducible answer matters
    more than an optimal one. Raises :class:`FalsifyDeriveError` if no probe clears the spec, which
    is a statement about the spec rather than a failure of the search.
    """
    terms = list(compiled.keep_out_terms)
    if not terms:
        return tuple(0.0 for _ in range(spatial_dim))

    def clearance(point: list[float]) -> float:
        return min(keepout_barrier(term, point) for term in terms)

    for radius in _PROBE_RADII_M:
        for direction in _PROBE_DIRECTIONS:
            norm = math.sqrt(sum(component * component for component in direction))
            point = [radius * direction[i] / norm for i in range(spatial_dim)]
            if clearance(point) >= clearance_m:
                return tuple(point)

    raise FalsifyDeriveError(
        f"no start position clears {compiled.spec_id!r}'s {len(terms)} keep-out region(s) by "
        f"{clearance_m} m within {_PROBE_RADII_M[-1]:.0f} m of the origin. Its keep-out geometry "
        "may cover the reachable space, or its margins may be larger than the frame it is "
        "declared in."
    )


def initial_state(
    compiled: CompiledSafetyModel,
    *,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    spatial_dim: int = 3,
) -> PlantState:
    """A well-inside-the-safe-set start for ``compiled`` — position and signals both.

    The generic counterpart of
    :func:`~astro_mine.guard.falsify.adversary.anchor_initial_state`, and for the same reason: a
    violation should be the adversary's doing, not the initial condition's.
    """
    return PlantState(
        position=safe_position(compiled, clearance_m=clearance_m, spatial_dim=spatial_dim),
        velocity=tuple(0.0 for _ in range(spatial_dim)),
        signals=safe_signals(compiled),
    )
