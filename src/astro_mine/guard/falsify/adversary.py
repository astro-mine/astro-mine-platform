"""Adversaries: the untrusted attackers the falsification search drives at the shield.

The wrapped policy and the injected disturbances are *adversarial input* — the whole point of
falsification is that no adversary can defeat the shield (issue #5; guard.md §9.1). Each adversary
produces, per tick: a **proposed commanded acceleration** (what the untrusted policy asks for), a
bounded **external acceleration disturbance** (an unmodeled push the shield cannot see within the
tick), and the **next safety-signal vector** (how energy / thermal / torque / speed evolve). All are
pure-``random.Random``-seeded and stdlib-only, so a run is reproducible and the CI gate is
deterministic (conventions §11 seeded/determinism gate).

Two adversaries ship in the default gate:

- :class:`SeededAdversary` — pseudo-random over the action box and a bounded disturbance, with a
  random walk of the scalar signals that repeatedly crosses the constraint boundaries;
- :class:`WorstCaseAdversary` — structured worst case: full-thrust toward the nearest keep-out
  region, while draining the energy and thermal signals monotonically toward their survival floors.

An optimizer-based (CMA-ES / BO) search is intentionally **not** in the default gate — it would be a
``slow``-marked variant (deselected in CI); the two above plus the property-based Hypothesis driver
give the fast, deterministic coverage.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Protocol

from astro_mine.guard.falsify.derive import signal_envelopes
from astro_mine.guard.falsify.oracle import keepout_barrier
from astro_mine.guard.falsify.rollout import PlantState
from astro_mine.guard.spec.enums import GeometryKind
from astro_mine.guard.spec.ir import CompiledSafetyModel, KeepOutTerm

__all__ = [
    "ANCHOR_SAFE_SIGNALS",
    "DEFAULT_DRAIN_FRACTION",
    "Adversary",
    "SeededAdversary",
    "WorstCaseAdversary",
    "anchor_initial_state",
]

#: The share of a signal's envelope :class:`WorstCaseAdversary` crosses per tick — the generic form
#: of the anchor's old hand-tuned per-signal drain rates. Large enough that the default 120-tick
#: horizon breaches comfortably, small enough that the crossing tick is visible rather than instant.
DEFAULT_DRAIN_FRACTION = 0.05

#: How far past each edge of an envelope :class:`SeededAdversary` samples, as a share of its scale.
#: Sampling strictly inside could never trip a bound, so the shield would never be asked.
_ENVELOPE_OVERSHOOT = 0.35

#: A safe baseline for every anchor scalar signal (comfortably inside each bound) — the starting
#: point the adversaries drift away from. ``charging_window_active`` is temporal-only (not a scalar
#: bound); held at 1.0.
ANCHOR_SAFE_SIGNALS: dict[str, float] = {
    "anchor_torque_nm": 10.0,
    "battery_soc_j": 400_000.0,
    "charging_window_active": 1.0,
    "chassis_temp_k": 250.0,
    "power_available_w": 50.0,
    "traverse_speed_mps": 0.05,
}


def anchor_initial_state(*, position: tuple[float, float, float] = (45.0, 0.0, 25.0)) -> PlantState:
    """A well-inside-the-safe-set start: clear of the lander sphere / slope, all signals safe.

    The default position sits ~45 m from the lander-zone sphere centre (safe set radius 33 m) and
    well above the slope-edge half-space (z ≥ -1 m) — so a violation, if one occurs, is the
    adversary's doing, not the initial condition's."""
    return PlantState(
        position=position, velocity=(0.0, 0.0, 0.0), signals=dict(ANCHOR_SAFE_SIGNALS)
    )


class Adversary(Protocol):
    """The attacker contract the rollout drives (all methods deterministic given the seed)."""

    def action(
        self, index: int, position: list[float], velocity: list[float], spatial_dim: int
    ) -> list[float]:
        """The proposed commanded acceleration this tick (the untrusted policy's ask)."""
        ...

    def accel_disturbance(
        self, index: int, position: list[float], velocity: list[float], spatial_dim: int
    ) -> list[float]:
        """A bounded external acceleration added after certification (an unmodeled push)."""
        ...

    def next_signals(self, index: int, signals: Mapping[str, float]) -> dict[str, float]:
        """The safety-signal vector for the next tick (how the scalar state evolves)."""
        ...


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _unit_toward_keepout(term: KeepOutTerm, position: list[float]) -> list[float]:
    """A unit vector pointing from ``position`` into the keep-out region of ``term`` (toward
    violation) — toward the centre for a sphere/box, opposite the normal for a half-space."""
    if term.shape == GeometryKind.HALF_SPACE:
        vec = [-n for n in term.normal]
    else:  # sphere / box: head for the centre
        vec = [term.center[i] - position[i] for i in range(len(term.center))]
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return [0.0] * max(len(vec), 1)
    return [v / norm for v in vec]


class SeededAdversary:
    """Pseudo-random attacker: uniform actions over the ``±u_max`` box, a small bounded disturbance,
    and a random walk of the scalar signals that repeatedly crosses the constraint boundaries.

    Pass ``compiled`` to attack **any** spec's signals: the walk is then drawn from each signal's
    own envelope, widened past both edges so the boundaries are actually crossed. Without it the
    walk is the anchor's six signals, which is what the anchor gate has always exercised — kept as
    the default so the published anchor result is bit-identical, not because the attack is
    anchor-specific (issue #35).
    """

    def __init__(
        self,
        seed: int,
        *,
        u_max: float = 20.0,
        disturbance: float = 1.0,
        compiled: CompiledSafetyModel | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._u_max = u_max
        self._disturbance = disturbance
        self._envelopes = None if compiled is None else signal_envelopes(compiled)

    def action(
        self, index: int, position: list[float], velocity: list[float], spatial_dim: int
    ) -> list[float]:
        return [self._rng.uniform(-self._u_max, self._u_max) for _ in range(spatial_dim)]

    def accel_disturbance(
        self, index: int, position: list[float], velocity: list[float], spatial_dim: int
    ) -> list[float]:
        d = self._disturbance
        return [self._rng.uniform(-d, d) for _ in range(spatial_dim)]

    def next_signals(self, index: int, signals: Mapping[str, float]) -> dict[str, float]:
        if self._envelopes is not None:
            return self._walk_envelopes(signals)
        rng = self._rng
        soc = _clamp(signals["battery_soc_j"] + rng.uniform(-40_000.0, 20_000.0), 0.0, 600_000.0)
        temp = _clamp(signals["chassis_temp_k"] + rng.uniform(-15.0, 12.0), 80.0, 360.0)
        return {
            "anchor_torque_nm": rng.uniform(0.0, 60.0),
            "battery_soc_j": soc,
            "charging_window_active": 1.0,
            "chassis_temp_k": temp,
            "power_available_w": rng.uniform(0.0, 60.0),
            "traverse_speed_mps": rng.uniform(0.0, 0.2),
        }

    def _walk_envelopes(self, signals: Mapping[str, float]) -> dict[str, float]:
        """Sample each bounded signal from a range overshooting its envelope on both sides.

        Overshoot is what makes the walk an attack: a sample drawn strictly *inside* the envelope
        can never trip a bound, so the shield would never be asked the question.
        """
        assert self._envelopes is not None
        rng = self._rng
        out: dict[str, float] = {}
        for key, env in self._envelopes.items():
            current = float(signals.get(key, env.safe_value()))
            lo, hi = env.floor, env.ceiling
            if lo is None and hi is None:
                out[key] = current  # temporal-only signal: nothing to cross
                continue
            overshoot = _ENVELOPE_OVERSHOOT * env.scale
            low = (lo if lo is not None else current) - overshoot
            high = (hi if hi is not None else current) + overshoot
            out[key] = rng.uniform(low, high)
        return out


class WorstCaseAdversary:
    """Structured worst case: full ``u_max`` thrust toward the nearest keep-out region each tick,
    while draining the bounded signals monotonically toward (and through) their own limits — the
    pincer of a spatial approach and a resource-exhaustion attack.

    Both halves are **derived from the compiled spec**. The spatial half always was — it aims at
    whatever keep-out geometry the model carries. The signal half named the anchor's
    ``battery_soc_j`` / ``chassis_temp_k`` outright, which is what stopped `falsify` from taking a
    spec: any other spec either lacked those keys (a `KeyError`) or had bounds nothing pushed
    against.
    It now walks each signal's own envelope toward the side that signal's own bounds
    forbid (issue #35).

    ``drain_fraction`` is the share of a signal's envelope crossed per tick — the generic form
    of the old per-signal drain rates, chosen so the anchor's floors are still breached well
    inside the default horizon.
    """

    def __init__(
        self,
        compiled: CompiledSafetyModel,
        *,
        u_max: float = 20.0,
        drain_fraction: float = DEFAULT_DRAIN_FRACTION,
    ) -> None:
        self._terms = list(compiled.keep_out_terms)
        self._u_max = u_max
        self._drain_fraction = drain_fraction
        self._envelopes = signal_envelopes(compiled)

    def action(
        self, index: int, position: list[float], velocity: list[float], spatial_dim: int
    ) -> list[float]:
        if not self._terms:
            return [0.0] * spatial_dim
        nearest = min(self._terms, key=lambda t: keepout_barrier(t, position))
        direction = _unit_toward_keepout(nearest, position)
        return [self._u_max * _at(direction, i) for i in range(spatial_dim)]

    def accel_disturbance(
        self, index: int, position: list[float], velocity: list[float], spatial_dim: int
    ) -> list[float]:
        return [0.0] * spatial_dim

    def next_signals(self, index: int, signals: Mapping[str, float]) -> dict[str, float]:
        out = dict(signals)
        for key, env in self._envelopes.items():
            if env.floor is None and env.ceiling is None:
                continue  # temporal-only signal: no side to drive it toward
            current = float(signals.get(key, env.safe_value()))
            out[key] = env.toward_violation(current, fraction=self._drain_fraction)
        return out


def _at(vec: list[float], i: int) -> float:
    return vec[i] if i < len(vec) else 0.0
