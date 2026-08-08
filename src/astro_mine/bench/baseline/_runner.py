"""The episode-runner seam + a dependency-clean reference runner (RM-P0-BENCH-05).

An :class:`EpisodeRunner` maps a resolved scenario + policy + seed to the
:class:`~astro_mine.bench.metrics.EpisodeTrace` the metric set scores. It is the seam the
local scoring path (:func:`~astro_mine.bench.baseline.run`) drives, and — per the narrow
waist (conventions.md §1.1) and bench.md §2.2 ("Bench composes, never a second simulator")
— the seam an **injected** [Sim](sim.md) runner slots into: that runner drives Sim's
``Simulator`` (a Core ``Environment``) and lives in ``astro-mine-sim`` (the Sim repo), injected
through this seam. Bench ships no Sim code and never imports Sim, so the base package stays
dep-clean (core + pydantic).

:func:`reference_episode_runner` is the always-available default: a **deterministic trace
fixture**, not a physics engine. It closes the policy loop over synthetic observations —
so a real Core :class:`~astro_mine.core.policy.Policy` genuinely drives it and swapping in
Sim is a drop-in — and folds the policy's decisions into a policy-sensitive score, so the
scoring path and the leaderboard are demonstrable offline with no Sim, no cloud, no account
(mirrors the harness's ``reference_runner`` deferral). Real physics comes from the injected
Sim runner; different policies may score identically under this fixture where they would
differ under Sim.

Backlog: RM-P0-BENCH-05 — astro-mine-bench#5
"""

from __future__ import annotations

from typing import Protocol

from astro_mine.bench.scenario import ResolvedScenario
from astro_mine.bench.scenario._hash import content_hash
from astro_mine.core.messages import (
    CommsObservationMask,
    Observation,
    SensorReading,
    StateSample,
)
from astro_mine.core.messages.model import Quat, Transform, Vec3
from astro_mine.core.policy import DecisionContext, Policy
from astro_mine.core.resource import FieldDistribution
from astro_mine.core.scoring import BeliefSnapshot, EpisodeTrace, ScoringContext
from astro_mine.core.units import MOON_BODY_FIXED

__all__ = [
    "REFERENCE_EPISODE_RUNNER_ID",
    "EpisodeRunner",
    "reference_episode_runner",
    "resolve_episode_runner_id",
]

#: The runner identity a :class:`~astro_mine.bench.metrics.Scorecard` records for the
#: dependency-clean :func:`reference_episode_runner` fixture — the ``--runner fixture`` default.
#: A score's runner is part of its identity (conventions.md §1.5; bench.md §2.1, §11 "signed
#: runner attestation"), so a fixture score and a Sim score are distinguishable by *provenance*,
#: not only by value. Parallels the harness's ``REFERENCE_RUNNER_ID`` on the ``Runner`` protocol.
REFERENCE_EPISODE_RUNNER_ID = "fixture/0.1.0"


class EpisodeRunner(Protocol):
    """Runs a policy on a resolved scenario for one seed, producing a scorable trace.

    A conforming runner MUST be a **pure, deterministic** function of
    ``(resolved.scenario_hash, policy, seed)`` — no wall-clock, no global RNG, no network —
    so a run reproduces byte-for-byte and Bench's determinism gate holds over scoring.

    A runner that finds it *cannot honestly score* this scenario MUST raise
    :class:`~astro_mine.bench.baseline.ScoringRefused` rather than return a degraded trace: a
    scorecard is a published claim, and a refusal that can be ignored will be. Callers present that
    exception as an actionable error; any other exception is a bug and keeps its traceback.
    """

    def __call__(self, resolved: ResolvedScenario, policy: Policy, seed: int) -> EpisodeTrace: ...


def resolve_episode_runner_id(runner: EpisodeRunner, runner_id: str | None) -> str:
    """The identity to stamp on a scorecard for ``runner`` (mirrors the harness resolver).

    An explicit ``runner_id`` always wins (the injected Sim runner supplies its own
    ``SIM_RUNNER_ID``); otherwise the built-in fixture resolves to
    :data:`REFERENCE_EPISODE_RUNNER_ID`, and any other runner falls back to its ``__name__``.
    """
    if runner_id is not None:
        return runner_id
    if runner is reference_episode_runner:
        return REFERENCE_EPISODE_RUNNER_ID
    return getattr(runner, "__name__", repr(runner))


# --- the reference trace fixture ----------------------------------------------------------

_AGENT = "rover"
_TICKS = 8
_DT_S = 3600.0  # 1-hour illustrative decision tick (a fixture cadence, not the anchor's 60 s)
_CELLS = ("psr-c0", "psr-c1", "psr-c2", "psr-c3")
_CELL_AREA_M2 = 100.0
_PRIOR_VARIANCE = 1.0
_CHARACTERIZED_VARIANCE = 0.05
_CHARACTERIZED_THRESHOLD = 0.1
_DETECTION_THRESHOLD = 0.2
_SPECIES = "water"  # the stored/produced resource the ISRU tank reports and the belief tracks
_DISCOVERY_SPECIES = "hydrogen"  # what the neutron spectrometer detects — not the stored water
_BATTERY_START_J = 1_000_000.0
_TEMPERATURE_K = 120.0
_SURVIVABLE_TEMPERATURE_K = 100.0
_MAX_WATER_KG = 500.0

_IDENTITY_POSE = Transform(
    translation_m=Vec3(x=0.0, y=0.0, z=0.0),
    rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
)


def _unit(payload: object) -> float:
    """A deterministic pseudo-scalar in ``[0, 1)`` from the canonical hash of ``payload``."""
    digest = content_hash(payload)
    hex_start = len("sha256:")
    return int(digest[hex_start : hex_start + 12], 16) / float(1 << 48)


def _discharge_per_tick(gain: float) -> float:
    """Per-tick battery discharge — a working agent (higher ``gain``) spends more energy."""
    return 2000.0 + gain * 3000.0


def _base_state(gain: float, tick: int) -> StateSample:
    """The gain-independent self-state a policy sees / a metric reads (battery, thermal)."""
    return StateSample(
        agent_id=_AGENT,
        frame=MOON_BODY_FIXED,
        pose=_IDENTITY_POSE,
        battery_soc_j=_BATTERY_START_J - tick * _discharge_per_tick(gain),
        temperature_k=_TEMPERATURE_K,
    )


def _comms(tick: int) -> CommsObservationMask:
    """A fixed relay/PSR contact pattern (comms environment, policy-independent)."""
    return CommsObservationMask(agent_id=_AGENT, earth_contact=tick % 2 == 0)


def _probe_observation(tick: int) -> Observation:
    """The gain-independent observation the policy decides against, to fingerprint it."""
    return Observation(
        tick=tick,
        sim_time_s=tick * _DT_S,
        agent_id=_AGENT,
        self_state=_base_state(0.0, tick),
        comms=_comms(tick),
    )


def _scored_observation(seed: int, tick: int, gain: float, detect_tick: int | None) -> Observation:
    """The scored observation: the probe channels plus gain-driven ISRU / detection readings."""
    sensors = [
        SensorReading(
            sensor="isru_tank",
            values=[round(_MAX_WATER_KG * gain * (tick + 1) / _TICKS, 3)],
            unit="kg",
            resource_species=_SPECIES,
        ),
        SensorReading(
            sensor="neutron",
            values=[gain if detect_tick is not None and tick >= detect_tick else 0.0],
            resource_species=_DISCOVERY_SPECIES,
        ),
    ]
    return Observation(
        tick=tick,
        sim_time_s=tick * _DT_S,
        agent_id=_AGENT,
        self_state=_base_state(gain, tick),
        sensors=sensors,
        comms=_comms(tick),
    )


def _scoring_context(gain: float) -> ScoringContext:
    """The scorer-only inputs (prospect.md §9): belief prior/history, PSR mask, night windows."""
    prior = {
        cell: FieldDistribution(mean=0.3, variance=_PRIOR_VARIANCE, species=_SPECIES, unit="kg")
        for cell in _CELLS
    }
    characterized = round(gain * len(_CELLS))
    posterior = {
        cell: FieldDistribution(
            mean=0.3,
            variance=_CHARACTERIZED_VARIANCE if index < characterized else _PRIOR_VARIANCE,
            species=_SPECIES,
            unit="kg",
        )
        for index, cell in enumerate(_CELLS)
    }
    return ScoringContext(
        prior_belief=prior,
        belief_history=(BeliefSnapshot(sim_time_s=(_TICKS - 1) * _DT_S, cells=posterior),),
        psr_cells=frozenset(_CELLS),
        cell_area_m2=_CELL_AREA_M2,
        characterized_variance_threshold=_CHARACTERIZED_THRESHOLD,
        discovery_species=_DISCOVERY_SPECIES,
        discovery_threshold=_DETECTION_THRESHOLD,
        water_species=_SPECIES,
        night_intervals=((0.0, (_TICKS - 1) * _DT_S),),
        survivable_temperature_k=_SURVIVABLE_TEMPERATURE_K,
    )


def reference_episode_runner(resolved: ResolvedScenario, policy: Policy, seed: int) -> EpisodeTrace:
    """Drive ``policy`` over a deterministic synthetic episode and return its scorable trace.

    Closes the Core decision loop — ``policy.decide({agent: obs}, context)`` each tick — over
    gain-independent probe observations, folds the emitted actions into a policy-sensitive
    ``gain``, then synthesizes the scored :class:`EpisodeTrace` (ISRU/detection readings, belief
    history) from ``gain`` and the seed. A fixture, not physics (see the module docstring).
    """
    context = DecisionContext(seed=seed, sim_time_s=0.0)
    fingerprint = f"{resolved.scenario_hash}:{seed}"
    for tick in range(_TICKS):
        batch = policy.decide({_AGENT: _probe_observation(tick)}, context)
        fingerprint = content_hash({"prev": fingerprint, "batch": batch.model_dump(mode="json")})

    gain = _unit({"seed": seed, "policy": fingerprint, "scenario": resolved.scenario_hash})
    detect_tick = round((1.0 - gain) * (_TICKS - 1)) if gain >= _DETECTION_THRESHOLD else None
    observations = tuple(
        _scored_observation(seed, tick, gain, detect_tick) for tick in range(_TICKS)
    )
    return EpisodeTrace(observations=observations, context=_scoring_context(gain))
