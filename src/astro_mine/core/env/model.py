# SPDX-License-Identifier: Apache-2.0
"""Environment API v0.1 — result containers (RM-P0-CORE-02).

The typed returns of the Core Environment contract. Multi-agent by construction:
``reset``/``step`` yield per-agent maps keyed by agent id, with the single-agent case
the degenerate one-key form. Observations are the canonical hot-path message
(:class:`astro_mine.core.messages.Observation`, which carries the per-tick
:class:`~astro_mine.core.messages.CommsObservationMask`); actions are an
:class:`~astro_mine.core.messages.ActionBatch`.

These are lightweight in-memory returns, **not** wire documents — frozen dataclasses,
not Pydantic models, so the per-tick ``step`` path does not re-validate the already
validated nested message models. The reproducibility/serialization surface is the
nested Observation/Action messages (and Sim's MCAP recording), not this container.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from astro_mine.core.messages.model import Observation
from astro_mine.core.sadf.enums import Regime

__all__ = ["AgentId", "Info", "ResetResult", "StepResult"]

#: A swarm agent's stable identifier (matches ``Observation.agent_id`` / SADF identity).
AgentId = str

#: Free-form per-agent step metadata (the Gymnasium/PettingZoo ``info`` dict). Sim may
#: record fidelity/error provenance here; Core defines no schema for it in v0.1.
Info = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ResetResult:
    """The return of :meth:`Environment.reset` — the initial per-agent observations and
    infos (the Gymnasium/PettingZoo ``reset() -> (obs, info)`` shape, multi-agent)."""

    observations: Mapping[AgentId, Observation]
    infos: Mapping[AgentId, Info] = field(default_factory=dict)
    # RFC-0001 reserved (RM-P1-CORE-04): the bounded regime dimension the Environment API
    # gains so a consumer can branch or refuse per phase (mission-model.md §2.2). Optional
    # and unset by default — no P1 consumer reads it; a single-regime (surface) run leaves
    # it None and behaves exactly as before. Additive, no mechanism.
    regime: Regime | None = None


@dataclass(frozen=True, slots=True)
class StepResult:
    """The return of :meth:`Environment.step` — the PettingZoo-parallel five-tuple as
    named per-agent fields, plus first-class variable-timestep state.

    ``rewards`` defaults empty: v0.1 is **reward-free by default** (scoring is
    trace-based via ObjectiveSpec/Bench); an env MAY populate it. ``terminations`` mark
    a terminal condition (e.g. a battery-floor failure), ``truncations`` a horizon /
    time-limit cutoff — both per agent. ``sim_time_s`` is the episode time after the
    step and ``dt_s`` the advanced duration, making the **variable timestep** explicit
    (per-engine multi-rate sub-stepping stays internal to the implementor).
    """

    observations: Mapping[AgentId, Observation]
    sim_time_s: float
    rewards: Mapping[AgentId, float] = field(default_factory=dict)
    terminations: Mapping[AgentId, bool] = field(default_factory=dict)
    truncations: Mapping[AgentId, bool] = field(default_factory=dict)
    infos: Mapping[AgentId, Info] = field(default_factory=dict)
    dt_s: float | None = None
    # RFC-0001 reserved (RM-P1-CORE-04): the current phase's regime, so a multi-regime
    # consumer can branch or refuse (mission-model.md §2.2). Optional and unset by default
    # — existing single-regime consumers ignore it and run unchanged. Additive, no mechanism.
    regime: Regime | None = None
