"""Declarative ``CommsModel`` configuration (RM-P1-LEARN-02).

The comms regime is described *declaratively* — a Pydantic v2 document validated by JSON
Schema — so it is the reproducibility key recorded with a run (learn.md §3; conventions.md
§5, §11) and can be replayed byte-for-byte. Four composable stages describe the channel a
message crosses each tick:

1. :class:`RangeGateConfig` — line-of-sight / range / link-quality gating. When a
   :class:`~astro_mine.core.messages.model.CommsObservationMask` from
   [Link](../../../../../docs/architecture/link.md) is present it supplies the reachable
   set (LOS already solved); the gate only *tightens* it (range, margin, latency). Without
   Link, the gate synthesizes reachability from neighbour geometry — the same knobs.
2. :class:`BandwidthBudgetConfig` — a per-agent message budget per tick; links beyond it
   are shed (lowest priority first).
3. :class:`DropConfig` — an independent per-link Bernoulli drop.
4. :class:`DelayConfig` — a fixed or sampled delivery delay; a peer is seen through its
   most recently *delivered* (possibly stale) message.

Every field has an inert default, so ``CommsModelConfig()`` is an identity channel — it
degrades nothing and preserves the un-wrapped rollout (asserted in the tests).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "BandwidthBudgetConfig",
    "CommsModelConfig",
    "DelayConfig",
    "DropConfig",
    "RangeGateConfig",
]

#: Fixed salt mixed into the run seed so the comms RNG stream is *independent* of the
#: environment's own RNG advancement (same config + seed ⇒ identical drop/delay realization
#: regardless of what the policy does). Bump only on an intentional realization change.
_DEFAULT_SEED_SALT = 0x00C0_0115

_Prob = Annotated[float, Field(ge=0.0, le=1.0)]
_NonNegFloat = Annotated[float, Field(ge=0.0)]
_NonNegInt = Annotated[int, Field(ge=0)]


class _Config(BaseModel):
    """Frozen, ``extra``-forbidding base — a config typo fails loudly rather than silently
    disabling a channel stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class RangeGateConfig(_Config):
    """Line-of-sight / range / link-quality gating.

    ``honor_reachable`` keeps Link's own ``reachable`` verdict (its LOS/occlusion solve) as
    the starting set; the remaining bounds only remove links. ``max_range_m`` is evaluated
    against the Euclidean distance to the peer's observed pose and is the sole reachability
    source when no Link mask is present (the synthetic stand-in)."""

    honor_reachable: bool = True
    max_range_m: _NonNegFloat | None = None
    min_margin_db: float | None = None
    max_latency_s: _NonNegFloat | None = None


class BandwidthBudgetConfig(_Config):
    """A per-agent outbound message budget per tick.

    Each admitted link consumes ``message_bits``; once ``per_agent_bits_per_tick`` is
    exhausted the remaining candidate links are shed (a *budget* drop, distinct from a
    stochastic loss). ``priority`` orders which links are admitted first; ties and the
    ``peer_order`` policy break deterministically on peer id."""

    per_agent_bits_per_tick: _NonNegFloat | None = None
    message_bits: Annotated[float, Field(gt=0.0)] = 1.0
    priority: Literal["peer_order", "margin_db", "rate_bps", "latency_s"] = "margin_db"


class DropConfig(_Config):
    """Independent per-link Bernoulli message loss, drawn once per link per tick."""

    probability: _Prob = 0.0


class DelayConfig(_Config):
    """Delivery delay in whole ticks — fixed or sampled per delivered message.

    A delivered message about a peer is buffered and released ``d`` ticks later; the agent
    observes that peer through its most recently *released* message, so a link that is down
    this tick still shows the last-known (stale) neighbour. ``max_ticks`` bounds buffer
    memory. ``kind``:

    - ``none`` — immediate delivery (``d = 0``).
    - ``fixed`` — constant ``ticks``.
    - ``uniform`` — integer uniform on ``[low, high]``.
    - ``geometric`` — ``min(Geometric(1 / (1 + mean_ticks)) - 1, max_ticks)``.
    """

    kind: Literal["none", "fixed", "uniform", "geometric"] = "none"
    ticks: _NonNegInt = 0
    low: _NonNegInt = 0
    high: _NonNegInt = 0
    mean_ticks: _NonNegFloat = 0.0
    max_ticks: Annotated[int, Field(ge=0, le=1024)] = 16

    @model_validator(mode="after")
    def _check_bounds(self) -> DelayConfig:
        if self.kind == "uniform" and self.high < self.low:
            raise ValueError("uniform delay requires high >= low")
        if self.kind == "fixed" and self.ticks > self.max_ticks:
            raise ValueError("fixed delay ticks exceeds max_ticks")
        return self


class CommsModelConfig(_Config):
    """The full declarative comms regime — the reproducibility key recorded with a run.

    Emit its JSON Schema with :meth:`model_json_schema` and round-trip it through JSON with
    :meth:`model_dump_json` / :meth:`model_validate_json` (both exercised in the tests)."""

    range_gate: RangeGateConfig = RangeGateConfig()
    bandwidth: BandwidthBudgetConfig = BandwidthBudgetConfig()
    drop: DropConfig = DropConfig()
    delay: DelayConfig = DelayConfig()
    seed_salt: int = _DEFAULT_SEED_SALT
    #: Bumped when the *meaning* of the config or the realization algorithm changes.
    schema_version: Literal["0.1.0"] = "0.1.0"

    @property
    def is_identity(self) -> bool:
        """True when the config degrades nothing — used to short-circuit to the un-wrapped
        rollout and to assert the null-model invariant."""
        return (
            self.range_gate == RangeGateConfig()
            and self.bandwidth == BandwidthBudgetConfig()
            and self.drop.probability == 0.0
            and self.delay.kind == "none"
        )
