"""Type stub for the trusted Rust safety core (RM-P1-GUARD-02).

The compiled PyO3 extension (`rust/`, built by maturin into `astro_mine.guard._core`). This stub is
the typed Python view of the core; the implementation is the Rust TCB and the exhaustive tests live
in `rust/tests/`. See ``docs/architecture/guard.md`` sections 3-4 and 9.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

class Verdict(TypedDict):
    """The auditable per-tick output of the arbiter."""

    #: The single certified command, in ``certified_control_mode``. Empty for a certified
    #: ``MODE``/``TASK`` directive, which carries no numeric command.
    certified_action: list[float]
    #: The control mode ``certified_action`` is expressed in: ``"effort"`` | ``"velocity"`` |
    #: ``"position"``. A rejected actuator command is answered in **its own** channel; a rejected
    #: directive in the configured ``fallback_control_mode``.
    certified_control_mode: str
    #: Which layer produced it: ``"primary"`` | ``"shield"`` | ``"backup"``.
    layer: str
    #: What Guard did to the proposal: ``"none"`` | ``"modified"`` | ``"fallback"``.
    intervention: str
    #: Audit reason: ``certified`` | ``shield_corrected`` | ``scalar_violated`` | ``monitor_fired``
    #: | ``qp_uncertifiable`` | ``bad_input`` | ``watchdog_expired`` | ``not_certifiable`` (the
    #: action gate refused the *kind* of action proposed).
    reason: str
    #: Constraint ids that fired this tick (empty unless a detect layer tripped).
    fired: list[str]
    #: The backup behaviour, if the arbiter fell back: ``brake_to_stop`` | ``hold`` |
    #: ``safe_state``; ``None`` when the primary/shield path certified.
    backup_kind: str | None
    #: Smallest keep-out barrier margin this tick (the safety certificate); ``+inf`` if none.
    min_barrier_margin: float

class SafetyCore:
    """The trusted safety core: decode a compiled model once, certify one action per tick."""

    @property
    def spatial_dim(self) -> int: ...
    @property
    def spec_id(self) -> str: ...
    @property
    def spec_content_hash(self) -> str: ...
    @property
    def v_max(self) -> float | None:
        """The enforced commanded-speed ceiling (the reviewed spec's ``max_velocity_mps``, else the
        configured ``v_max``); ``None`` for a non-spatial model, which has no shield."""

    @staticmethod
    def from_wire(
        compiled_wire: bytes,
        *,
        u_max: float = ...,
        v_max: float = ...,
        k0: float = ...,
        k1: float = ...,
        k_brake: float = ...,
        predictive_horizon_samples: int = ...,
        deadline_us: int | None = ...,
        max_history_cap: int = ...,
        certified_modes: Sequence[str] | None = ...,
        certified_tasks: Sequence[str] | None = ...,
        fallback_control_mode: str = ...,
    ) -> SafetyCore:
        """Load a core from a ``CompiledSafetyModel`` protobuf payload (``compiled_to_wire``).

        ``certified_modes`` / ``certified_tasks`` are the action gate's allowlists (empty ⇒ no
        directive is certified); ``fallback_control_mode`` is the channel a rejected directive is
        answered in. An unknown ``fallback_control_mode`` raises — a misconfigured core must fail
        loud at construction, never mid-episode."""

    def step(
        self,
        signals: list[float],
        position: list[float],
        velocity: list[float],
        proposed_action: list[float],
        *,
        action_kind: str = ...,
        directive: str | None = ...,
        watchdog: bool = ...,
    ) -> Verdict:
        """Certify one action.

        ``action_kind`` classifies the proposal for the action gate: ``effort`` | ``velocity`` |
        ``position`` (modelled commands carried in ``proposed_action``), ``mode`` | ``task``
        (discrete directives named by ``directive``), or ``opaque`` (an actuator command the core
        has no plant model for). **Any unrecognised token is treated as ``opaque``** — the core
        substitutes a verified safe command rather than reason about an action it cannot model.

        ``watchdog=True`` arms the real-time per-tick deadline."""
