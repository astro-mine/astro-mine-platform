# SPDX-License-Identifier: Apache-2.0
"""PolicyShield — implements the Core Policy/Planner API and wraps any policy (RM-P1-GUARD-03).

The transparent runtime-assurance wrapper (guard.md §3, §6): :class:`PolicyShield` *is* a
:class:`~astro_mine.core.policy.protocol.Policy`, so a shielded Mind planner / Allocate allocation /
Learn ONNX policy drops into Sim / Studio / Ops with **no consumer-side Guard code**. Each tick it
reads only the wrapped policy's proposed action (adversarial input), routes it through the trusted
Rust safety core, and re-emits the certified action — fail-safe, never fail-open. All certification
stays in the core; the wrapper is pure marshalling (no trust surface).

**Every** action crosses the core (LUNAR-FR-006): ``EFFORT`` / ``VELOCITY`` / ``POSITION`` actuator
setpoints are projected onto the reviewed kinematic envelope ∩ the keep-out safe set; ``MODE`` /
``TASK`` directives are gated against a certified allowlist (empty by default); and an actuator
command in a control mode the TCB has no plant model for (``IMPEDANCE`` / ``TRAJECTORY``) is
**rejected** — answered with a verified safe command, never passed through.

Public API:

- :class:`PolicyShield` — the wrapper (and its :class:`CoreConfig` core-tuning);
- :class:`ActionPolicy` — the certified ``MODE``/``TASK`` allowlist + the rejected-action channel;
- :class:`SignalResolver` — the signal-resolution seam, with the minimal
  :class:`DefaultSignalResolver` / :class:`MappingSignalResolver` (GUARD-04 fills real resolution);
- :class:`ActionCodec` + :data:`MODELLED_CONTROL_MODES` / :data:`ACCEL_CONTROL_MODE` — the
  Action ↔ command convention (the shield's coverage);
- :data:`SHIELD_CORE_INTERFACES` — the Core interfaces the shield claims (contract test).
"""

from __future__ import annotations

from astro_mine.guard.wrap.shield import (
    ACCEL_CONTROL_MODE,
    MODELLED_CONTROL_MODES,
    SHIELD_CORE_INTERFACES,
    ActionCodec,
    ActionPolicy,
    CoreConfig,
    DefaultSignalResolver,
    MappingSignalResolver,
    PolicyShield,
    SignalResolver,
)

__all__ = [
    "ACCEL_CONTROL_MODE",
    "MODELLED_CONTROL_MODES",
    "SHIELD_CORE_INTERFACES",
    "ActionCodec",
    "ActionPolicy",
    "CoreConfig",
    "DefaultSignalResolver",
    "MappingSignalResolver",
    "PolicyShield",
    "SignalResolver",
]
