"""``PolicyShield`` as a Mind shield plugin — the RFC-0006 sibling binding (RM-P1-GUARD-03).

Registered under the ``astro_mine.mind.tier_plugins`` entry-point group by the ``[mind]`` extra, so
Mind's :class:`~astro_mine.mind.registry.registry.TierRegistry` discovers the **real** Guard TCB as
a drop-in replacement for its reference ``ConstraintShield``. Three things make the binding work:

1. **A shielded policy is a policy.** Mind's shield stage shields the *already-composed* proposal,
   which the executive threads through ``DecisionContext.upstream``
   (``astro_mine.mind.guardrail.shield.shield_egress``). :class:`PolicyShield` wraps a *policy*, so
   the adapter hands it an identity-over-``upstream`` policy: the shield then reads the composed
   batch exactly as it would read any wrapped planner's proposal. (RFC-0006 spells out this exact
   construction.)

2. **The report seam.** Mind's :class:`~astro_mine.mind.guardrail.report.ReportingShield` protocol
   lets a shield surface *what it did* without Mind importing Guard. :class:`GuardShield`
   implements it by draining Guard's own ``SafetyVerdict`` stream (RM-P1-GUARD-06) into a
   :class:`~astro_mine.mind.guardrail.report.ShieldReport` — so the trace's ``clauses`` are the
   invoked ``SafetySpec`` constraint ids (RFC-0004) and the ``certificate`` is the verdict's
   content hash, not an inference from a batch diff.

3. **Fail-closed construction.** The factory **requires** a ``SafetySpec`` (``spec_path`` or an
   inline ``spec``). A stack that binds ``guard.shield`` but authors no contract gets a loud
   ``ValueError``, never a silently non-enforcing shield — the one failure mode a safety plugin
   must not have.

The manifest (``manifests/shield.yaml``) is a Core :class:`PluginManifest`, so it passes the same
registry gates (validity → Core-interface negotiation → signature) as every other plugin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ControlMode
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry.loader import load_manifest as _load_core_manifest
from astro_mine.core.registry.model import PluginManifest
from astro_mine.guard.audit.model import SafetyVerdict
from astro_mine.guard.audit.sink import CollectingSink
from astro_mine.guard.spec import compile_spec, load_safety_spec
from astro_mine.guard.spec.compiler import DEFAULT_SAMPLE_PERIOD_S
from astro_mine.guard.wrap import ActionPolicy, CoreConfig, PolicyShield
from astro_mine.mind.guardrail.report import InterventionKind, ShieldReport
from astro_mine.mind.registry.registry import TierPlugin

__all__ = ["PLUGIN_NAME", "GuardShield", "guard_shield_plugin", "load_manifest"]

#: The registry name Mind's stack specs bind (``shield: {plugin: guard.shield}``).
PLUGIN_NAME = "guard.shield"

#: Guard's audit ``reason`` → Mind's :class:`InterventionKind`. ``shield_corrected`` is the CBF-QP
#: projection (a ``shield_edit``); ``monitor_fired`` is an STL/MTL breach; everything else that
#: intervened handed control to the verified simplex backup — including ``not_certifiable``, the
#: action gate's refusal of an action kind the TCB cannot certify.
_KIND_OF: dict[str, InterventionKind] = {
    "shield_corrected": InterventionKind.SHIELD_EDIT,
    "monitor_fired": InterventionKind.MONITOR_BREACH,
    "scalar_violated": InterventionKind.BACKUP_ACTIVATION,
    "qp_uncertifiable": InterventionKind.BACKUP_ACTIVATION,
    "bad_input": InterventionKind.BACKUP_ACTIVATION,
    "watchdog_expired": InterventionKind.BACKUP_ACTIVATION,
    "not_certifiable": InterventionKind.BACKUP_ACTIVATION,
}


class _UpstreamPolicy:
    """The identity-over-``upstream`` policy the shield wraps.

    Mind's shield stage receives the *composed* proposal in ``DecisionContext.upstream``, while
    :class:`PolicyShield` wraps a *policy* and calls it. This one-line adapter is the whole of the
    impedance mismatch: it returns the upstream batch, so the shield reads it exactly as it reads
    any
    wrapped planner's proposal — and treats it, correctly, as adversarial input."""

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        return context.upstream if context.upstream is not None else ActionBatch()


class GuardShield:
    """Guard's :class:`PolicyShield` behind Mind's shield + :class:`ReportingShield` seams.

    A Core :class:`~astro_mine.core.policy.protocol.Policy` (so Mind's executive can bind it as the
    shield stage) that additionally answers :meth:`report` — the structured account of what the TCB
    did on the most recent ``decide``, drained from Guard's own ``SafetyVerdict`` stream rather
    than
    inferred from a batch diff."""

    def __init__(self, shield: PolicyShield, sink: CollectingSink) -> None:
        self._shield = shield
        self._sink = sink
        self._last_report: ShieldReport | None = None

    @property
    def shield(self) -> PolicyShield:
        """The wrapped :class:`PolicyShield` (its spec id / content hashes are the provenance)."""
        return self._shield

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        before = len(self._sink.verdicts)
        batch = self._shield.decide(observations, context)
        self._last_report = _report(self._sink.verdicts[before:])
        return batch

    def report(self) -> ShieldReport | None:
        """The :class:`ShieldReport` for the most recent ``decide`` (``None`` before the first)."""
        return self._last_report


def _report(verdicts: Sequence[SafetyVerdict]) -> ShieldReport:
    """Fold this tick's per-agent verdicts into one Mind :class:`ShieldReport`.

    A tick intervened if *any* agent's action was touched. The reported ``kind`` is the most severe
    one seen (a backup activation outranks a monitor breach outranks a shield edit), so a report
    can never make a tick look gentler than it was. ``clauses`` are the union of the invoked
    ``SafetySpec`` constraint ids; ``certificate`` is the first intervening verdict's content hash
    — the handle into
    the MCAP verdict stream that carries the full record (RM-P1-GUARD-06)."""
    intervened = [v for v in verdicts if v.intervention != "none"]
    if not intervened:
        return ShieldReport(intervened=False)
    severity = {
        InterventionKind.SHIELD_EDIT: 0,
        InterventionKind.MONITOR_BREACH: 1,
        InterventionKind.BACKUP_ACTIVATION: 2,
    }
    kinds = [_KIND_OF.get(v.reason, InterventionKind.BACKUP_ACTIVATION) for v in intervened]
    clauses = sorted({cid for v in intervened for cid in v.constraint_ids})
    return ShieldReport(
        intervened=True,
        kind=max(kinds, key=lambda k: severity[k]),
        clauses=tuple(clauses),
        certificate=intervened[0].content_hash(),
    )


def load_manifest() -> PluginManifest:
    """The Core plugin manifest Mind's registry gates this shield through."""
    text = (
        resources.files("astro_mine.guard.mind")
        .joinpath("manifests/shield.yaml")
        .read_text(encoding="utf-8")
    )
    return _load_core_manifest(text).manifest


def _build(params: Mapping[str, Any]) -> GuardShield:
    """Build the real Guard shield from a Mind stack spec's ``params``.

    Recognised params — ``spec_path`` (a ``SafetySpec`` YAML/JSON file) **or** ``spec`` (the
    document inline), ``sample_period_s``, the core ceilings (``u_max`` / ``v_max`` / ``k_brake`` /
    ``max_history_cap`` / ``deadline_us``), and the action gate's *narrowing* knobs
    (``certified_modes`` / ``certified_tasks``) plus the rejected-action channel
    (``fallback_control_mode`` / ``fallback_target``).

    **A shield with no contract is not a shield.** Omitting the spec raises rather than defaulting
    to something permissive: the entire value of binding the real Guard is that the constraints are
    the reviewed, content-addressed ones.

    **``certified_modes`` / ``certified_tasks`` here can only NARROW the reviewed grant, never
    create one** (RFC-0004 Amendment 2). The MODE/TASK allowlist the gate enforces is
    ``spec.admissible_directives ∩ these params``, so a stack that lists a directive its
    ``SafetySpec`` does not admit gets **nothing** — it must author the grant in the spec, where it
    is reviewed, content-addressed, signed, and bound into every ``SafetyVerdict``. These params
    remain useful for running a deployment *stricter* than its contract.

    They also gate ``ModeCommand.mode`` **names**, not
    :class:`~astro_mine.core.messages.enums.ControlMode` values: an ``ACTUATOR`` action in any
    control mode is a *numeric* command and goes to the shield, never to this allowlist. Listing a
    control-mode name here does nothing for actuator commands — and would be a standing grant for
    any ``MODE`` directive that happened to share the name."""
    if "spec_path" in params:
        text = Path(str(params["spec_path"])).read_text(encoding="utf-8")
    elif "spec" in params:
        text = str(params["spec"])
    else:
        raise ValueError(
            f"the {PLUGIN_NAME!r} plugin requires a SafetySpec: pass params.spec_path (a file) or "
            "params.spec (the document inline). Guard never enforces a default contract."
        )
    compiled = compile_spec(
        load_safety_spec(text),
        sample_period_s=float(params.get("sample_period_s", DEFAULT_SAMPLE_PERIOD_S)),
    )

    policy = ActionPolicy(
        certified_modes=frozenset(params.get("certified_modes", ())),
        certified_tasks=frozenset(params.get("certified_tasks", ())),
        fallback_control_mode=ControlMode(params.get("fallback_control_mode", "effort")),
        fallback_target=str(params.get("fallback_target", "body")),
    )
    config_keys = ("u_max", "v_max", "k0", "k1", "k_brake", "max_history_cap", "deadline_us")
    overrides = {k: params[k] for k in config_keys if k in params}
    core_config = CoreConfig(action_policy=policy, **overrides)

    # The verdict sink is Guard's own audit stream (RM-P1-GUARD-06); the adapter drains it into the
    # Mind report. Best-effort by contract — a telemetry fault can never change a certified action.
    sink = CollectingSink()
    shield = PolicyShield(
        _UpstreamPolicy(),
        compiled,
        sink=sink,
        watchdog=bool(params.get("watchdog", False)),
        core_config=core_config,
    )
    return GuardShield(shield, sink)


def guard_shield_plugin() -> TierPlugin:
    """The entry-point provider: Guard's real ``PolicyShield`` as a Mind shield plugin (RFC-0006).

    Advertised under ``astro_mine.mind.tier_plugins`` by the ``[mind]`` extra;
    ``TierRegistry.from_entry_points()`` resolves it, gates the manifest through Core, and
    ``instantiate("guard.shield", params)`` builds the shield."""
    return TierPlugin(manifest=load_manifest(), factory=_build)
