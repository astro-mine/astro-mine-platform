"""First-cut rule-based multi-fidelity scheduler with error tracking (RM-P0-SIM-05).

The scheduler answers "which fidelity tier runs each agent this episode, and what error does
that choice imply" — by *rule*, not yet by a calibrated error budget (the error-budget-driven
scheduler that auto-selects the cheapest tier meeting a tolerance is Phase 1, sim.md §11).

A :class:`FidelityPolicy` (carried on the scenario) declares the rule; :class:`Scheduler`
resolves it against each agent's available :class:`~astro_mine.core.sadf.model.FidelityProfile`
ladder — sourced from Fleet's multi-fidelity profiles (RM-P0-FLEET-05) when present, else the
single tier the agent's RM-P0-SIM-03 engine declares — and emits a :class:`FidelitySelection`
per agent. Each selection carries:

- the chosen ``tier`` and its ``determinism_class`` (the input to the RM-P0-SIM-10
  determinism gate);
- the ``reference_tier`` (the finest available) and ``implied_error_rungs`` — how many rungs
  coarser the choice is than that reference. This is the **rule-based stand-in for the error
  implied by the substitution**; a numeric, oracle-validated error budget is the Phase-1
  error-budget scheduler.

A run can be **pinned** to a tier (per-agent or run-wide) for reproducibility; pinning a tier
an agent does not offer fails loudly. The selections are recorded into the run provenance
(:func:`~astro_mine.sim.runtime.run_episode`), the same envelope RM-P0-SIM-09's MCAP carries
as its "error-budget outcomes". The Phase-0 engines are single-tier, so a selection is
*recorded and pinnable* but does not yet swap the engine binding — that binding grows as
engines gain fidelity ladders (Phase 1); the policy/mechanism is in place here.

Backlog: RM-P0-SIM-05 -- astro-mine-sim#5
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.provenance.model import ErrorBudgetOutcome
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier
from astro_mine.core.sadf.model import FidelityProfile
from astro_mine.sim.engines.brax import (
    BRAX_CONTACT_ENGINE_DESCRIPTOR,
    MJX_CONTACT_ENGINE_DESCRIPTOR,
)
from astro_mine.sim.engines.dem import DEM_GRANULAR_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.granular import GRANULAR_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.manipulation import MANIPULATION_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.mobility import MOBILITY_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.mujoco import MUJOCO_MOBILITY_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.orbital import ORBITAL_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.orekit import OREKIT_ORBITAL_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.reference import KINEMATIC_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from astro_mine.sim.engines.adapter import EngineDescriptor
    from astro_mine.sim.runtime.scenario import AgentSpec, Scenario

__all__ = [
    "FidelityPolicy",
    "FidelityRule",
    "FidelitySelection",
    "Scheduler",
    "select_fidelity",
]

#: Coarsest → finest fidelity order — the substitution ladder a choice is measured against
#: (shared with the coupler). Substituting a coarser tier for a finer one is "more error".
_TIER_ORDER: tuple[FidelityTier, ...] = (
    FidelityTier.MASSMODEL,
    FidelityTier.KINEMATIC,
    FidelityTier.ARTICULATED,
    FidelityTier.SURROGATE,
)


def _engine_profile(descriptor: EngineDescriptor) -> FidelityProfile:
    """The single-tier profile an RM-P0-SIM-03 engine declares (the default ladder rung)."""
    return FidelityProfile(
        tier=descriptor.fidelity.tier, determinism_class=descriptor.determinism_class
    )


#: ``dynamics.kind`` → the default single-tier ladder, read straight off the engine that
#: owns the regime so the scheduler's defaults never drift from the engines' declarations.
_KIND_DEFAULT_PROFILE: dict[str, FidelityProfile] = {
    "kinematic": _engine_profile(KINEMATIC_ENGINE_DESCRIPTOR),
    "orbital": _engine_profile(ORBITAL_ENGINE_DESCRIPTOR),
    "mobility": _engine_profile(MOBILITY_ENGINE_DESCRIPTOR),
    "manipulation": _engine_profile(MANIPULATION_ENGINE_DESCRIPTOR),
    "granular": _engine_profile(GRANULAR_ENGINE_DESCRIPTOR),
    "dem_granular": _engine_profile(DEM_GRANULAR_ENGINE_DESCRIPTOR),
    "brax_contact": _engine_profile(BRAX_CONTACT_ENGINE_DESCRIPTOR),
    "mjx_contact": _engine_profile(MJX_CONTACT_ENGINE_DESCRIPTOR),
    "orekit_orbital": _engine_profile(OREKIT_ORBITAL_ENGINE_DESCRIPTOR),
    "mujoco_mobility": _engine_profile(MUJOCO_MOBILITY_ENGINE_DESCRIPTOR),
}


class FidelityRule(StrEnum):
    """How the scheduler picks a tier from an agent's available ladder when it is not pinned.

    ``COARSEST`` is the cheapest tier (the documented default for large sweeps); ``FINEST``
    is the most accurate."""

    COARSEST = "coarsest"
    FINEST = "finest"


class FidelityPolicy(BaseModel):
    """A declarative, rule-based fidelity request — the first-cut scheduler input.

    Resolution per agent: an explicit ``agent_pins`` entry wins; else a run-wide
    ``pinned_tier`` (the "pin a run to one tier for reproducibility" lever) wins; else
    ``rule`` selects from the agent's available profiles. A pinned tier the agent does not
    offer is rejected loudly (fail-fast, conventions.md §1)."""

    model_config = ConfigDict(extra="forbid")

    rule: FidelityRule = FidelityRule.COARSEST
    pinned_tier: FidelityTier | None = None
    agent_pins: dict[str, FidelityTier] = Field(default_factory=dict)
    #: Per-output-channel error tolerance the task accepts (RM-P1-SIM-03). When set, the scheduler
    #: runs **error-budget-driven** (sim.md §11): a tier carrying a declared per-channel budget (a
    #: [Surrogate](surrogate.md) tier, via its manifest ``recommended_error_budget``) is admitted
    #: only while every channel's declared budget is at or below this tolerance; otherwise the
    #: scheduler falls back to the high-fidelity reference. Unset ⇒ the rule-based path (unchanged).
    error_budget: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class FidelitySelection:
    """The tier chosen for one agent, plus the *implied error* of that choice.

    ``implied_error_rungs`` is how many rungs coarser than ``reference_tier`` (the finest
    available) the selection is — ``0`` means the finest available tier was chosen. It is the
    rule-based stand-in for a calibrated error budget; the numeric, oracle-validated budget is
    the Phase-1 error-budget scheduler. ``determinism_class`` is the chosen tier's class, the
    input to the RM-P0-SIM-10 determinism gate."""

    agent_id: str
    tier: FidelityTier
    reference_tier: FidelityTier
    determinism_class: DeterminismClass
    implied_error_rungs: int
    pinned: bool
    #: The declared per-channel budget of the chosen tier (RM-P1-SIM-03) — present only when the
    #: policy is error-budget-driven and the choice is a budgeted (surrogate) tier; ``None`` on the
    #: rule-based path or when the reference was chosen.
    admitted_budget: dict[str, float] | None = None
    #: The task tolerance the choice was admitted against (the policy's ``error_budget``), when
    #: error-budget-driven — else ``None``.
    tolerance: dict[str, float] | None = None
    #: Whether a budgeted tier was admitted within tolerance (``True`` when the surrogate met
    #: budget; ``False`` when it was rejected and the run fell back to the reference).
    within_budget: bool = True

    def as_provenance(self) -> dict[str, object]:
        """The JSON-able form stamped into the run provenance (RM-P0-SIM-09 carries it)."""
        record: dict[str, object] = {
            "tier": self.tier.value,
            "reference_tier": self.reference_tier.value,
            "determinism_class": self.determinism_class.value,
            "implied_error_rungs": self.implied_error_rungs,
            "pinned": self.pinned,
        }
        # Additive: the error-budget verdict only appears when the scheduler ran budget-driven, so
        # the rule-based provenance (and its pinned golden hash) is byte-identical.
        if self.tolerance is not None:
            record["within_budget"] = self.within_budget
            record["tolerance"] = dict(self.tolerance)
            if self.admitted_budget is not None:
                record["admitted_budget"] = dict(self.admitted_budget)
        return record

    def error_budget_outcomes(self) -> list[ErrorBudgetOutcome]:
        """Per-channel :class:`ErrorBudgetOutcome`s for the Parquet report + run provenance.

        One row per channel of the task tolerance: the surrogate's declared budget (``value``) vs
        the tolerance, with ``within_budget`` the admission verdict. Empty on the rule-based path.
        """
        if self.tolerance is None:
            return []
        budget = self.admitted_budget or {}
        return [
            ErrorBudgetOutcome(
                name=f"{self.agent_id}:{channel}",
                within_budget=self.within_budget,
                tier=self.tier.value,
                metric=channel,
                value=budget.get(channel),
                tolerance=tolerance,
            )
            for channel, tolerance in sorted(self.tolerance.items())
        ]


def _base_selection(
    agent_id: str,
    chosen: FidelityTier,
    reference: FidelityTier,
    by_tier: dict[FidelityTier, FidelityProfile],
    *,
    pinned: bool,
) -> FidelitySelection:
    """A rule-based / pinned selection (no error-budget verdict)."""
    return FidelitySelection(
        agent_id=agent_id,
        tier=chosen,
        reference_tier=reference,
        determinism_class=by_tier[chosen].determinism_class,
        implied_error_rungs=_TIER_ORDER.index(reference) - _TIER_ORDER.index(chosen),
        pinned=pinned,
    )


def _within(budget: Mapping[str, float], tolerance: Mapping[str, float]) -> bool:
    """Whether a declared per-channel ``budget`` meets every channel of ``tolerance``.

    Admissible only if the tier declares a budget for **every** requested channel and each is at or
    below the tolerance — a channel with no declared budget is never admissible (conservative,
    surrogate.md §6: the surrogate never claims accuracy it did not measure)."""
    return all(channel in budget and budget[channel] <= tol for channel, tol in tolerance.items())


def _budget_selection(
    agent_id: str,
    by_tier: dict[FidelityTier, FidelityProfile],
    tolerance: Mapping[str, float],
    budgeted: dict[FidelityTier, Mapping[str, float]],
) -> FidelitySelection:
    """Admit the cheapest budgeted tier meeting ``tolerance``, else fall back to the ground truth.

    The **ground-truth reference** is the finest *non-budgeted* tier — the high-fidelity engine (the
    DEM oracle) a learned tier is validated against and substitutes for. A budgeted (surrogate) tier
    is admitted only while its declared error stays inside the task tolerance (sim.md §10
    surrogate-error gate); otherwise the run escalates to that reference. On fallback the rejected
    tier's declared budget is still recorded so the error-budget report shows *why* it fell back.
    The numeric budget — not the rung gap — is the error signal here, so ``implied_error_rungs`` is
    ``0`` (a budget selection carries a calibrated verdict, not a rung stand-in)."""
    ground_truth = max(
        (tier for tier in by_tier if tier not in budgeted),
        key=_TIER_ORDER.index,
        default=max(budgeted, key=_TIER_ORDER.index),
    )
    admissible = sorted(
        (tier for tier, budget in budgeted.items() if _within(budget, tolerance)),
        key=_TIER_ORDER.index,
    )
    chosen = admissible[0] if admissible else ground_truth
    reported_budget = (
        budgeted[admissible[0]] if admissible else budgeted[min(budgeted, key=_TIER_ORDER.index)]
    )
    return FidelitySelection(
        agent_id=agent_id,
        tier=chosen,
        reference_tier=ground_truth,
        determinism_class=by_tier[chosen].determinism_class,
        implied_error_rungs=0,
        pinned=False,
        admitted_budget=dict(reported_budget),
        tolerance=dict(tolerance),
        within_budget=bool(admissible),
    )


def select_fidelity(
    agent_id: str,
    available: Sequence[FidelityProfile],
    policy: FidelityPolicy,
    *,
    tier_budgets: Mapping[FidelityTier, Mapping[str, float]] | None = None,
) -> FidelitySelection:
    """Resolve ``policy`` to a tier for one agent over its ``available`` ladder.

    The finest available tier is the reference. A pin wins; else, when ``policy.error_budget`` is
    set **and** some available tier carries a declared budget (``tier_budgets`` — a surrogate tier's
    ``recommended_error_budget``), the scheduler runs **error-budget-driven** (admit cheapest tier
    within tolerance, else fall back to the reference); otherwise the rung ``rule`` chooses. A pin
    to a tier the agent does not offer raises :class:`ValueError`."""
    if not available:
        raise ValueError(f"agent {agent_id!r} has no available fidelity profiles to select from")
    by_tier = {profile.tier: profile for profile in available}
    reference = max(by_tier, key=_TIER_ORDER.index)

    pinned_tier = policy.agent_pins.get(agent_id, policy.pinned_tier)
    if pinned_tier is not None:
        if pinned_tier not in by_tier:
            offered = [tier.value for tier in by_tier]
            raise ValueError(
                f"cannot pin agent {agent_id!r} to fidelity tier {pinned_tier.value!r}: "
                f"its available tiers are {offered}"
            )
        return _base_selection(agent_id, pinned_tier, reference, by_tier, pinned=True)

    if policy.error_budget is not None:
        budgeted = {
            tier: tier_budgets[tier]
            for tier in by_tier
            if tier_budgets is not None and tier in tier_budgets
        }
        if budgeted:
            return _budget_selection(agent_id, by_tier, policy.error_budget, budgeted)

    chosen = (
        reference if policy.rule is FidelityRule.FINEST else min(by_tier, key=_TIER_ORDER.index)
    )
    return _base_selection(agent_id, chosen, reference, by_tier, pinned=False)


def _available_profiles(spec: AgentSpec) -> tuple[FidelityProfile, ...]:
    """An agent's available ladder: its declared Fleet profiles (RM-P0-FLEET-05) if any, else
    the single tier its regime engine declares."""
    if spec.fidelity_profiles:
        return spec.fidelity_profiles
    return (_KIND_DEFAULT_PROFILE[spec.dynamics.kind],)


class Scheduler:
    """Resolves a :class:`FidelityPolicy` over a scenario's agents (RM-P0-SIM-05, RM-P1-SIM-03).

    ``tier_budgets`` supplies the declared per-channel error budget of any budgeted tier (a
    surrogate tier's manifest ``recommended_error_budget``), so an error-budget policy can admit or
    fall back per task tolerance."""

    def __init__(
        self,
        policy: FidelityPolicy | None = None,
        *,
        tier_budgets: Mapping[FidelityTier, Mapping[str, float]] | None = None,
    ) -> None:
        self._policy = policy or FidelityPolicy()
        self._tier_budgets = tier_budgets

    @property
    def policy(self) -> FidelityPolicy:
        return self._policy

    def resolve(self, scenario: Scenario) -> dict[str, FidelitySelection]:
        """The per-agent fidelity selection for ``scenario`` under this scheduler's policy."""
        return {
            spec.agent_id: select_fidelity(
                spec.agent_id,
                _available_profiles(spec),
                self._policy,
                tier_budgets=self._tier_budgets,
            )
            for spec in scenario.agents
        }

    def error_budget_outcomes(self, scenario: Scenario) -> list[ErrorBudgetOutcome]:
        """Every agent's per-channel error-budget outcomes — Parquet report rows + provenance."""
        outcomes: list[ErrorBudgetOutcome] = []
        for selection in self.resolve(scenario).values():
            outcomes.extend(selection.error_budget_outcomes())
        return outcomes
