"""The adaptive surrogate/DEM granular engine (RM-P1-SIM-03; sim.md §10, §11).

The runtime surrogate-error gate. The engine steps each excavator's particle bed with the **learned
surrogate** (ONNX Runtime) while the tier is trusted, and **escalates to the DEM ground truth**
mid-episode when it is not — either because a live query left the trust region (``in_domain`` is
false — an OOD excursion, surrogate.md principle 3) or because a periodic **re-validation** against
the DEM oracle found the tracked deviation outside the task tolerance (sim.md §10). Once escalated
for an agent, it stays on ground truth (a retired surrogate). Every re-validation records a Core
:class:`~astro_mine.core.provenance.model.ErrorBudgetOutcome` — the rows of the Parquet error-budget
report Bench ingests.

It reuses the DEM engine (RM-P1-SIM-06) for actions, coupling, and the ground-truth steps — the
surrogate replaces only the per-step **particle kinematics**; the energy/battery coupling on the
surrogate path is a reduced proxy (documented) pending a force-predicting surrogate, and is carried
exactly by DEM at every re-validation. numpy + onnxruntime are the ``[dem]``/``[surrogate]`` extras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.core.provenance.model import ErrorBudgetOutcome
from astro_mine.sim.engines.dem._engine import _blade_in_bed, build_dem_engine
from astro_mine.sim.engines.dem._solver import DemBed, substep
from astro_mine.sim.engines.surrogate._descriptor import SURROGATE_GRANULAR_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy as np
    import numpy.typing as npt

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.sim.engines.adapter import CouplingState, EngineDescriptor, RegimeEngine
    from astro_mine.sim.engines.dem._engine import _DemDiggerState
    from astro_mine.sim.engines.surrogate._loader import LoadedSurrogate, SurrogateStep
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import DemGranularDynamics, Scenario
    from astro_mine.sim.scheduler import FidelityPolicy

    FloatArray = npt.NDArray[np.float64]

__all__ = [
    "AdaptiveGranularEngine",
    "build_scheduled_granular_engine",
    "build_surrogate_granular_engine",
]

#: Output channels the surrogate predicts, in the (N,2)+(N,2) layout order — the deviation +
#: budget channels the scheduler tolerance is keyed by.
_CHANNELS = ("pos_x", "pos_z", "vel_x", "vel_z")
#: Reduced specific-energy proxy (J per kg of excavated mass) for the battery draw on the
#: surrogate path; the DEM ground truth carries the exact draft-work energy at re-validation.
_SPECIFIC_ENERGY_J_PER_KG = 5.0e4


def _config_vector(dynamics: DemGranularDynamics, order: list[str]) -> FloatArray:
    """The surrogate's config vector [density, friction, restitution, tool_speed] in its declared
    ``param_names`` order, read off the scenario's terramechanics spec (the alignment RM-P1-SIM-03
    relies on — restitution is a spec input, not a stored DEM param)."""
    import numpy as np

    source = {
        "density": dynamics.regolith_density_kg_m3,
        "friction": dynamics.friction_coeff,
        "restitution": dynamics.restitution,
        "tool_speed": dynamics.tool_speed_mps,
    }
    return np.asarray([source[name] for name in order], dtype=np.float64)


def _advance_dem(state: _DemDiggerState, dt_s: float) -> None:
    """Advance one agent's bed by DEM for ``dt_s`` (the DemGranularEngine.advance body)."""
    params = state.params
    for _ in range(max(1, round(dt_s / params.dt_internal_s))):
        active = state.digging and _blade_in_bed(state)
        substep(state.bed, params, params.dt_internal_s, tool_active=active)
        if active:
            work = state.bed.tool_reaction_n * params.tool_speed_mps * params.dt_internal_s
            state.battery_soc_j = max(state.battery_floor_j, state.battery_soc_j - work)
    _maybe_stop_digging(state)


def _maybe_stop_digging(state: _DemDiggerState) -> None:
    if state.digging and (
        state.bed.displaced_mass_kg(state.params) >= state.target_mass_kg
        or not _blade_in_bed(state)
    ):
        state.digging = False


class AdaptiveGranularEngine:
    """A SURROGATE-tier granular engine that escalates to DEM ground truth on drift/OOD.

    Built by :func:`build_surrogate_granular_engine`. Delegates actions and coupling to a wrapped
    DEM engine (RM-P1-SIM-06); ``advance`` steps each still-trusted agent with the surrogate and
    hot-swaps to DEM on an out-of-domain query or a re-validation breach. Accumulated
    :attr:`error_budget_outcomes` are the Parquet report rows."""

    def __init__(
        self,
        scenario: Scenario,
        rng: RngStreams,
        surrogate: LoadedSurrogate,
        *,
        tolerance: Mapping[str, float],
        revalidate_every: int,
    ) -> None:
        import numpy as np

        self._np = np
        self._dem = build_dem_engine(scenario, rng)
        self._surrogate = surrogate
        self._tolerance = dict(tolerance)
        self._revalidate_every = max(1, revalidate_every)
        # Refuse to grade the tier over a longer rollout than its budget was calibrated for. The
        # budget holds up to `budget_horizon_steps` of drift (astro-mine-surrogate#23); re-anchoring
        # every `revalidate_every` steps caps drift at that, so a coarser cadence would check a
        # bound the producer never declared — the exact silent mismatch this fix exists to close.
        horizon = getattr(surrogate, "budget_horizon_steps", 1)
        if self._revalidate_every > horizon:
            raise ValueError(
                f"revalidate_every={self._revalidate_every} exceeds the tier's declared "
                f"budget_horizon_steps={horizon}: its error budget was calibrated to hold over a "
                f"{horizon}-step rollout, so grading it every {self._revalidate_every} steps "
                "checks a bound it never made. Re-validate at least as often as that horizon."
            )
        dynamics = {
            spec.agent_id: spec.dynamics
            for spec in scenario.agents
            if spec.dynamics.kind == "dem_granular"
        }
        self._config = {
            aid: _config_vector(dynamics[aid], surrogate.input_config_order)
            for aid in self._dem._states
        }
        self._mode = {aid: "surrogate" for aid in self._dem._states}
        self._step_count = {aid: 0 for aid in self._dem._states}
        self._outcomes: list[ErrorBudgetOutcome] = []

    @property
    def descriptor(self) -> EngineDescriptor:
        return SURROGATE_GRANULAR_ENGINE_DESCRIPTOR

    @property
    def error_budget_outcomes(self) -> list[ErrorBudgetOutcome]:
        """The per-re-validation deviation-vs-DEM outcomes accumulated so far (Parquet rows)."""
        return list(self._outcomes)

    def modes(self) -> dict[str, str]:
        """Each agent's current tier — ``"surrogate"`` (trusted) or ``"dem"`` (escalated)."""
        return dict(self._mode)

    def apply_actions(self, actions: ActionBatch) -> None:
        self._dem.apply_actions(actions)

    def advance(self, dt_s: float) -> None:
        for agent_id, state in self._dem._states.items():
            if self._mode[agent_id] == "dem":
                _advance_dem(state, dt_s)
                continue
            self._step_surrogate(agent_id, state, dt_s)

    def _step_surrogate(self, agent_id: str, state: _DemDiggerState, dt_s: float) -> None:
        self._step_count[agent_id] += 1
        step = self._surrogate.step(
            state.bed.pos, state.bed.vel, state.bed.tool_x_m, self._config[agent_id]
        )
        if not step.in_domain:
            # OOD excursion → escalate to ground truth permanently; DEM carries this tick.
            self._record(agent_id, "in_domain", value=step.ood_margin, tolerance=0.0, within=False)
            self._mode[agent_id] = "dem"
            _advance_dem(state, dt_s)
            return
        if self._step_count[agent_id] % self._revalidate_every == 0:
            # A re-validation is due. Advance the **real** bed by DEM — with its correct
            # battery/tool/digging accounting — score the surrogate's prediction against that DEM
            # result, and keep the DEM bed whether or not the check passes.
            #
            # Keeping it on a *passing* check is the fix for astro-mine-surrogate#23. A surrogate
            # left to roll on its own predictions drifts without bound: its output is its next
            # input. So the state Sim graded it on was arbitrarily far downstream of anything the
            # error budget was calibrated on, and a tier that passed early breached late. Re-
            # anchoring to ground truth every `revalidate_every` steps caps that drift at exactly
            # `revalidate_every` — the horizon the budget can then be calibrated to hold at.
            #
            # And it is free: the DEM step scored against was going to be computed either way. The
            # only change is that its result is adopted instead of discarded. The old code stepped a
            # throwaway DEM copy, compared, and then threw the ground truth away.
            surrogate_pos, surrogate_vel = step.next_pos, step.next_vel
            _advance_dem(state, dt_s)
            if self._check(agent_id, surrogate_pos, surrogate_vel, state.bed):
                self._mode[agent_id] = "dem"  # breached → retire the surrogate for this agent
            return
        self._adopt_surrogate(state, step, dt_s)

    def _check(
        self,
        agent_id: str,
        surrogate_pos: FloatArray,
        surrogate_vel: FloatArray,
        reference: DemBed,
    ) -> bool:
        """Score the surrogate's prediction against the DEM-advanced bed; record per-channel
        outcomes. Returns ``True`` if any channel's max deviation exceeds its tolerance."""
        surrogate_cols = (
            surrogate_pos[:, 0],
            surrogate_pos[:, 1],
            surrogate_vel[:, 0],
            surrogate_vel[:, 1],
        )
        reference_cols = (
            reference.pos[:, 0],
            reference.pos[:, 1],
            reference.vel[:, 0],
            reference.vel[:, 1],
        )
        breached = False
        for channel, surrogate_col, reference_col in zip(
            _CHANNELS, surrogate_cols, reference_cols, strict=True
        ):
            tolerance = self._tolerance.get(channel)
            if tolerance is None:
                continue
            deviation = float(self._np.abs(surrogate_col - reference_col).max())
            within = deviation <= tolerance
            self._record(agent_id, channel, value=deviation, tolerance=tolerance, within=within)
            breached = breached or not within
        return breached

    def _adopt_surrogate(self, state: _DemDiggerState, step: SurrogateStep, dt_s: float) -> None:
        params = state.params
        prior_mass = state.bed.displaced_mass_kg(params)
        state.bed.pos = step.next_pos
        state.bed.vel = step.next_vel
        if state.digging and _blade_in_bed(state):
            state.bed.tool_x_m += params.tool_speed_mps * dt_s
            excavated = max(0.0, state.bed.displaced_mass_kg(params) - prior_mass)
            drawn = (
                _SPECIFIC_ENERGY_J_PER_KG * excavated
            )  # reduced energy proxy on the surrogate path
            state.battery_soc_j = max(state.battery_floor_j, state.battery_soc_j - drawn)
        _maybe_stop_digging(state)

    def _record(
        self, agent_id: str, metric: str, *, value: float, tolerance: float, within: bool
    ) -> None:
        self._outcomes.append(
            ErrorBudgetOutcome(
                name=f"{agent_id}:{metric}",
                within_budget=within,
                tier="surrogate",
                metric=metric,
                value=value,
                tolerance=tolerance,
            )
        )

    def export_coupling_state(self) -> CouplingState:
        return self._dem.export_coupling_state()

    def import_coupling_state(self, state: CouplingState) -> None:
        self._dem.import_coupling_state(state)

    def retire(self, agent_ids: Iterable[str]) -> None:
        self._dem.retire(agent_ids)
        for agent_id in agent_ids:
            self._mode.pop(agent_id, None)
            self._config.pop(agent_id, None)
            self._step_count.pop(agent_id, None)


def build_surrogate_granular_engine(
    scenario: Scenario,
    rng: RngStreams,
    surrogate: LoadedSurrogate,
    *,
    tolerance: Mapping[str, float],
    revalidate_every: int | None = None,
) -> AdaptiveGranularEngine:
    """Build an :class:`AdaptiveGranularEngine` for the scenario's ``dem_granular`` agents.

    ``surrogate`` is a verified :class:`LoadedSurrogate`; ``tolerance`` is the task's per-channel
    error budget (the scenario's ``FidelityPolicy.error_budget``); ``revalidate_every`` sets the
    DEM re-validation cadence in steps. Left unset it defaults to the tier's declared
    ``budget_horizon_steps`` — the coarsest cadence at which the budget is still a bound the
    producer made (astro-mine-surrogate#23), so a caller need not know the tier's internals to run
    it safely.
    """
    cadence = surrogate.budget_horizon_steps if revalidate_every is None else revalidate_every
    return AdaptiveGranularEngine(
        scenario, rng, surrogate, tolerance=tolerance, revalidate_every=cadence
    )


def build_scheduled_granular_engine(
    scenario: Scenario,
    rng: RngStreams,
    surrogate: LoadedSurrogate,
    *,
    policy: FidelityPolicy,
    revalidate_every: int | None = None,
) -> RegimeEngine:
    """Let the **scheduler's admission decide the engine binding** (RM-P1-SIM-03).

    Runs :func:`~astro_mine.sim.scheduler.select_fidelity` over a ``(DEM-reference, surrogate)``
    ladder with the surrogate's declared ``recommended_error_budget`` and the ``policy``'s task
    tolerance. If the surrogate is admitted within budget, the run binds the
    :class:`AdaptiveGranularEngine` (which starts on the surrogate and escalates on drift/OOD);
    otherwise it binds the plain DEM ground-truth engine — the static half of the surrogate-error
    gate, its runtime half handled inside the adaptive engine."""
    from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier
    from astro_mine.core.sadf.model import FidelityProfile
    from astro_mine.sim.scheduler import select_fidelity

    ladder = (
        FidelityProfile(
            tier=FidelityTier.ARTICULATED, determinism_class=DeterminismClass.TOLERANCE
        ),
        FidelityProfile(tier=FidelityTier.SURROGATE, determinism_class=DeterminismClass.TOLERANCE),
    )
    admitted = any(
        select_fidelity(
            spec.agent_id,
            ladder,
            policy,
            tier_budgets={FidelityTier.SURROGATE: surrogate.recommended_error_budget},
        ).tier
        is FidelityTier.SURROGATE
        for spec in scenario.agents
        if spec.dynamics.kind == "dem_granular"
    )
    if admitted and policy.error_budget is not None:
        cadence = surrogate.budget_horizon_steps if revalidate_every is None else revalidate_every
        return AdaptiveGranularEngine(
            scenario,
            rng,
            surrogate,
            tolerance=policy.error_budget,
            revalidate_every=cadence,
        )
    return build_dem_engine(scenario, rng)
