# SPDX-License-Identifier: Apache-2.0
"""The learned-DEM excavation surrogate — a concrete ``SurrogateModel`` (RM-P1-SURR-02).

Ties the pieces together: a deep-ensemble GNS trained on the frozen DEM fixture, split-conformal
per-channel bounds, and an enforced excavation-parameter trust region — wrapped as a
:class:`~astro_mine.surrogate.model.SurrogateModel` whose ``predict`` returns per-particle field
predictions with calibrated uncertainty, and which carries the calibrated
:class:`~astro_mine.surrogate.report.ErrorReport` (the "error is the product" artifact) built from
a held-out validation split against the SIM-06 DEM oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from astro_mine.surrogate.enums import ChannelKind, PhysicsDomain
from astro_mine.surrogate.model import ParticleFields, Prediction, SurrogateState
from astro_mine.surrogate.models.conformal import ConformalCalibration, calibrate_conformal
from astro_mine.surrogate.models.dataset import DemDataset, load_dem_dataset
from astro_mine.surrogate.models.gns import GNS
from astro_mine.surrogate.models.train import (
    Normalizer,
    TrainConfig,
    Transition,
    build_transitions,
    ensemble_predict,
    fit_normalizer,
    train_ensemble,
)
from astro_mine.surrogate.models.trust_region import ExcavationTrustRegion
from astro_mine.surrogate.report import (
    ChannelError,
    ContinuousMetrics,
    CoveragePoint,
    ErrorReport,
    OracleRef,
    RolloutError,
    SubstitutionPolicy,
    TailBehavior,
)

__all__ = ["ExcavationSurrogate", "build_excavation_surrogate"]

FloatArray = npt.NDArray[np.float64]

_CHANNELS = ("pos_x", "pos_z", "vel_x", "vel_z")
_UNITS = ("m", "m", "m/s", "m/s")
_NOMINAL_COVERAGE = 0.9
#: Uncertainty inflation applied outside the trust region — never a confident extrapolation.
_OOD_INFLATION = 4.0
#: Headroom over the **worst** deviation on held-out validation — see :func:`_build_error_report`
#: for why the budget bounds a *max* and not an RMSE (surrogate#21).
#:
#: Not larger, because the budget is what Sim admits and re-validates against: inflate it and
#: "within budget" stops meaning anything. Not 1.0 either, because the validation set is a finite
#: sample of the trust region, and a config it never drew should not trip the guard for being a
#: hair worse than the worst thing it happened to see.
_BUDGET_MARGIN = 1.5

#: The rollout horizon the budget is calibrated to hold at, and the horizon Sim must therefore
#: re-validate at least as often as (surrogate#23).
#:
#: A step surrogate feeds its own output back as its next input, so left alone it drifts without
#: bound — and the state Sim grades it on is then arbitrarily far downstream of anything the budget
#: was calibrated on. A budget calibrated at horizon 1 (one step from a DEM bed) is comfortably met
#: on the *first* step and breached a few steps into a rollout; that is exactly what blocked
#: astro-mine-bench#31. Sim now re-anchors the bed to DEM ground truth on every passing
#: re-validation (``sim/engines/surrogate/_engine.py``), capping the drift at ``revalidate_every``
#: steps — so the budget only has to hold over a rollout of that length, and is calibrated to.
#:
#: **2, and measured, not assumed.** The tier declares this in its manifest and Sim re-validates at
#: least this often, refusing a coarser cadence. Running the anchor through the real adaptive
#: engine: over 1 or 2 free drift steps every channel stays inside budget, but over 3
#: (``revalidate_every=4``) the surrogate's rollout in the *deep-blade* regime — late in a dig,
#: blade fully engaged — drifts into a bed DEM itself blows up on (``vel_x`` ~18 m/s), and it
#: escalates. So 4 would be a horizon
#: the model cannot actually hold, and declaring it would be exactly the kind of unbacked bound
#: surrogate#21 and #23 exist to stop.
#:
#: The honest consequence is a modest speedup: re-validating every 2 steps pays a DEM reference step
#: half the ticks, so the measured ratio is well under the tier's raw per-step speed. That the tier
#: cannot sustain a longer rollout in the hard regime is a *model* limitation (rollout stability —
#: more deep-blade data, noise-injection training), tracked separately; the budget's job is only to
#: state the horizon truthfully, which is what this does.
_BUDGET_HORIZON_STEPS = 2


@dataclass(frozen=True)
class _Split:
    train: list[Transition]
    calibrate: list[Transition]
    validate: list[Transition]


class ExcavationSurrogate:
    """A learned-DEM granular surrogate: a calibrated per-particle prediction on every call.

    ``predict`` takes a :data:`~astro_mine.surrogate.model.ParticleFields` state
    (``position`` (N,2), ``velocity`` (N,2), ``tool_x`` (1,), ``config`` (P,)) and returns a
    :class:`~astro_mine.surrogate.model.Prediction` whose ``fields`` are the next
    position/velocity and whose ``field_uncertainty`` are conformal-calibrated per-particle
    half-widths — raised, with ``in_domain=False``, outside the trust region.
    """

    def __init__(
        self,
        *,
        models: list[GNS],
        normalizer: Normalizer,
        dataset: DemDataset,
        trust_region: ExcavationTrustRegion,
        conformal: ConformalCalibration,
        error_report: ErrorReport,
    ) -> None:
        self._models = models
        self._normalizer = normalizer
        self._dataset = dataset
        self._trust_region = trust_region
        self._conformal = conformal
        self._error_report = error_report

    @property
    def error_report(self) -> ErrorReport:
        return self._error_report

    def predict(self, state: SurrogateState, action: SurrogateState | None = None) -> Prediction:
        particle_state, tool_x, config = _parse_state(state)
        in_domain = self._trust_region.contains(config)
        margin = self._trust_region.margin(config)
        mean, std = ensemble_predict(
            self._models, particle_state, tool_x, config, self._dataset, self._normalizer
        )
        half = self._conformal.half_widths(std)  # (N, 4)
        if not in_domain:
            half = half * _OOD_INFLATION  # raise uncertainty, never a confident extrapolation
        fields: dict[str, FloatArray] = {"position": mean[:, :2], "velocity": mean[:, 2:]}
        field_uncertainty: dict[str, FloatArray] = {
            "position": half[:, :2],
            "velocity": half[:, 2:],
        }
        return Prediction(
            channels={},
            uncertainty={},
            in_domain=in_domain,
            ood_margin=margin,
            fields=fields,
            field_uncertainty=field_uncertainty,
        )

    def rollout(
        self,
        state: FloatArray,
        tool_x_m: float,
        config: FloatArray,
        *,
        steps: int,
        tool_speed: float,
    ) -> FloatArray:
        """Autoregressively roll the surrogate ``steps`` forward, feeding its own predictions.

        Returns the predicted state trajectory ``(steps+1, N, 4)`` — where rollout drift (which
        noise-injection training bounds) shows up.
        """
        traj = [state]
        current, tool = state, tool_x_m
        for _ in range(steps):
            mean, _ = ensemble_predict(
                self._models, current, tool, config, self._dataset, self._normalizer
            )
            traj.append(mean)
            current, tool = mean, tool + tool_speed * self._dataset.dt_s
        return np.stack(traj)


def _parse_state(state: SurrogateState) -> tuple[FloatArray, float, FloatArray]:
    """Extract ``(particle_state (N,4), tool_x, config (P,))`` from a ParticleFields query."""
    fields: ParticleFields = state  # type: ignore[assignment]
    position = np.asarray(fields["position"], dtype=np.float64)
    velocity = np.asarray(fields["velocity"], dtype=np.float64)
    tool_x = float(np.asarray(fields["tool_x"], dtype=np.float64).reshape(-1)[0])
    config = np.asarray(fields["config"], dtype=np.float64).reshape(-1)
    return np.hstack([position, velocity]), tool_x, config


def _split(transitions: list[Transition], seed: int) -> _Split:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(transitions))
    n = len(idx)
    n_train, n_cal = int(0.7 * n), int(0.15 * n)

    def pick(ids: npt.NDArray[np.intp]) -> list[Transition]:
        return [transitions[int(i)] for i in ids]

    return _Split(
        train=pick(idx[:n_train]),
        calibrate=pick(idx[n_train : n_train + n_cal]),
        validate=pick(idx[n_train + n_cal :]),
    )


def _residuals_and_std(
    surrogate_models: list[GNS],
    transitions: list[Transition],
    dataset: DemDataset,
    normalizer: Normalizer,
) -> tuple[FloatArray, FloatArray]:
    resid, stds = [], []
    for tr in transitions:
        mean, std = ensemble_predict(
            surrogate_models, tr.state, tr.tool_x_m, tr.config, dataset, normalizer
        )
        resid.append(np.abs(mean - tr.next_state))
        stds.append(std)
    return np.vstack(resid), np.vstack(stds)


def _build_error_report(
    *,
    name: str,
    version: str,
    dataset: DemDataset,
    trust_region: ExcavationTrustRegion,
    val_resid: FloatArray,
    val_std: FloatArray,
    conformal: ConformalCalibration,
    rollout_rmse: list[float],
    budget: dict[str, float],
) -> ErrorReport:
    channels = []
    half_widths = conformal.half_widths(val_std)  # (M, 4)
    for c, (chan, unit) in enumerate(zip(_CHANNELS, _UNITS, strict=True)):
        col = val_resid[:, c]
        covered = float(np.mean(col <= half_widths[:, c]))
        channels.append(
            ChannelError(
                channel=chan,
                kind=ChannelKind.CONTINUOUS,
                continuous=ContinuousMetrics(
                    unit=unit,
                    rmse=float(np.sqrt(np.mean(col**2))),
                    coverage=[CoveragePoint(nominal=_NOMINAL_COVERAGE, empirical=covered)],
                    tail=TailBehavior(
                        p95_abs_error=float(np.quantile(col, 0.95)),
                        p99_abs_error=float(np.quantile(col, 0.99)),
                        max_abs_error=float(col.max()),
                    ),
                ),
            )
        )
    # ``budget`` comes from :func:`_rollout_budget`. Two properties matter, and both were bugs
    # that blocked astro-mine-bench#31:
    #
    #   * It is a **max** over particles, not an RMSE. Sim enforces
    #     ``abs(surrogate - reference).max()`` over the bed; an RMSE bounds nothing, since half the
    #     particles exceed it by construction (surrogate#21).
    #   * It holds over the **rollout horizon** Sim grades at, not just one step. A step surrogate
    #     drifts on its own output, so a horizon-1 budget is met on step 1 and breached a few steps
    #     in — which is exactly what happened after #21 fixed the statistic (surrogate#23).
    #
    # `val_resid` (single step from a DEM bed) is kept only for the reported per-channel RMSE
    # tail above; it is deliberately *not* the budget any more.
    return ErrorReport(
        surrogate_name=name,
        surrogate_version=version,
        domain=PhysicsDomain.GRANULAR_EXCAVATION,
        channels=channels,
        trust_region=trust_region.to_report_trust_region(),
        validation_dataset_hash=dataset.content_hash(),
        oracle=OracleRef(producer="astro-mine-sim", producer_version="RM-P1-SIM-06"),
        substitution_policy=SubstitutionPolicy(
            recommended_error_budget=budget, budget_horizon_steps=_BUDGET_HORIZON_STEPS
        ),
        rollout=RolloutError(horizon_steps=len(rollout_rmse), rmse_by_horizon=rollout_rmse),
    )


def build_excavation_surrogate(
    *,
    dataset: DemDataset | None = None,
    config: TrainConfig | None = None,
    seed: int = 0,
    name: str = "excavation-gns",
    version: str = "0.1.0",
) -> ExcavationSurrogate:
    """Train + calibrate the surrogate on the DEM fixture and build its calibrated ErrorReport.

    Deterministic for a given ``seed`` + fixture (torch CPU, in-process). Kept small so CI trains
    in seconds. Splits the transitions 70/15/15 into train / conformal-calibration / validation.
    """
    dataset = dataset or load_dem_dataset()
    config = config or TrainConfig()
    transitions = build_transitions(dataset)
    split = _split(transitions, seed)
    normalizer = fit_normalizer(split.train, dataset)
    models = train_ensemble(dataset, split.train, normalizer, config, seed=seed)

    cal_resid, cal_std = _residuals_and_std(models, split.calibrate, dataset, normalizer)
    conformal = calibrate_conformal(
        cal_resid, cal_std, _CHANNELS, nominal_coverage=_NOMINAL_COVERAGE
    )
    val_resid, val_std = _residuals_and_std(models, split.validate, dataset, normalizer)

    trust_region = ExcavationTrustRegion.from_configs(dataset.param_names, dataset.params)
    rollout_rmse = _rollout_error(models, dataset, normalizer)
    budget = _rollout_budget(models, dataset, normalizer, horizon=_BUDGET_HORIZON_STEPS)
    error_report = _build_error_report(
        name=name,
        version=version,
        dataset=dataset,
        trust_region=trust_region,
        val_resid=val_resid,
        val_std=val_std,
        conformal=conformal,
        rollout_rmse=rollout_rmse,
        budget=budget,
    )
    return ExcavationSurrogate(
        models=models,
        normalizer=normalizer,
        dataset=dataset,
        trust_region=trust_region,
        conformal=conformal,
        error_report=error_report,
    )


def _rollout_error(
    models: list[GNS], dataset: DemDataset, normalizer: Normalizer, *, horizon: int = 5
) -> list[float]:
    """Per-horizon autoregressive rollout RMSE against the DEM truth on the first config."""
    truth = dataset.states[0]  # (T+1, N, 4)
    tool_speed = float(dataset.params[0, 3])
    current, tool = truth[0], float(dataset.tool_x[0, 0])
    rmse = []
    for h in range(1, horizon + 1):
        mean, _ = ensemble_predict(models, current, tool, dataset.params[0], dataset, normalizer)
        rmse.append(float(np.sqrt(np.mean((mean - truth[h]) ** 2))))
        current, tool = mean, tool + tool_speed * dataset.dt_s
    return rmse


def _rollout_budget(
    models: list[GNS], dataset: DemDataset, normalizer: Normalizer, *, horizon: int
) -> dict[str, float]:
    """Per-channel admission budget: the worst per-particle deviation over a ``horizon``-step
    autoregressive rollout, from **every** DEM frame of **every** config (surrogate#23).

    This mirrors what Sim actually grades. Sim re-anchors the bed to DEM ground truth every
    ``revalidate_every`` steps, so at any check the surrogate is at most that many steps into a
    rollout — but the frame it re-anchored *from* can be anywhere in the episode, including the
    deep-blade regime late in a dig where the physics is hardest. So the budget is the worst
    deviation
    over a ``horizon``-step rollout launched from **each** frame ``t``, not just from the settled
    start: a budget calibrated only from frame 0 misses exactly the beds Sim will re-anchor to.

    Two axes, both of which a prior budget got wrong and both of which blocked astro-mine-bench#31:
    the reduction over particles is a **max** (Sim enforces a max, not the old RMSE — #21), and the
    reduction over time spans the whole rollout horizon (not a single step from a DEM bed, which
    held on step 1 and broke by step 4 — #23).
    """
    worst = np.zeros(len(_CHANNELS))
    n_frames = dataset.states.shape[1]
    for config in range(dataset.states.shape[0]):
        truth = dataset.states[config]  # (T+1, N, 4)
        params = dataset.params[config]
        tool_speed = float(params[3])
        for start in range(n_frames - horizon):
            current = truth[start]
            tool = float(dataset.tool_x[config, start])
            for h in range(1, horizon + 1):
                mean, _ = ensemble_predict(models, current, tool, params, dataset, normalizer)
                worst = np.maximum(worst, np.abs(mean - truth[start + h]).max(axis=0))
                current, tool = mean, tool + tool_speed * dataset.dt_s
    return {chan: _BUDGET_MARGIN * float(worst[c]) for c, chan in enumerate(_CHANNELS)}
