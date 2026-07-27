"""SurrogateModel runtime seam + Prediction (RM-P1-SURR-01).

Proves the behavioural contract: a model exposes a static ErrorReport and returns a
calibrated per-channel prediction on every call, with the trust-region flag lowered
(never a confident extrapolation) outside its declared domain.
"""

from __future__ import annotations

from astro_mine.surrogate import ErrorReport, Prediction, SurrogateModel
from astro_mine.surrogate.model import ChannelVector
from tests.surrogate.factories import granular_report


class _ExcavationStub:
    """A minimal duck-typed surrogate over one continuous channel."""

    def __init__(self, report: ErrorReport) -> None:
        self._report = report

    @property
    def error_report(self) -> ErrorReport:
        return self._report

    def predict(self, state: ChannelVector, action: ChannelVector | None = None) -> Prediction:
        depth = state.get("tool_depth_m", 0.0)
        in_domain = 0.0 <= depth <= 0.5
        # Out of the declared trust region: raise uncertainty and lower the flag.
        sigma = 0.3 if in_domain else 5.0
        return Prediction(
            channels={"reaction_force_n": 12.0},
            uncertainty={"reaction_force_n": sigma},
            in_domain=in_domain,
            ood_margin=(0.5 - depth) if depth > 0.5 else None,
        )


def test_prediction_carries_value_uncertainty_and_flag() -> None:
    pred = Prediction(channels={"f": 1.0}, uncertainty={"f": 0.2}, in_domain=True)
    assert pred.channels["f"] == 1.0
    assert pred.uncertainty["f"] == 0.2
    assert pred.in_domain is True
    assert pred.categoricals == {}
    assert pred.ood_margin is None


def test_stub_satisfies_the_protocol_structurally() -> None:
    model = _ExcavationStub(granular_report())
    assert isinstance(model, SurrogateModel)


def test_a_type_missing_predict_is_not_a_surrogate_model() -> None:
    class NoPredict:
        @property
        def error_report(self) -> ErrorReport:
            return granular_report()

    assert not isinstance(NoPredict(), SurrogateModel)


def test_every_prediction_has_per_channel_uncertainty() -> None:
    model = _ExcavationStub(granular_report())
    pred = model.predict({"tool_depth_m": 0.25})
    # Uncertainty keys cover the predicted channels — a bound on every output, not a scalar.
    assert set(pred.uncertainty) == set(pred.channels)
    assert pred.in_domain is True


def test_out_of_domain_query_lowers_the_flag_and_raises_uncertainty() -> None:
    model = _ExcavationStub(granular_report())
    in_dom = model.predict({"tool_depth_m": 0.25})
    ood = model.predict({"tool_depth_m": 2.0})
    assert ood.in_domain is False
    assert ood.uncertainty["reaction_force_n"] > in_dom.uncertainty["reaction_force_n"]
    assert ood.ood_margin is not None


def test_model_exposes_its_static_error_report() -> None:
    report = granular_report()
    assert _ExcavationStub(report).error_report is report
