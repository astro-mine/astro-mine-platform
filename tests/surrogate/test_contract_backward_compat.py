"""The SURR-01 scalar contract is unchanged by the particle extension (RM-P1-SURR-02).

The per-particle ``fields`` are additive: a scalar surrogate constructs a ``Prediction`` exactly as
before and gets empty field maps. No torch — this guards the merged SURR-01 contract layer.
"""

from __future__ import annotations

from astro_mine.surrogate import ParticleFields, Prediction, SurrogateState


def test_scalar_prediction_is_unchanged() -> None:
    pred = Prediction(
        channels={"draft_force_n": 12.0}, uncertainty={"draft_force_n": 2.0}, in_domain=True
    )
    assert pred.channels["draft_force_n"] == 12.0
    assert pred.uncertainty["draft_force_n"] == 2.0
    assert pred.fields == {}  # additive: defaults empty, so scalar surrogates are unaffected
    assert pred.field_uncertainty == {}
    assert pred.categoricals == {}


def test_field_prediction_carries_arrays() -> None:
    import numpy as np

    pos = np.zeros((5, 2))
    pred = Prediction(
        channels={},
        uncertainty={},
        in_domain=True,
        fields={"position": pos},
        field_uncertainty={"position": pos + 0.1},
    )
    assert pred.fields["position"].shape == (5, 2)
    assert (pred.field_uncertainty["position"] > 0.0).all()


def test_new_contract_aliases_are_exported() -> None:
    # additive public surface: the particle-state aliases are exported alongside ChannelVector.
    assert ParticleFields is not None
    assert SurrogateState is not None
