"""``ServedTier`` — static admission + live fall-back over the two error channels (RM-P1-SURR-04).

The surrogate-side reference of the admit/fall-back contract Sim's scheduler consumes (surrogate.md
§11 "both — static ErrorReport for admission, plus live per-query uncertainty for fallback").
"""

from __future__ import annotations

from astro_mine.surrogate.serve import OnnxServedSurrogate, ServedTier


def test_advance_matches_predict(served_bundle, served_query) -> None:
    served = OnnxServedSurrogate(served_bundle)
    tier = ServedTier(served)
    assert tier.advance(served_query).in_domain == served.predict(served_query).in_domain
    assert tier.error_report.content_hash() == served.error_report.content_hash()


def test_admits_only_within_the_recommended_budget(served_bundle) -> None:
    tier = ServedTier(OnnxServedSurrogate(served_bundle))
    budget = served_bundle.error_report.substitution_policy.recommended_error_budget
    loose = {channel: value * 2 for channel, value in budget.items()}
    tight = {channel: value * 0.1 for channel, value in budget.items()}
    assert tier.admits(loose) is True
    assert tier.admits(tight) is False


def test_admits_refuses_a_channel_with_no_declared_budget(served_bundle) -> None:
    tier = ServedTier(OnnxServedSurrogate(served_bundle))
    # A channel the surrogate never measured is not admissible (conservative).
    assert tier.admits({"unmeasured_channel": 1e9}) is False


def test_should_escalate_on_out_of_domain_query(served_bundle, ood_query) -> None:
    served = OnnxServedSurrogate(served_bundle)
    tier = ServedTier(served)
    prediction = served.predict(ood_query)
    assert prediction.in_domain is False
    # An OOD query always escalates, regardless of the tolerance.
    assert tier.should_escalate(prediction, max_uncertainty=1e9) is True


def test_should_escalate_respects_the_uncertainty_tolerance(served_bundle, served_query) -> None:
    served = OnnxServedSurrogate(served_bundle)
    tier = ServedTier(served)
    prediction = served.predict(served_query)
    assert prediction.in_domain is True
    # A generous tolerance keeps the in-domain tier admitted; a zero tolerance forces escalation.
    assert tier.should_escalate(prediction, max_uncertainty=1e9) is False
    assert tier.should_escalate(prediction, max_uncertainty=0.0) is True
