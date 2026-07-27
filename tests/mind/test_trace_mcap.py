"""Determinism, MCAP trace, deep provenance, and plan explanation (RM-P1-MIND-07)."""

from __future__ import annotations

import pytest

from astro_mine.mind.exec import Executive
from astro_mine.mind.trace import content_hash, explain, to_canonical_json
from tests.mind.support.harness import compose_stack, run_stack
from tests.mind.support.toy_env import ToyProspectingEnv

_BLACKOUT = (3, 4, 5, 6, 7)


def _degrade_trace():  # type: ignore[no-untyped-def]
    graph = compose_stack("lunar_prospecting_degrade.yaml", seed=7)
    env = ToyProspectingEnv(horizon=10, comms_denied_ticks=_BLACKOUT)
    return Executive(graph).run(env, max_ticks=10, seed=7).trace


# --- content hashing / provenance -----------------------------------------------------


def test_content_hash_is_canonical_and_deterministic() -> None:
    assert content_hash("abc").startswith("sha256:")
    assert content_hash("abc") == content_hash(b"abc")
    assert content_hash("abc") != content_hash("abd")


def test_compose_threads_input_hashes_into_provenance() -> None:
    hashes = {"stack_spec": content_hash("spec-bytes"), "sadf": content_hash("asset-bytes")}
    graph = compose_stack("lunar_prospecting.yaml", seed=7)
    from astro_mine.mind.compose import compose
    from astro_mine.mind.reference import load_reference_stack
    from tests.mind.support.harness import reference_registry

    graph = compose(load_reference_stack(), reference_registry(), seed=7, input_hashes=hashes)
    assert graph.provenance.input_hashes == hashes
    assert graph.provenance.to_dict()["input_hashes"] == hashes


# --- MCAP serialization ---------------------------------------------------------------


def test_mcap_round_trips_all_record_kinds() -> None:
    pytest.importorskip("mcap")
    from astro_mine.mind.trace.mcap import MCAP_CHANNELS, read_mcap_messages, to_mcap_bytes

    trace = _degrade_trace()
    messages = read_mcap_messages(to_mcap_bytes(trace))
    topics = {topic for topic, _ in messages}
    assert topics == set(MCAP_CHANNELS)  # every channel is exercised by the degrade trace
    # tier decisions == one per (tick, tier)
    tier_decisions = [m for topic, m in messages if topic == "mind/tier_decision"]
    assert len(tier_decisions) == sum(len(t.tiers) for t in trace.ticks)
    # guard interventions carry the clause provenance
    guard = [m for topic, m in messages if topic == "mind/guard_intervention"]
    assert guard and all(m["clauses"] for m in guard)


def test_mcap_is_deterministic() -> None:
    pytest.importorskip("mcap")
    from astro_mine.mind.trace.mcap import to_mcap_bytes

    trace = _degrade_trace()
    assert to_mcap_bytes(trace) == to_mcap_bytes(trace)


def test_write_mcap_to_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("mcap")
    from astro_mine.mind.trace.mcap import read_mcap_messages, write_mcap

    trace = _degrade_trace()
    path = tmp_path / "trace.mcap"
    write_mcap(trace, path)
    assert path.exists() and read_mcap_messages(path)


# --- plan explanation (LUNAR-UX-003) --------------------------------------------------


def test_plan_explanation_narrates_degradation_and_guard() -> None:
    explanation = explain(_degrade_trace())
    assert explanation.seed == 7
    text = explanation.to_text()
    assert "act-while-stale" in text
    assert "reconciled on comms recovery" in text
    assert "guard intervened" in text
    # one explanation line per tick
    assert len(explanation.ticks) == len(_degrade_trace().ticks)


def test_plan_explanation_for_quiet_stack() -> None:
    trace = run_stack("lunar_prospecting.yaml", horizon=3, max_ticks=3).trace
    explanation = explain(trace)
    assert any("planned" in t.summary or "cached" in t.summary for t in explanation.ticks)


def test_determinism_gate_is_byte_exact() -> None:
    a = run_stack("lunar_prospecting_backends.yaml", horizon=5, max_ticks=5).trace
    b = run_stack("lunar_prospecting_backends.yaml", horizon=5, max_ticks=5).trace
    assert to_canonical_json(a) == to_canonical_json(b)
