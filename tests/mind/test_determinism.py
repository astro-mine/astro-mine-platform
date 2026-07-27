"""Determinism-on-demand + the golden decision-trace gate (RM-P1-MIND-01; conventions.md §11).

Seed + pinned plugin set + fixed inputs ⇒ identical decisions, and byte-identical against a
stored golden trace. CI fails on any drift; regenerate an intended change with
``UPDATE_GOLDENS=1 pytest``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from astro_mine.mind.exec import Executive
from astro_mine.mind.trace import to_canonical_json
from tests.mind.support.harness import (
    assert_deterministic_trace,
    compose_reference_bt,
    compose_stack,
    run_reference,
)
from tests.mind.support.toy_env import ToyProspectingEnv

_GOLDEN_DIR = Path(__file__).parent / "golden"
_HORIZON = 8


def _run(resource: str):  # type: ignore[no-untyped-def]
    """Run a shipped stack over a fixed scenario for the golden gate; the degrade stack runs
    a fixed injected-blackout schedule so the trace exercises comms-loss handling."""
    if resource == "lunar_prospecting_bt.yaml":
        graph = compose_reference_bt(seed=7)
        env = ToyProspectingEnv(horizon=_HORIZON)
    elif resource == "lunar_prospecting_degrade.yaml":
        graph = compose_stack(resource, seed=7)
        env = ToyProspectingEnv(horizon=_HORIZON, comms_denied_ticks=(3, 4, 5, 6, 7))
    else:
        graph = compose_stack(resource, seed=7)
        env = ToyProspectingEnv(horizon=_HORIZON)
    return Executive(graph).run(env, max_ticks=_HORIZON, seed=7)


#: Every shipped reference stack is gated against a golden decision trace (RM-P1-MIND-07).
_GOLDENS = {
    "reference_stack.trace.json": "lunar_prospecting.yaml",
    "bt_stack.trace.json": "lunar_prospecting_bt.yaml",
    "backends_stack.trace.json": "lunar_prospecting_backends.yaml",
    "allocate_stack.trace.json": "lunar_prospecting_allocate.yaml",
    "degrade_stack.trace.json": "lunar_prospecting_degrade.yaml",
}


def test_run_is_deterministic() -> None:
    assert_deterministic_trace(lambda: run_reference(horizon=_HORIZON, max_ticks=_HORIZON))


@pytest.mark.parametrize(("golden", "resource"), sorted(_GOLDENS.items()))
def test_matches_golden_trace(golden: str, resource: str) -> None:
    trace_json = to_canonical_json(_run(resource).trace)
    path = _GOLDEN_DIR / golden
    if os.environ.get("UPDATE_GOLDENS"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(trace_json, encoding="utf-8")
    assert path.exists(), f"golden {golden} missing; regenerate with UPDATE_GOLDENS=1"
    assert trace_json == path.read_text(encoding="utf-8"), (
        f"decision trace for {resource} drifted from {golden}; if the change is intended, "
        "regenerate with UPDATE_GOLDENS=1 pytest"
    )
