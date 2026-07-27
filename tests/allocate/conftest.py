"""Shared test configuration for the allocation contract tests (RM-P1-ALLOC-01)."""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# A deterministic, CI-friendly Hypothesis profile: enough examples to exercise the
# feasibility oracle across shapes without a slow CI step.
settings.register_profile(
    "alloc",
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("alloc")
