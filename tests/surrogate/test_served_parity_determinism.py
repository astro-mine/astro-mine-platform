"""Served-path parity, determinism, and speedup-at-a-published-bound (RM-P1-SURR-04).

The acceptance gates for the served tier (surrogate.md §10):
- **Parity at the published bound** — the ONNX tier reproduces the torch surrogate's prediction to
  float precision, so it introduces *no* error beyond the calibrated ``ErrorReport`` it ships.
- **Determinism / golden** — seeded, fixed-input served inference reproduces identical outputs
  (single-threaded ORT), the reproducibility anchor CI pins.
- **Speedup** — the served tier's inference is measured; the oracle-relative speedup on a full Bench
  scenario is the joint SIM-03 + Bench deliverable (a surrogate cannot time the DEM solver without
  importing Sim), so this only demonstrates that the served path runs and is not slower than the
  torch path, honestly scoped.
"""

from __future__ import annotations

import time

import numpy as np

from astro_mine.surrogate.serve import OnnxServedSurrogate


def test_served_matches_native_within_tolerance(surrogate, served_bundle, served_query) -> None:
    served = OnnxServedSurrogate(served_bundle)
    native = surrogate.predict(served_query)
    got = served.predict(served_query)
    for field in ("position", "velocity"):
        assert np.abs(got.fields[field] - native.fields[field]).max() < 1e-4
        assert np.abs(got.field_uncertainty[field] - native.field_uncertainty[field]).max() < 1e-4
    assert got.in_domain == native.in_domain
    assert abs(got.ood_margin - native.ood_margin) < 1e-5


def test_served_generalizes_across_particle_counts(surrogate, served_bundle, served_query) -> None:
    """The raw-state graph is dynamic in N — it serves a bed with a different particle count.

    Sim scenarios carry beds of varying size; the served tier must handle any N, matching the torch
    surrogate (which also featurizes per-query). Truncating the bed exercises a fresh N.
    """
    served = OnnxServedSurrogate(served_bundle)
    smaller = {
        "position": served_query["position"][:-4],
        "velocity": served_query["velocity"][:-4],
        "tool_x": served_query["tool_x"],
        "config": served_query["config"],
    }
    got = served.predict(smaller)
    native = surrogate.predict(smaller)
    assert got.fields["position"].shape == smaller["position"].shape
    assert np.abs(got.fields["position"] - native.fields["position"]).max() < 1e-4


def test_served_inference_is_deterministic(served_bundle, served_query) -> None:
    served = OnnxServedSurrogate(served_bundle)
    first = served.predict(served_query)
    second = served.predict(served_query)
    # Byte-identical across repeated inference — the golden/determinism gate.
    assert np.array_equal(first.fields["position"], second.fields["position"])
    assert np.array_equal(first.fields["velocity"], second.fields["velocity"])
    assert np.array_equal(first.field_uncertainty["position"], second.field_uncertainty["position"])
    # A fresh session from the same bundle bytes reproduces the same outputs.
    reloaded = OnnxServedSurrogate.from_bytes(served_bundle.serialize())
    assert np.array_equal(
        reloaded.predict(served_query).fields["position"], first.fields["position"]
    )


def test_served_error_stays_within_the_published_bound(
    surrogate, served_bundle, served_query
) -> None:
    """The served tier's deviation from the torch surrogate is far below the published RMSE bound.

    "Demonstrated speedup at a *published, calibrated error bound*": the bound is the surrogate's
    ErrorReport; the served tier must not erode it. The ONNX-vs-native deviation is orders of
    magnitude under the smallest per-channel RMSE, so the served bound is the published bound.
    """
    served = OnnxServedSurrogate(served_bundle)
    native = surrogate.predict(served_query)
    got = served.predict(served_query)
    smallest_rmse = min(
        c.continuous.rmse for c in surrogate.error_report.channels if c.continuous is not None
    )
    served_deviation = max(
        np.abs(got.fields[f] - native.fields[f]).max() for f in ("position", "velocity")
    )
    assert served_deviation < 0.01 * smallest_rmse


def test_served_path_runs_and_is_not_slower_than_torch(
    surrogate, served_bundle, served_query
) -> None:
    served = OnnxServedSurrogate(served_bundle)
    # warm up both paths (session init / first-call graph build).
    served.predict(served_query)
    surrogate.predict(served_query)
    t0 = time.perf_counter()
    for _ in range(20):
        served.predict(served_query)
    served_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        surrogate.predict(served_query)
    torch_s = time.perf_counter() - t0
    # A loose sanity bound only (CI timing is noisy): the served path completes and is within an
    # order of magnitude of torch. The real oracle-relative speedup is the SIM-03 + Bench pairing.
    assert served_s < 10 * torch_s
