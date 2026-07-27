"""Generate the RM-P1-WORLDS-10 illumination-surrogate test fixtures.

Builds a tiny **self-contained ONNX graph** — a linear ``lit score = easting_m`` stand-in for a
trained illumination-field surrogate — plus a schema-valid ``ErrorReport`` + ``SurrogateAttributes``
JSON, written under ``tests/fixtures/illumination/``. Worlds itself never depends on ``onnx``:
this script is run once from an environment that has it (the ``astro-mine-surrogate`` venv), and the
committed fixtures are loaded by the adapter tests via ONNX Runtime only — the same "frozen fixture
built by a script" pattern Sim uses for its surrogate tier.

The fixture surrogate is deliberately trivial and *distinguishable* from the horizon / ray-cast
reference (it lights a cell iff its easting is positive, independent of Sun and terrain), so the
adapter tests can tell "the surrogate served this" from "the reference served this" and exercise the
trust-region / OOD escalation path. ``error_report_digest`` is the Core content hash of the
``ErrorReport`` so the adapter's fail-closed integrity check passes.

Usage (from the surrogate venv):
    python scripts/gen_illumination_surrogate_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from astro_mine.core.hashing import content_hash_json

INPUT_CHANNELS = ["easting_m", "northing_m", "epoch_s", "sun_elevation_deg"]
OUTPUT_CHANNELS = ["visibility", "solar_flux_w_m2"]
_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "illumination"

_TRUST_REGION = {
    "bounds": {
        "easting_m": {"low": -1000000.0, "high": 1000000.0},
        "northing_m": {"low": -1000000.0, "high": 1000000.0},
        "epoch_s": {"low": 0.0, "high": 1000000000000.0},
        "sun_elevation_deg": {"low": -90.0, "high": 90.0},
    }
}
_RECOMMENDED_ERROR_BUDGET = {"visibility": 0.03, "solar_flux_w_m2": 15.0}


def build_onnx() -> bytes:
    """A linear graph ``lit = features @ [1,0,0,0]^T`` — lit score equals easting_m."""
    weight = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    bias = np.array([0.0], dtype=np.float32)
    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["features", "W"], ["score"]),
            helper.make_node("Add", ["score", "b"], ["lit"]),
        ],
        name="illumination_surrogate_fixture",
        inputs=[
            helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", len(INPUT_CHANNELS)])
        ],
        outputs=[helper.make_tensor_value_info("lit", TensorProto.FLOAT, ["N", 1])],
        initializer=[numpy_helper.from_array(weight, "W"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


def build_error_report() -> dict[str, object]:
    """A schema-valid illumination-field ``ErrorReport`` (categorical + continuous channels)."""
    return {
        "surrogate_name": "shackleton-illumination-fixture",
        "surrogate_version": "0.1.0",
        "domain": "illumination_field",
        "channels": [
            {
                "channel": "visibility",
                "kind": "categorical",
                "categorical": {
                    "classes": ["shadow", "lit"],
                    "accuracy": 0.97,
                    "reliability": [{"nominal": 0.9, "empirical": 0.9}],
                },
            },
            {
                "channel": "solar_flux_w_m2",
                "kind": "continuous",
                "continuous": {
                    "unit": "W/m^2",
                    "rmse": 12.0,
                    "coverage": [{"nominal": 0.9, "empirical": 0.9}],
                    "tail": {"p95_abs_error": 20.0, "p99_abs_error": 35.0, "max_abs_error": 60.0},
                },
            },
        ],
        "trust_region": _TRUST_REGION,
        "validation_dataset_hash": "sha256:" + "0" * 64,
        "oracle": {"producer": "astro-mine-worlds", "producer_version": "0.1.0"},
        "substitution_policy": {
            "recommended_error_budget": _RECOMMENDED_ERROR_BUDGET,
            "escalate_on_ood": True,
        },
        "rollout": None,
    }


def build_attributes(error_report: dict[str, object]) -> dict[str, object]:
    """The ``SurrogateAttributes`` a Core manifest folds in — admission read from here alone."""
    return {
        "domain": "illumination_field",
        "input_channels": INPUT_CHANNELS,
        "output_channels": OUTPUT_CHANNELS,
        "trust_region": _TRUST_REGION,
        "recommended_error_budget": _RECOMMENDED_ERROR_BUDGET,
        "served_backend": "onnx",
        "native_graph_fallback": False,
        "error_report_digest": content_hash_json(error_report),
        "error_report_media_type": "application/vnd.astro-mine.surrogate.error-report.v1+json",
    }


def main() -> None:
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (_FIXTURE_DIR / "model.onnx").write_bytes(build_onnx())
    report = build_error_report()
    (_FIXTURE_DIR / "error_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (_FIXTURE_DIR / "surrogate_attributes.json").write_text(
        json.dumps(build_attributes(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote fixtures to {_FIXTURE_DIR}")


if __name__ == "__main__":
    main()
