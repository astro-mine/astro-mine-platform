"""Behavioural tests for the run-provenance schema, models, and loader (issue #18)."""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.provenance import (
    ErrorBudgetOutcome,
    RunProvenance,
    RunProvenanceDocument,
    RunProvenanceValidationError,
    load_run_provenance,
    validate_run_provenance,
)

EXAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "examples" / "run-provenance").glob(
        "*.run-provenance.yaml"
    )
)


def test_empty_run_provenance_is_valid() -> None:
    # Every field is optional: a bare, versioned envelope round-trips.
    doc = RunProvenanceDocument(run_provenance_version="0.1", run_provenance=RunProvenance())
    assert doc.run_provenance.input_hashes == []
    assert doc.run_provenance.seeds == {}
    assert doc.run_provenance.error_budget_outcomes == []


def test_full_record_round_trips() -> None:
    doc = RunProvenanceDocument(
        run_provenance_version="0.1",
        run_provenance=RunProvenance(
            run_id="r1",
            input_hashes=["sha256:00", "sha256:11"],
            engine_versions={"orbital": "basilisk-2.2.0"},
            fidelity_tiers={"orbital": "kinematic"},
            seed=42,
            seeds={"episode": 42},
            error_budget_outcomes=[
                ErrorBudgetOutcome(name="orbital_position", within_budget=True, value=1.0)
            ],
        ),
    )
    restored = RunProvenanceDocument.model_validate(doc.model_dump(mode="json"))
    assert restored == doc


def test_loader_accepts_examples() -> None:
    assert EXAMPLES, "expected example run-provenance documents"
    for path in EXAMPLES:
        doc = load_run_provenance(path.read_text(encoding="utf-8"))
        assert doc.run_provenance_version == "0.1"


def test_loader_rejects_unknown_field() -> None:
    with pytest.raises(RunProvenanceValidationError):
        load_run_provenance("run_provenance_version: '0.1'\nrun_provenance:\n  bogus: 1\n")


def test_loader_rejects_bad_version_const() -> None:
    with pytest.raises(RunProvenanceValidationError):
        load_run_provenance("run_provenance_version: '0.2'\nrun_provenance: {}\n")


def test_loader_rejects_wrong_type() -> None:
    with pytest.raises(RunProvenanceValidationError):
        load_run_provenance("run_provenance_version: '0.1'\nrun_provenance:\n  seed: not-an-int\n")


def test_error_budget_outcome_requires_name_and_verdict() -> None:
    with pytest.raises(RunProvenanceValidationError):
        load_run_provenance(
            "run_provenance_version: '0.1'\n"
            "run_provenance:\n"
            "  error_budget_outcomes:\n"
            "    - value: 1.0\n"  # missing required name + within_budget
        )


def test_validate_accepts_dict_text_and_model() -> None:
    data = {"run_provenance_version": "0.1", "run_provenance": {"seed": 1}}
    validate_run_provenance(data)  # dict
    validate_run_provenance("run_provenance_version: '0.1'\nrun_provenance: {}\n")  # text
    validate_run_provenance(
        RunProvenanceDocument(run_provenance_version="0.1", run_provenance=RunProvenance())
    )  # model


def test_validate_rejects_unsupported_type() -> None:
    with pytest.raises(RunProvenanceValidationError):
        validate_run_provenance(42)  # type: ignore[arg-type]


def test_loader_accepts_bytes() -> None:
    doc = load_run_provenance(b"run_provenance_version: '0.1'\nrun_provenance:\n  seed: 7\n")
    assert doc.run_provenance.seed == 7


def test_loader_rejects_non_mapping() -> None:
    with pytest.raises(RunProvenanceValidationError, match="must be a YAML/JSON mapping"):
        load_run_provenance("42")
