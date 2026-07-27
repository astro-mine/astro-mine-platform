"""STUDIO-03 — the design-loop worker: the other end of Cloud's file-based job contract.

``tests/test_cloud_dispatch.py`` exercises the worker the way Cloud does — as a *subprocess*,
which is the real integration but is invisible to the in-process coverage tracer. These tests
drive the same code in-process, and cover the failure paths a happy-path subprocess never hits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.objective import ObjectiveDocument
from astro_mine.studio.models import AssetSelection, DesignCandidate
from astro_mine.studio.orchestrate import SiblingClients
from astro_mine.studio.orchestrate.worker import (
    CLIENTS_ENV,
    DEFAULT_CLIENTS_FACTORY,
    OBJECTIVE_INPUT,
    OUTCOME_OUTPUT,
    REQUEST_INPUT,
    EvaluationOutcome,
    EvaluationRequest,
    encode_objective,
    load_clients,
    main,
    run_request,
)

_UNSAFE = DesignCandidate(
    id="unsafe",
    swarm=[AssetSelection(sadf_ref="rover", count=1)],
    decision_vector={"unsafe": 1.0},
)


# ---- the injected clients seam -------------------------------------------- #


def test_load_clients_defaults_to_the_local_bundle() -> None:
    assert isinstance(load_clients(), SiblingClients)


def test_load_clients_honors_an_explicit_factory() -> None:
    assert isinstance(load_clients(DEFAULT_CLIENTS_FACTORY), SiblingClients)


def test_load_clients_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam a real deployment uses to bind live Sim/Learn/Mind/Allocate/Guard/Bench clients."""
    monkeypatch.setenv(CLIENTS_ENV, DEFAULT_CLIENTS_FACTORY)
    assert isinstance(load_clients(), SiblingClients)


@pytest.mark.parametrize("spec", ["no_colon_here", "module:", ":factory"])
def test_load_clients_rejects_a_malformed_spec(spec: str) -> None:
    with pytest.raises(ValueError, match="must be 'module:factory'"):
        load_clients(spec)


# ---- one evaluation ------------------------------------------------------- #


def test_run_request_scores_a_candidate(
    objective_doc: ObjectiveDocument, clients: SiblingClients, candidate: DesignCandidate
) -> None:
    request = EvaluationRequest(candidate=candidate, seed=2, max_steps=3)
    outcome = run_request(request, objective_doc, clients=clients)

    assert outcome.ok
    assert outcome.error is None
    assert outcome.evaluated is not None
    assert outcome.evaluated.seed == 2


def test_run_request_reports_an_infeasible_candidate_as_a_result_not_a_crash(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    """A Guard veto is a *result*: the job ran, the candidate is infeasible, the reason survives."""
    outcome = run_request(
        EvaluationRequest(candidate=_UNSAFE, seed=0, max_steps=2), objective_doc, clients=clients
    )
    assert not outcome.ok
    assert outcome.evaluated is None
    assert outcome.error is not None and "certification" in outcome.error


# ---- the file-based job contract ------------------------------------------ #


def _stage(
    tmp_path: Path, request: EvaluationRequest, objective: ObjectiveDocument
) -> tuple[Path, Path]:
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    (inputs / REQUEST_INPUT).write_text(request.model_dump_json())
    (inputs / OBJECTIVE_INPUT).write_bytes(encode_objective(objective))
    return inputs, outputs


def test_main_round_trips_the_inputs_and_outputs_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    objective_doc: ObjectiveDocument,
    candidate: DesignCandidate,
) -> None:
    """`$ASTRO_MINE_INPUTS` -> evaluate -> `$ASTRO_MINE_OUTPUTS`, exactly as Cloud stages it."""
    inputs, outputs = _stage(
        tmp_path, EvaluationRequest(candidate=candidate, seed=4, max_steps=3), objective_doc
    )
    monkeypatch.setenv("ASTRO_MINE_INPUTS", str(inputs))
    monkeypatch.setenv("ASTRO_MINE_OUTPUTS", str(outputs))

    assert main() == 0  # exit 0: the worker *ran*

    outcome = EvaluationOutcome.model_validate_json((outputs / OUTCOME_OUTPUT).read_text())
    assert outcome.ok
    assert outcome.evaluated is not None
    assert outcome.evaluated.candidate.id == candidate.id
    assert outcome.evaluated.seed == 4


def test_main_exits_zero_for_an_infeasible_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, objective_doc: ObjectiveDocument
) -> None:
    """The contract that keeps Guard's reason alive across the job boundary (see the module doc):
    Cloud collects declared outputs only on exit 0, and a RunResult carries no error payload."""
    inputs, outputs = _stage(
        tmp_path, EvaluationRequest(candidate=_UNSAFE, seed=0, max_steps=2), objective_doc
    )
    monkeypatch.setenv("ASTRO_MINE_INPUTS", str(inputs))
    monkeypatch.setenv("ASTRO_MINE_OUTPUTS", str(outputs))

    assert main() == 0  # the job succeeded; the *candidate* did not

    outcome = EvaluationOutcome.model_validate_json((outputs / OUTCOME_OUTPUT).read_text())
    assert not outcome.ok
    assert outcome.error is not None and "certification" in outcome.error


def test_the_objective_rides_in_cores_byte_stable_wire_form(
    objective_doc: ObjectiveDocument,
) -> None:
    """Same bytes Studio content-addresses the objective by, so Cloud's provenance and Studio's
    cache key agree on one identity."""
    from astro_mine.core.objective import to_wire

    assert encode_objective(objective_doc) == to_wire(objective_doc)
