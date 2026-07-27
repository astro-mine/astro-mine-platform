"""Plan / ContingentPlan schema — models, loader, and schema/model consistency (RFC-0006)."""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import Action, ActionBatch, TaskDirective
from astro_mine.core.plan import (
    PLAN_VERSION,
    Assumption,
    ContingencyBranch,
    ContingentPlan,
    Plan,
    PlanDocument,
    PlanValidationError,
    PlanValidity,
    load_plan,
    load_schema,
    validate_plan,
)


def _plan() -> Plan:
    actions = ActionBatch(
        actions=[
            Action(
                agent_id="r0", kind=ActionKind.TASK, task=TaskDirective(task_kind=TaskKind.STANDBY)
            )
        ]
    )
    return Plan(
        plan_id="mission@0.0",
        tier="mission",
        validity=PlanValidity(issued_at_s=0.0, horizon_s=3.0),
        actions=actions,
        assumptions=[Assumption(key="earth_contact", description="ground link", holds=True)],
    )


def _document() -> PlanDocument:
    contingent = ContingentPlan(
        base=_plan(),
        branches=[
            ContingencyBranch(trigger="comms_lost", action="hold_cached"),
            ContingencyBranch(trigger="plan_expired", action="reconcile"),
        ],
    )
    return PlanDocument(plan_version="0.1", plan=contingent)


EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples" / "plan").glob("*.plan.yaml"))


def test_plan_version() -> None:
    assert PLAN_VERSION == "0.1"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_corpus_loads(path: Path) -> None:
    # The checked-in corpus is the shared parity fixture: the Rust fast path validates the same
    # files in `cargo test` (validator/rust/src/lib.rs::plan_corpus_is_structurally_valid).
    doc = load_plan(path.read_text(encoding="utf-8"))
    assert doc.plan_version == "0.1"
    assert doc.plan.base.plan_id


def test_examples_corpus_is_not_empty() -> None:
    assert EXAMPLES, "the plan example corpus is the Rust parity fixture; it must not be empty"


def test_document_round_trips() -> None:
    doc = _document()
    assert PlanDocument.model_validate(doc.model_dump(mode="json")) == doc


def test_load_and_validate_accepts_valid_document() -> None:
    payload = _document().model_dump(mode="json")
    import json

    loaded = load_plan(json.dumps(payload))
    assert loaded.plan.base.tier == "mission"
    assert [b.trigger for b in loaded.plan.branches] == ["comms_lost", "plan_expired"]
    validate_plan(payload)  # dict
    validate_plan(loaded)  # typed document
    validate_plan(json.dumps(payload))  # text


def test_load_accepts_bytes_and_yaml() -> None:
    import json

    payload = json.dumps(_document().model_dump(mode="json"))
    assert load_plan(payload.encode("utf-8")).plan_version == "0.1"  # bytes
    yaml_text = (
        "plan_version: '0.1'\n"
        "plan:\n  base:\n    plan_id: p\n    tier: control\n    validity: {issued_at_s: 0.0}\n"
    )
    assert load_plan(yaml_text).plan.base.tier == "control"  # YAML


def test_standing_plan_allows_null_horizon() -> None:
    plan = Plan(plan_id="p", tier="control", validity=PlanValidity(issued_at_s=1.0))
    assert plan.validity.horizon_s is None


def test_extra_fields_are_rejected() -> None:
    payload = _document().model_dump(mode="json")
    payload["plan"]["base"]["bogus"] = 1
    with pytest.raises(PlanValidationError):
        validate_plan(payload)


def test_wrong_version_is_rejected() -> None:
    payload = _document().model_dump(mode="json")
    payload["plan_version"] = "0.2"
    with pytest.raises(PlanValidationError):
        validate_plan(payload)


def test_duplicate_branch_triggers_are_rejected() -> None:
    payload = _document().model_dump(mode="json")
    payload["plan"]["branches"].append({"trigger": "comms_lost", "action": "safe_idle"})
    with pytest.raises(PlanValidationError, match="duplicate contingency trigger"):
        validate_plan(payload)


def test_blank_branch_labels_are_rejected() -> None:
    payload = _document().model_dump(mode="json")
    payload["plan"]["branches"] = [{"trigger": "  ", "action": "hold_cached"}]
    with pytest.raises(PlanValidationError, match="non-empty label"):
        validate_plan(payload)
    payload["plan"]["branches"] = [{"trigger": "comms_lost", "action": ""}]
    with pytest.raises(PlanValidationError):
        validate_plan(payload)


def test_non_mapping_and_bad_type_are_rejected() -> None:
    with pytest.raises(PlanValidationError):
        validate_plan(12345)
    with pytest.raises(PlanValidationError):
        load_plan("- not-a-mapping")


def test_schema_and_model_agree_on_a_canonical_document() -> None:
    # The pydantic dump validates against the canonical JSON Schema, and every top-level model
    # field appears as a schema property — the two stay one structural contract (RFC-0006).
    schema = load_schema()
    plan_props = set(schema["$defs"]["Plan"]["properties"])
    assert set(Plan.model_fields) <= plan_props
    contingent_props = set(schema["$defs"]["ContingentPlan"]["properties"])
    assert set(ContingentPlan.model_fields) <= contingent_props
    validate_plan(_document().model_dump(mode="json"))  # dump passes the schema gate
