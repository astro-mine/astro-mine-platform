"""Drift guards between the canonical control-plane JSON Schema and the Pydantic models.

Mirrors ``tests/test_sadf_consistency.py``: (1) every enum ``$def`` matches its Python
enum; (2) a corpus of valid/invalid documents gets the same structural verdict from the
JSON Schema validator and from Pydantic, per root message type. Semantics (the tagged-
union checks) live in the loader, so the two are a single structural contract here.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.messages import enums
from astro_mine.core.messages.loader import load_schema
from astro_mine.core.messages.model import ActionBatch, ContactPlan
from astro_mine.core.schemas import schema_registry

ENUM_DEFS = {
    "ActionKind": enums.ActionKind,
    "ControlMode": enums.ControlMode,
    "TaskKind": enums.TaskKind,
    "SampleMethod": enums.SampleMethod,
    "ExcavationTool": enums.ExcavationTool,
    "ExcavationPattern": enums.ExcavationPattern,
    "ChargeSource": enums.ChargeSource,
    "NodeRole": enums.NodeRole,
    "ContactConfidence": enums.ContactConfidence,
}


@pytest.mark.parametrize("name", list(ENUM_DEFS), ids=list(ENUM_DEFS))
def test_schema_enum_matches_python_enum(name: str) -> None:
    schema = load_schema()
    assert name in schema["$defs"], f"{name} missing from schema $defs"
    schema_values = set(schema["$defs"][name]["enum"])
    python_values = {member.value for member in ENUM_DEFS[name]}
    assert schema_values == python_values


def _root_validator(root: str) -> Draft202012Validator:
    schema = load_schema()
    return Draft202012Validator(
        {"$ref": schema["$id"] + f"#/$defs/{root}"}, registry=schema_registry(schema)
    )


def _jsonschema_ok(data: Any, root: str) -> bool:
    return not list(_root_validator(root).iter_errors(data))


def _pydantic_ok(data: Any, model: type[ActionBatch] | type[ContactPlan]) -> bool:
    try:
        model.model_validate(data)
    except ValidationError:
        return False
    return True


def _excavate() -> dict[str, Any]:
    return {
        "region": {
            "frame": "psr",
            "center_m": {"x": 0, "y": 0, "z": 0},
            "dimensions_m": {"x": 1, "y": 1, "z": 1},
        },
        "tool": "bucket",
        "pattern": "trench",
        "target_volume_m3": 2.0,
    }


def _action_corpus() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    cases.append(("empty-batch", {}))
    cases.append(("empty-actions", {"actions": []}))
    cases.append(
        (
            "actuator-action",
            {
                "actions": [
                    {
                        "agent_id": "a",
                        "kind": "actuator",
                        "actuator": {"target": "j", "control_mode": "velocity", "setpoint": [1.0]},
                    }
                ]
            },
        )
    )
    cases.append(
        (
            "mode-action",
            {
                "actions": [
                    {
                        "agent_id": "a",
                        "kind": "mode",
                        "mode": {"mode": "drive", "params": {"x": "y"}},
                    }
                ]
            },
        )
    )
    cases.append(
        (
            "task-excavate",
            {
                "actions": [
                    {
                        "agent_id": "a",
                        "kind": "task",
                        "task": {"task_kind": "excavate", "excavate": _excavate()},
                    }
                ]
            },
        )
    )
    cases.append(
        (
            "task-with-null-fields",
            {
                "actions": [
                    {
                        "agent_id": "a",
                        "kind": "task",
                        "task": {"task_kind": "standby", "goto": None, "deadline_s": None},
                    }
                ]
            },
        )
    )
    # invalids (structural)
    cases.append(("unknown-field", {"actions": [{"agent_id": "a", "kind": "mode", "bogus": 1}]}))
    cases.append(("bad-kind", {"actions": [{"agent_id": "a", "kind": "warp"}]}))
    cases.append(
        (
            "bad-control-mode",
            {
                "actions": [
                    {
                        "agent_id": "a",
                        "kind": "actuator",
                        "actuator": {"target": "j", "control_mode": "warp"},
                    }
                ]
            },
        )
    )
    cases.append(("missing-agent", {"actions": [{"kind": "mode", "mode": {"mode": "drive"}}]}))
    cases.append(
        (
            "bad-excavate-tool",
            {
                "actions": [
                    {
                        "agent_id": "a",
                        "kind": "task",
                        "task": {
                            "task_kind": "excavate",
                            "excavate": {**_excavate(), "tool": "spork"},
                        },
                    }
                ]
            },
        )
    )
    cases.append(
        (
            "wrong-type-setpoint",
            {
                "actions": [
                    {
                        "agent_id": "a",
                        "kind": "actuator",
                        "actuator": {"target": "j", "control_mode": "velocity", "setpoint": "fast"},
                    }
                ]
            },
        )
    )
    # RFC-0007 typed ReferenceFrame sibling on Volume, resolved via the cross-file $ref.
    _region = {
        "frame": "psr",
        "center_m": {"x": 0, "y": 0, "z": 0},
        "dimensions_m": {"x": 1, "y": 1, "z": 1},
    }
    cases.append(
        (
            "excavate-region-frame-ref",
            {
                "actions": [
                    {
                        "agent_id": "a",
                        "kind": "task",
                        "task": {
                            "task_kind": "excavate",
                            "excavate": {
                                **_excavate(),
                                "region": {
                                    **_region,
                                    "frame_ref": {
                                        "name": "MOON_ME",
                                        "frame_class": "body_fixed",
                                        "center": "MOON",
                                    },
                                },
                            },
                        },
                    }
                ]
            },
        )
    )
    cases.append(
        (
            "excavate-region-bad-frame-ref",  # whitespace name -> both reject (deep cross-file)
            {
                "actions": [
                    {
                        "agent_id": "a",
                        "kind": "task",
                        "task": {
                            "task_kind": "excavate",
                            "excavate": {
                                **_excavate(),
                                "region": {
                                    **_region,
                                    "frame_ref": {"name": "MOON ME", "frame_class": "body_fixed"},
                                },
                            },
                        },
                    }
                ]
            },
        )
    )
    return cases


def _contact_corpus() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    cases.append(("empty-plan", {}))
    cases.append(
        (
            "full-plan",
            {
                "nodes": [{"id": "relay-1", "role": "space", "kind": "relay_orbiter"}],
                "intervals": [
                    {
                        "node_a": "r",
                        "node_b": "relay-1",
                        "start_tdb_s": 1.0,
                        "end_tdb_s": 2.0,
                        "confidence": "high",
                        "link_budget": {"margin_db": 3.0},
                    }
                ],
                "routes": [
                    {
                        "source": "r",
                        "dest": "g",
                        "hops": ["r", "relay-1", "g"],
                        "store_and_forward": True,
                    }
                ],
            },
        )
    )
    cases.append(
        (
            "null-link-budget",
            {
                "intervals": [
                    {
                        "node_a": "r",
                        "node_b": "s",
                        "start_tdb_s": 1.0,
                        "end_tdb_s": 2.0,
                        "link_budget": None,
                    }
                ]
            },
        )
    )
    # invalids
    cases.append(
        (
            "interval-missing-end",
            {"intervals": [{"node_a": "r", "node_b": "s", "start_tdb_s": 1.0}]},
        )
    )
    cases.append(("bad-node-role", {"nodes": [{"id": "n", "role": "deep_space"}]}))
    cases.append(
        (
            "bad-confidence",
            {
                "intervals": [
                    {
                        "node_a": "r",
                        "node_b": "s",
                        "start_tdb_s": 1.0,
                        "end_tdb_s": 2.0,
                        "confidence": "maybe",
                    }
                ]
            },
        )
    )
    cases.append(("unknown-route-field", {"routes": [{"source": "a", "dest": "b", "bogus": 1}]}))
    # RFC-0007 typed epoch/window siblings, resolved via the cross-file $ref to units.
    cases.append(
        (
            "plan-with-window",
            {
                "window": {
                    "start": {"tdb_seconds": 0.0, "scale": "tdb"},
                    "end": {"tdb_seconds": 1.0, "scale": "et"},
                }
            },
        )
    )
    cases.append(
        (
            "plan-window-bad-scale",  # utc is not a TimeScale -> both reject (deep cross-file)
            {
                "window": {
                    "start": {"tdb_seconds": 0.0, "scale": "utc"},
                    "end": {"tdb_seconds": 1.0, "scale": "tdb"},
                }
            },
        )
    )
    cases.append(
        (
            "route-earliest-delivery",
            {
                "routes": [
                    {
                        "source": "a",
                        "dest": "b",
                        "earliest_delivery": {"tdb_seconds": 10.0, "scale": "et"},
                    }
                ]
            },
        )
    )
    return cases


@pytest.mark.parametrize("case", _action_corpus(), ids=lambda c: c[0])
def test_action_jsonschema_and_pydantic_agree(case: tuple[str, dict[str, Any]]) -> None:
    _, data = case
    assert _jsonschema_ok(data, "ActionBatch") == _pydantic_ok(data, ActionBatch)


@pytest.mark.parametrize("case", _contact_corpus(), ids=lambda c: c[0])
def test_contact_jsonschema_and_pydantic_agree(case: tuple[str, dict[str, Any]]) -> None:
    _, data = case
    assert _jsonschema_ok(data, "ContactPlan") == _pydantic_ok(data, ContactPlan)


def test_schema_is_self_consistent_json() -> None:
    raw = json.dumps(load_schema())
    Draft202012Validator.check_schema(json.loads(raw))
