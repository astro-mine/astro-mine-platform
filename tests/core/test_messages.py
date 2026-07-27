"""Control-plane message tests (RM-P0-CORE-04): action + contact-plan families.

Byte-stable Protobuf round-trips, loud rejection of malformed messages, and the
tagged-union semantic checks for Action / TaskDirective (the fuller-depth action
model) and contact-interval ordering.
"""

from __future__ import annotations

import pytest

from astro_mine.core import messages as M
from astro_mine.core.messages import enums
from astro_mine.core.messages import model as m


def _excavate_action() -> m.Action:
    return m.Action(
        agent_id="excavator-2",
        kind=enums.ActionKind.TASK,
        task=m.TaskDirective(
            task_kind=enums.TaskKind.EXCAVATE,
            priority=3,
            excavate=m.ExcavateTask(
                region=m.Volume(
                    frame="psr",
                    center_m=m.Vec3(x=1.0, y=2.0, z=0.0),
                    dimensions_m=m.Vec3(x=4.0, y=4.0, z=1.0),
                ),
                tool=enums.ExcavationTool.BUCKET,
                pattern=enums.ExcavationPattern.TRENCH,
                target_volume_m3=2.5,
            ),
        ),
    )


def _full_batch() -> m.ActionBatch:
    return m.ActionBatch(
        actions=[
            m.Action(
                agent_id="rover-1",
                kind=enums.ActionKind.ACTUATOR,
                actuator=m.ActuatorCommand(
                    target="drill_z",
                    control_mode=enums.ControlMode.EFFORT,
                    setpoint=[120.0],
                    unit="N",
                ),
            ),
            m.Action(
                agent_id="rover-1",
                kind=enums.ActionKind.MODE,
                mode=m.ModeCommand(mode="prospect", params={"rate": "slow"}),
            ),
            _excavate_action(),
            m.Action(
                agent_id="hopper-1",
                kind=enums.ActionKind.TASK,
                task=m.TaskDirective(
                    task_kind=enums.TaskKind.HOP,
                    hop=m.HopTask(
                        launch_frame="body", target_point_m=m.Vec3(x=30, y=0, z=0), range_m=30.0
                    ),
                ),
            ),
            m.Action(
                agent_id="rover-3",
                kind=enums.ActionKind.TASK,
                task=m.TaskDirective(task_kind=enums.TaskKind.STANDBY),
            ),
        ]
    )


def _full_contact_plan() -> m.ContactPlan:
    return m.ContactPlan(
        nodes=[
            m.ContactNode(id="relay-1", role=enums.NodeRole.SPACE, kind="relay_orbiter"),
            m.ContactNode(id="dsn-goldstone", role=enums.NodeRole.GROUND, kind="ground_station"),
        ],
        intervals=[
            m.ContactInterval(
                node_a="rover-1",
                node_b="relay-1",
                start_tdb_s=1000.0,
                end_tdb_s=1300.0,
                max_rate_bps=2.0e6,
                min_latency_s=1.3,
                confidence=enums.ContactConfidence.HIGH,
                band="uhf",
                link_budget=m.LinkBudget(eirp_dbw=5.0, margin_db=3.2, modcod="gmsk_r1_2"),
            )
        ],
        routes=[
            m.Route(
                source="rover-1",
                dest="dsn-goldstone",
                hops=["rover-1", "relay-1", "dsn-goldstone"],
                store_and_forward=True,
                total_latency_s=1.3,
            )
        ],
        epoch_start_tdb_s=1000.0,
        epoch_end_tdb_s=2000.0,
    )


# --- round-trips (byte-stable) ---------------------------------------------------


def test_action_batch_roundtrips_byte_stably() -> None:
    batch = _full_batch()
    M.validate_action_batch(batch)
    wire = M.action_batch_to_wire(batch)
    restored = M.action_batch_from_wire(wire)
    assert restored == batch
    assert M.action_batch_to_wire(restored) == wire


def test_contact_plan_roundtrips_byte_stably() -> None:
    plan = _full_contact_plan()
    M.validate_contact_plan(plan)
    wire = M.contact_plan_to_wire(plan)
    restored = M.contact_plan_from_wire(wire)
    assert restored == plan
    assert M.contact_plan_to_wire(restored) == wire


def test_load_action_batch_from_yaml() -> None:
    text = """
    actions:
      - agent_id: rover-1
        kind: mode
        mode: { mode: drive }
    """
    batch = M.load_action_batch(text)
    assert batch.actions[0].mode is not None
    assert batch.actions[0].mode.mode == "drive"


# --- loud rejection --------------------------------------------------------------


def test_unknown_field_rejected() -> None:
    with pytest.raises(M.MessagesValidationError):
        M.validate_action_batch({"actions": [{"agent_id": "a", "kind": "mode", "bogus": 1}]})


def test_bad_action_kind_enum_rejected() -> None:
    with pytest.raises(M.MessagesValidationError):
        M.validate_action_batch({"actions": [{"agent_id": "a", "kind": "teleport"}]})


def test_missing_required_field_rejected() -> None:
    # ContactInterval requires start/end.
    with pytest.raises(M.MessagesValidationError):
        M.validate_contact_plan({"intervals": [{"node_a": "x", "node_b": "y"}]})


def test_bad_task_kind_enum_rejected() -> None:
    with pytest.raises(M.MessagesValidationError):
        M.validate_action_batch(
            {"actions": [{"agent_id": "a", "kind": "task", "task": {"task_kind": "teleport"}}]}
        )


# --- tagged-union semantics ------------------------------------------------------


def test_action_kind_payload_mismatch_rejected() -> None:
    # kind=actuator but a mode payload is set instead.
    bad = {"actions": [{"agent_id": "a", "kind": "actuator", "mode": {"mode": "drive"}}]}
    with pytest.raises(M.MessagesValidationError, match="requires exactly the 'actuator'"):
        M.validate_action_batch(bad)


def test_action_kind_missing_payload_rejected() -> None:
    bad = {"actions": [{"agent_id": "a", "kind": "task"}]}
    with pytest.raises(M.MessagesValidationError, match="requires exactly the 'task'"):
        M.validate_action_batch(bad)


def test_task_kind_payload_mismatch_rejected() -> None:
    # task_kind=goto but an excavate payload is attached.
    bad = {
        "actions": [
            {
                "agent_id": "a",
                "kind": "task",
                "task": {
                    "task_kind": "goto",
                    "excavate": {
                        "region": {
                            "frame": "f",
                            "center_m": {"x": 0, "y": 0, "z": 0},
                            "dimensions_m": {"x": 1, "y": 1, "z": 1},
                        },
                        "tool": "bucket",
                        "pattern": "trench",
                    },
                },
            }
        ]
    }
    with pytest.raises(M.MessagesValidationError, match="requires exactly the 'goto'"):
        M.validate_action_batch(bad)


def test_custom_task_requires_directive() -> None:
    bad = {"actions": [{"agent_id": "a", "kind": "task", "task": {"task_kind": "custom"}}]}
    with pytest.raises(M.MessagesValidationError, match="requires a 'directive'"):
        M.validate_action_batch(bad)


def test_paramless_task_rejects_payload() -> None:
    bad = {
        "actions": [
            {
                "agent_id": "a",
                "kind": "task",
                "task": {
                    "task_kind": "standby",
                    "charge": {"source": "solar"},
                },
            }
        ]
    }
    with pytest.raises(M.MessagesValidationError, match="carries no payload"):
        M.validate_action_batch(bad)


def test_contact_interval_inverted_window_rejected() -> None:
    bad = {"intervals": [{"node_a": "x", "node_b": "y", "start_tdb_s": 100.0, "end_tdb_s": 50.0}]}
    with pytest.raises(M.MessagesValidationError, match="precedes start"):
        M.validate_contact_plan(bad)


def test_custom_task_with_directive_valid() -> None:
    good = m.ActionBatch(
        actions=[
            m.Action(
                agent_id="a",
                kind=enums.ActionKind.TASK,
                task=m.TaskDirective(
                    task_kind=enums.TaskKind.CUSTOM, directive="self_test", params={"depth": "1"}
                ),
            )
        ]
    )
    M.validate_action_batch(good)
