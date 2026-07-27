"""The feasibility contract: status <-> plan / infeasibility-certificate (RM-P1-ALLOC-01).

The result type always carries either a feasible plan or an explicit infeasibility-certificate
slot (acceptance criterion). The ``Allocation`` model validator enforces it structurally; the
request/schedule validators enforce the well-formedness the contract rests on.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.allocate import (
    Allocation,
    AllocationProvenance,
    AllocationStatus,
    AssetSchedule,
    InfeasibilityCertificate,
    ScheduledTask,
    Task,
    TimeWindow,
    ValueEstimate,
)
from astro_mine.core.messages.enums import TaskKind
from tests.allocate.factories import anchor_request

_PROV = AllocationProvenance(ir_version="0.1.0", backend="trivial-stub")


def _schedule() -> list[AssetSchedule]:
    return [
        AssetSchedule(
            asset_id="rover-1",
            tasks=[ScheduledTask(task_id="t", kind=TaskKind.PROSPECT, start_s=0.0, end_s=1.0)],
        )
    ]


# --- Allocation.status <-> plan/certificate ---------------------------------------


@pytest.mark.parametrize("status", [AllocationStatus.OPTIMAL, AllocationStatus.FEASIBLE])
def test_feasible_status_requires_a_plan(status: AllocationStatus) -> None:
    with pytest.raises(ValidationError, match="requires a plan"):
        Allocation(status=status, plan=None, provenance=_PROV)


@pytest.mark.parametrize("status", [AllocationStatus.OPTIMAL, AllocationStatus.FEASIBLE])
def test_feasible_status_forbids_a_certificate(status: AllocationStatus) -> None:
    with pytest.raises(ValidationError, match="must not carry an infeasibility_certificate"):
        Allocation(
            status=status,
            plan=_schedule(),
            provenance=_PROV,
            infeasibility_certificate=InfeasibilityCertificate(),
        )


def test_infeasible_status_forbids_a_plan() -> None:
    with pytest.raises(ValidationError, match="must not carry a plan"):
        Allocation(status=AllocationStatus.INFEASIBLE, plan=_schedule(), provenance=_PROV)


def test_result_never_carries_both_plan_and_certificate() -> None:
    with pytest.raises(ValidationError, match="not carry both a plan and a certificate"):
        Allocation(
            status=AllocationStatus.TIMEOUT,
            plan=_schedule(),
            provenance=_PROV,
            infeasibility_certificate=InfeasibilityCertificate(),
        )


def test_valid_feasible_result() -> None:
    alloc = Allocation(
        status=AllocationStatus.FEASIBLE,
        plan=_schedule(),
        realized_objective=1.0,
        provenance=_PROV,
    )
    assert alloc.infeasibility_certificate is None


def test_valid_infeasible_result_with_reserved_empty_slot() -> None:
    # RM-P1-ALLOC-01 leaves the certificate slot None (reserved); RM-P1-ALLOC-06 fills the IIS.
    alloc = Allocation(status=AllocationStatus.INFEASIBLE, provenance=_PROV)
    assert alloc.plan is None
    assert alloc.infeasibility_certificate is None


def test_valid_infeasible_result_with_certificate() -> None:
    alloc = Allocation(
        status=AllocationStatus.INFEASIBLE,
        provenance=_PROV,
        infeasibility_certificate=InfeasibilityCertificate(
            constraint_ids=["cover::drill-1"], task_ids=["drill-1"], explanation="no eligible asset"
        ),
    )
    assert alloc.infeasibility_certificate is not None


def test_anytime_timeout_may_carry_an_incumbent_plan() -> None:
    alloc = Allocation(
        status=AllocationStatus.TIMEOUT, plan=_schedule(), realized_objective=1.0, provenance=_PROV
    )
    assert alloc.plan is not None


def test_unknown_status_without_plan_is_valid() -> None:
    assert Allocation(status=AllocationStatus.UNKNOWN, provenance=_PROV).plan is None


# --- well-formedness the contract rests on ---------------------------------------


def test_scheduled_task_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="precedes start_s"):
        ScheduledTask(task_id="t", kind=TaskKind.HAUL, start_s=5.0, end_s=1.0)


def test_asset_schedule_rejects_out_of_order_tasks() -> None:
    with pytest.raises(ValidationError, match="not time-ordered"):
        AssetSchedule(
            asset_id="rover-1",
            tasks=[
                ScheduledTask(task_id="b", kind=TaskKind.HAUL, start_s=10.0, end_s=10.0),
                ScheduledTask(task_id="a", kind=TaskKind.HAUL, start_s=1.0, end_s=1.0),
            ],
        )


def test_time_window_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="precedes start_s"):
        TimeWindow(start_s=100.0, end_s=1.0)


def test_request_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate task_id"):
        anchor_request(
            tasks=[
                Task(task_id="dup", kind=TaskKind.PROSPECT, value=ValueEstimate(mean=1.0)),
                Task(task_id="dup", kind=TaskKind.HAUL, value=ValueEstimate(mean=1.0)),
            ]
        )


def test_request_rejects_duplicate_asset_ids() -> None:
    from astro_mine.allocate import AssetRef

    with pytest.raises(ValidationError, match="duplicate asset_id"):
        anchor_request(assets=[AssetRef(asset_id="dup"), AssetRef(asset_id="dup")])


def test_request_rejects_self_precedence() -> None:
    with pytest.raises(ValidationError, match="lists itself in precedence"):
        anchor_request(
            tasks=[
                Task(
                    task_id="a",
                    kind=TaskKind.PROSPECT,
                    precedence=["a"],
                    value=ValueEstimate(mean=1.0),
                )
            ]
        )


def test_request_rejects_unknown_precedence_reference() -> None:
    with pytest.raises(ValidationError, match="references unknown task"):
        anchor_request(
            tasks=[
                Task(
                    task_id="a",
                    kind=TaskKind.PROSPECT,
                    precedence=["ghost"],
                    value=ValueEstimate(mean=1.0),
                )
            ]
        )


def test_request_rejects_a_precedence_cycle() -> None:
    with pytest.raises(ValidationError, match="contains a cycle"):
        anchor_request(
            tasks=[
                Task(
                    task_id="a",
                    kind=TaskKind.PROSPECT,
                    precedence=["b"],
                    value=ValueEstimate(mean=1.0),
                ),
                Task(
                    task_id="b",
                    kind=TaskKind.HAUL,
                    precedence=["a"],
                    value=ValueEstimate(mean=1.0),
                ),
            ]
        )
