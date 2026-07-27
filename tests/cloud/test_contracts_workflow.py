"""WorkflowSpec -- a validated DAG of JobSpecs with a deterministic topological order."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission.jobspec import JobSpec
from astro_mine.cloud.submission.workflowspec import WorkflowSpec, WorkflowStep

IMAGE = ImageRef.parse("ghcr.io/astro-mine/x@sha256:" + "12" * 32)
JOB = JobSpec(image=IMAGE, command=["run"])


def _step(name: str, deps: list[str] | None = None) -> WorkflowStep:
    return WorkflowStep(name=name, job=JOB, depends_on=deps or [])


def test_topological_order_respects_dependencies() -> None:
    wf = WorkflowSpec(
        name="pipe",
        steps=[
            _step("score", ["aggregate"]),
            _step("aggregate", ["sim"]),
            _step("sim", ["generate"]),
            _step("generate"),
        ],
    )
    order = wf.topological_order()
    assert order.index("generate") < order.index("sim") < order.index("aggregate")
    assert order.index("aggregate") < order.index("score")


def test_fan_in_is_deterministic() -> None:
    wf = WorkflowSpec(
        name="fan",
        steps=[_step("a"), _step("b"), _step("join", ["a", "b"])],
    )
    assert wf.topological_order() == ["a", "b", "join"]
    assert set(wf.steps_by_name) == {"a", "b", "join"}


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        WorkflowSpec(name="dup", steps=[_step("a"), _step("a")])


def test_unknown_dependency_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown step"):
        WorkflowSpec(name="bad", steps=[_step("a", ["ghost"])])


def test_self_dependency_rejected() -> None:
    with pytest.raises(ValidationError, match="itself"):
        WorkflowSpec(name="self", steps=[_step("a", ["a"])])


def test_cycle_rejected() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        WorkflowSpec(name="loop", steps=[_step("a", ["b"]), _step("b", ["a"])])


def test_at_least_one_step_required() -> None:
    with pytest.raises(ValidationError):
        WorkflowSpec(name="empty", steps=[])
