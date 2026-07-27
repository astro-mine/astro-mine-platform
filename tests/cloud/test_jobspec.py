"""JobSpec: defaults, strictness, and input/output name + address validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import JobSpec

IMAGE = ImageRef.parse("ghcr.io/astro-mine/astro-mine-sim@sha256:" + "ab" * 32)
ADDR = "sha256:" + "cd" * 32


def test_minimal_job_and_reserved_fields_default_empty() -> None:
    job = JobSpec(image=IMAGE, command=["run"])
    assert job.command == ["run"]
    assert job.inputs == {} and job.outputs == []
    assert job.seed is None and job.core_interface_version is None
    # reserved P1 cluster fields default empty/None (populated without a schema bump)
    assert job.resources == {} and job.tenant is None
    assert job.priority is None and job.budget is None


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        JobSpec(image=IMAGE, nope=1)  # type: ignore[call-arg]


def test_inputs_must_be_content_addresses() -> None:
    JobSpec(image=IMAGE, inputs={"x.txt": ADDR})  # ok
    with pytest.raises(ValidationError):
        JobSpec(image=IMAGE, inputs={"x.txt": "not-an-address"})


@pytest.mark.parametrize("bad", ["", "/abs.txt", "../escape.txt", "a\\b.txt", "sub/../../x"])
def test_unsafe_input_names_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        JobSpec(image=IMAGE, inputs={bad: ADDR})


@pytest.mark.parametrize("bad", ["", "/abs", "../out", "dir/../../out"])
def test_unsafe_output_names_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        JobSpec(image=IMAGE, outputs=[bad])


def test_nested_relative_names_allowed() -> None:
    job = JobSpec(image=IMAGE, inputs={"a/b/x.txt": ADDR}, outputs=["a/b/y.txt"])
    assert job.inputs["a/b/x.txt"] == ADDR
    assert job.outputs == ["a/b/y.txt"]


def test_core_interface_version_admits_compatible() -> None:
    assert JobSpec(image=IMAGE, core_interface_version="0.1.0").core_interface_version == "0.1.0"
    # patch is ignored under the 0.y exact-minor rule
    assert JobSpec(image=IMAGE, core_interface_version="0.1.9").core_interface_version == "0.1.9"


@pytest.mark.parametrize("bad", ["0.2.0", "0.0.0", "1.0.0", "0.1", "not-a-version"])
def test_core_interface_version_rejects_incompatible(bad: str) -> None:
    with pytest.raises(ValidationError):
        JobSpec(image=IMAGE, core_interface_version=bad)
