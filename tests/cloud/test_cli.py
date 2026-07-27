"""The astro-mine-cloud CLI -- submit / expand / compile / sweep / workflow / backends."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission.cli import main
from astro_mine.cloud.submission.jobspec import JobSpec
from astro_mine.cloud.submission.sweepspec import SweepSpec
from astro_mine.cloud.submission.workflowspec import WorkflowSpec, WorkflowStep

IMAGE = ImageRef.parse("ghcr.io/astro-mine/x@sha256:" + "9a" * 32)
_ECHO_INPUT = (
    "import os, pathlib;"
    "i=pathlib.Path(os.environ['ASTRO_MINE_INPUTS']);"
    "o=pathlib.Path(os.environ['ASTRO_MINE_OUTPUTS']);"
    "(o/'y.txt').write_bytes((i/'x.txt').read_bytes())"
)


def _write(path: Path, model: JobSpec | SweepSpec | WorkflowSpec) -> str:
    path.write_text(model.model_dump_json())
    return str(path)


def test_submit_stages_inputs_and_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "x.txt"
    src.write_bytes(b"payload")
    job = JobSpec(image=IMAGE, command=[sys.executable, "-c", _ECHO_INPUT], outputs=["y.txt"])
    spec = _write(tmp_path / "job.json", job)
    code = main(["submit", spec, "--store", str(tmp_path / "store"), "--input", f"x.txt={src}"])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "succeeded"
    assert "y.txt" in result["outputs"]


def test_submit_propagates_a_failing_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = JobSpec(image=IMAGE, command=[sys.executable, "-c", "raise SystemExit(3)"])
    code = main(["submit", _write(tmp_path / "job.json", job), "--store", str(tmp_path / "s")])
    assert code == 3


def test_bad_input_spec_is_rejected(tmp_path: Path) -> None:
    job = JobSpec(image=IMAGE, command=["true"])
    with pytest.raises(SystemExit, match="name=path"):
        main(["submit", _write(tmp_path / "job.json", job), "--input", "no-equals"])


def test_expand_previews_a_sweep(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sweep = SweepSpec(base=JobSpec(image=IMAGE, command=["run"]), grid={"lr": [0.1, 0.2, 0.3]})
    assert main(["expand", _write(tmp_path / "sweep.json", sweep)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["size"] == 3
    assert len(out["jobs"]) == 3


def test_compile_auto_and_forced_engine(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job = JobSpec(image=IMAGE, command=["run"])
    spec = _write(tmp_path / "job.json", job)
    assert main(["compile", spec, "--namespace", "acme"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "Job"
    assert main(["compile", spec, "--engine", "ray"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "RayJob"


def test_sweep_and_workflow_compile(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = JobSpec(image=IMAGE, command=["run"])
    sweep = SweepSpec(base=base, grid={"lr": [0.1, 0.2]})
    assert main(["sweep", _write(tmp_path / "sweep.json", sweep)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "Workflow"

    wf = WorkflowSpec(name="pipe", steps=[WorkflowStep(name="a", job=base)])
    assert main(["workflow", _write(tmp_path / "wf.json", wf)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "Workflow"


def test_backends_lists_registered(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backends"]) == 0
    assert "local" in capsys.readouterr().out.split()
