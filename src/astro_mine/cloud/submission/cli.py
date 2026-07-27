"""The ``astro-mine-cloud`` CLI -- submit and compile jobs/sweeps/workflows.

A dependency-free ``argparse`` front door to the submission library (``cloud.md`` §3): submit
a JobSpec through any backend (``local``/``docker``/``cluster``), preview a sweep's expansion,
or compile a JobSpec/SweepSpec/WorkflowSpec to the engine object it would run as. Specs are
JSON files validated against the pydantic contracts. ``submit --input name=path`` resolves a
local file to a content hash at submit time and records it on the job -- the client's half of
the content-addressing contract (``cloud.md`` §5).

Backlog: RM-P1-CLOUD-02 -- https://github.com/astro-mine/astro-mine-cloud/issues/13
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
from astro_mine.cloud.engines import compile_sweep, compile_workflow, get_engine, select_engine
from astro_mine.cloud.submission import submit
from astro_mine.cloud.submission.backend import registered_backends
from astro_mine.cloud.submission.jobspec import JobSpec
from astro_mine.cloud.submission.sweepspec import SweepSpec
from astro_mine.cloud.submission.workflowspec import WorkflowSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astro_mine.cloud.artifacts.store import ArtifactStore

__all__ = ["main"]


def _load(path: str, model: type[JobSpec] | type[SweepSpec] | type[WorkflowSpec]):  # type: ignore[no-untyped-def]
    return model.model_validate_json(Path(path).read_text())


def _resolve_inputs(job: JobSpec, inputs: list[str], store: ArtifactStore) -> JobSpec:
    """Stage each ``name=path`` local file into *store* and record its content address."""
    resolved = dict(job.inputs)
    for item in inputs:
        name, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--input must be name=path, got {item!r}")
        resolved[name] = store.put(Path(path).read_bytes())
    return job.model_copy(update={"inputs": resolved})


def _cmd_submit(args: argparse.Namespace) -> int:
    store = FilesystemArtifactStore(args.store) if args.store else FilesystemArtifactStore()
    job = _resolve_inputs(_load(args.spec, JobSpec), args.input, store)
    result = submit(job, backend=args.backend, store=store)
    print(result.model_dump_json(indent=2))
    return result.exit_code


def _cmd_expand(args: argparse.Namespace) -> int:
    sweep = _load(args.spec, SweepSpec)
    variants = sweep.expand()
    print(
        json.dumps({"size": len(variants), "jobs": [v.model_dump(mode="json") for v in variants]})
    )
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    job = _load(args.spec, JobSpec)
    engine = args.engine if args.engine else select_engine(job)
    print(json.dumps(get_engine(engine).compile(job, namespace=args.namespace), indent=2))
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    sweep = _load(args.spec, SweepSpec)
    print(json.dumps(compile_sweep(sweep, namespace=args.namespace), indent=2))
    return 0


def _cmd_workflow(args: argparse.Namespace) -> int:
    workflow = _load(args.spec, WorkflowSpec)
    print(json.dumps(compile_workflow(workflow, namespace=args.namespace), indent=2))
    return 0


def _cmd_backends(_args: argparse.Namespace) -> int:
    print("\n".join(registered_backends()))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astro-mine-cloud", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="submit a JobSpec through a backend")
    p_submit.add_argument("spec", help="path to a JobSpec JSON file")
    p_submit.add_argument("--backend", default="local", help="backend name (default: local)")
    p_submit.add_argument("--store", default=None, help="artifact store root")
    p_submit.add_argument(
        "--input", action="append", default=[], metavar="NAME=PATH", help="stage a local input file"
    )
    p_submit.set_defaults(func=_cmd_submit)

    p_expand = sub.add_parser("expand", help="preview a SweepSpec's expansion")
    p_expand.add_argument("spec", help="path to a SweepSpec JSON file")
    p_expand.set_defaults(func=_cmd_expand)

    p_compile = sub.add_parser("compile", help="compile a JobSpec to an engine manifest")
    p_compile.add_argument("spec", help="path to a JobSpec JSON file")
    p_compile.add_argument("--engine", default=None, help="force an engine (default: auto)")
    p_compile.add_argument("--namespace", default="default", help="target namespace")
    p_compile.set_defaults(func=_cmd_compile)

    p_sweep = sub.add_parser("sweep", help="compile a SweepSpec to an Argo Workflow")
    p_sweep.add_argument("spec", help="path to a SweepSpec JSON file")
    p_sweep.add_argument("--namespace", default="default", help="target namespace")
    p_sweep.set_defaults(func=_cmd_sweep)

    p_workflow = sub.add_parser("workflow", help="compile a WorkflowSpec to an Argo Workflow")
    p_workflow.add_argument("spec", help="path to a WorkflowSpec JSON file")
    p_workflow.add_argument("--namespace", default="default", help="target namespace")
    p_workflow.set_defaults(func=_cmd_workflow)

    p_backends = sub.add_parser("backends", help="list registered backends")
    p_backends.set_defaults(func=_cmd_backends)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse *argv* and run the selected subcommand; return its exit code."""
    args = _build_parser().parse_args(argv)
    return int(args.func(args))
