"""WorkflowSpec -- an explicit DAG of JobSpecs with fan-in.

A :class:`WorkflowSpec` is a directed acyclic graph of named steps, each a
:class:`~astro_mine.cloud.submission.jobspec.JobSpec`, with ``depends_on`` edges expressing
fan-out/fan-in (e.g. generate scenarios -> run sims -> aggregate -> score) (``cloud.md`` §3).
The graph is validated at construction -- unique names, resolvable dependencies, no cycles --
and exposes a deterministic topological order. It compiles to an Argo ``Workflow`` in
:mod:`astro_mine.cloud.engines.argo`.

Backlog: RM-P1-CLOUD-02 -- astro-mine-cloud#13
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.cloud.submission.jobspec import JobSpec

__all__ = ["WorkflowSpec", "WorkflowStep"]


class WorkflowStep(BaseModel):
    """One node in a :class:`WorkflowSpec`: a named JobSpec and its upstream dependencies."""

    model_config = ConfigDict(extra="forbid")

    name: str
    job: JobSpec
    depends_on: list[str] = Field(default_factory=list)


class WorkflowSpec(BaseModel):
    """A DAG of JobSpecs -- validated acyclic, with a deterministic topological order."""

    model_config = ConfigDict(extra="forbid")

    name: str
    steps: list[WorkflowStep] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_dag(self) -> WorkflowSpec:
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("workflow step names must be unique")
        known = set(names)
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in known:
                    raise ValueError(f"step {step.name!r} depends on unknown step {dep!r}")
                if dep == step.name:
                    raise ValueError(f"step {step.name!r} depends on itself")
        self.topological_order()  # raises ValueError on a cycle
        return self

    @property
    def steps_by_name(self) -> dict[str, WorkflowStep]:
        """The steps keyed by name (names are validated unique)."""
        return {step.name: step for step in self.steps}

    def topological_order(self) -> list[str]:
        """Return step names in a dependency-respecting order (Kahn's algorithm).

        Ties are broken by the steps' declared order, so the result is deterministic. Raises
        ``ValueError`` if the graph contains a cycle.
        """
        order = [step.name for step in self.steps]
        remaining = {step.name: set(step.depends_on) for step in self.steps}
        result: list[str] = []
        while remaining:
            ready = [name for name in order if name in remaining and not remaining[name]]
            if not ready:
                raise ValueError(
                    f"workflow {self.name!r} has a dependency cycle among {set(remaining)}"
                )
            for name in ready:
                result.append(name)
                del remaining[name]
                for deps in remaining.values():
                    deps.discard(name)
        return result
