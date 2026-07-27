"""The RunResult -- the backend-agnostic outcome of a submitted job.

Every backend returns the *same* :class:`RunResult`: a status, an exit code, the
content-addressed outputs, and the :class:`~astro_mine.cloud.artifacts.runcontext.RunContext`
provenance envelope (stored, its address recorded). Two backends running the same
deterministic :class:`~astro_mine.cloud.submission.jobspec.JobSpec` yield equal ``outputs``
and an equal ``run_context.content_address()`` -- that equivalence is the CLOUD-02
contract.

Backlog: RM-P0-CLOUD-02 -- https://github.com/astro-mine/astro-mine-cloud/issues/2
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.cloud.artifacts.runcontext import RunContext

__all__ = ["RunResult"]


class RunResult(BaseModel):
    """The outcome of a :func:`~astro_mine.cloud.submission.submit` call."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    exit_code: int
    outputs: dict[str, str] = Field(default_factory=dict)
    run_context_address: str
    run_context: RunContext

    @property
    def ok(self) -> bool:
        """Whether the job succeeded (exit code 0)."""
        return self.status == "succeeded"
