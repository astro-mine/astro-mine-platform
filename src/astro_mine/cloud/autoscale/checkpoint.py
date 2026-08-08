"""Content-addressed checkpoint-to-resume -- surviving a spot eviction.

Spot/preemptible compute is the default (``cloud.md`` §2 principle 5), so a job must survive
losing its node. :func:`run_checkpointed` runs a stepwise workload, writing a
**content-addressed** :class:`Checkpoint` to the artifact store after each step; on a
preemption it raises :class:`Preempted` carrying the last checkpoint, and a resumed run
continues from it -- losing at most one interval and reproducing the uninterrupted result
byte-for-byte (``cloud.md`` §8, §2 principle 4). Because checkpoints are content-addressed,
resume is deterministic: the same step sequence yields the same final state whether or not it
was interrupted.

Backlog: RM-P1-CLOUD-03 -- astro-mine-cloud#14
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Callable

    from astro_mine.core.artifacts import ArtifactStore

__all__ = ["Checkpoint", "CheckpointStore", "Preempted", "run_checkpointed"]


class Checkpoint(BaseModel):
    """A saved step: the step number and the content address of the state blob."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    state_address: str


class CheckpointStore:
    """Reads/writes content-addressed checkpoint state to an :class:`ArtifactStore`."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def save(self, step: int, state: bytes) -> Checkpoint:
        """Store *state* by content address and return the :class:`Checkpoint`."""
        return Checkpoint(step=step, state_address=self._store.put(state))

    def load(self, checkpoint: Checkpoint) -> bytes:
        """Return the state bytes referenced by *checkpoint*."""
        return self._store.get(checkpoint.state_address)


class Preempted(Exception):
    """Raised to simulate a spot eviction; carries the last committed checkpoint."""

    def __init__(self, checkpoint: Checkpoint | None) -> None:
        super().__init__(f"preempted after checkpoint {checkpoint}")
        self.checkpoint = checkpoint


def run_checkpointed(
    *,
    steps: int,
    store: ArtifactStore,
    step_fn: Callable[[bytes, int], bytes],
    initial: bytes = b"",
    resume: Checkpoint | None = None,
    preempt_at: int | None = None,
) -> tuple[bytes, Checkpoint | None]:
    """Run *steps* of *step_fn*, checkpointing each step; resume or preempt as directed.

    Returns ``(final_state, last_checkpoint)``. With *resume*, it restarts from that
    checkpoint's step + 1. With *preempt_at*, it raises :class:`Preempted` after committing
    that step's checkpoint -- exactly what a resumed run must recover from.
    """
    checkpoints = CheckpointStore(store)
    if resume is not None:
        state = checkpoints.load(resume)
        start = resume.step + 1
    else:
        state = initial
        start = 1
    last = resume
    for step in range(start, steps + 1):
        state = step_fn(state, step)
        last = checkpoints.save(step, state)
        if preempt_at is not None and step == preempt_at:
            raise Preempted(last)
    return state, last
