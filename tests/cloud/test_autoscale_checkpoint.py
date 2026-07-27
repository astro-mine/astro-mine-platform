"""Chaos/preemption: a checkpoint-resumed run matches the uninterrupted result.

The RM-P1-CLOUD-03 acceptance criterion -- kill a spot node mid-run, resume from the last
content-addressed checkpoint, and assert the resumed run reproduces the uninterrupted result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.cloud.artifacts.addressing import content_address
from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
from astro_mine.cloud.autoscale.checkpoint import (
    Checkpoint,
    CheckpointStore,
    Preempted,
    run_checkpointed,
)


def _accumulate(state: bytes, step: int) -> bytes:
    """A deterministic step function -- append the step number to the running state."""
    return state + f"{step};".encode()


def test_checkpoint_store_round_trips(tmp_path: Path) -> None:
    store = CheckpointStore(FilesystemArtifactStore(tmp_path))
    ckpt = store.save(3, b"state-3")
    assert ckpt.step == 3
    assert store.load(ckpt) == b"state-3"


def test_resume_after_preemption_reproduces_the_uninterrupted_run(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)

    # (1) the golden, uninterrupted run
    uninterrupted, _ = run_checkpointed(steps=5, store=store, step_fn=_accumulate)

    # (2) the same run, preempted right after step 3 commits its checkpoint
    with pytest.raises(Preempted) as excinfo:
        run_checkpointed(steps=5, store=store, step_fn=_accumulate, preempt_at=3)
    last = excinfo.value.checkpoint
    assert last is not None and last.step == 3  # lost <= one interval

    # (3) resume from the last checkpoint and finish
    resumed, final = run_checkpointed(steps=5, store=store, step_fn=_accumulate, resume=last)

    assert resumed == uninterrupted
    assert content_address(resumed) == content_address(uninterrupted)
    assert final is not None and final.step == 5


def test_resume_from_a_checkpoint_starts_at_the_next_step(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    seed = CheckpointStore(store).save(2, _accumulate(_accumulate(b"", 1), 2))
    resumed, _ = run_checkpointed(steps=3, store=store, step_fn=_accumulate, resume=seed)
    # equals a straight 1..3 run -> resume only executed step 3
    straight, _ = run_checkpointed(steps=3, store=store, step_fn=_accumulate)
    assert resumed == straight


def test_checkpoint_is_frozen() -> None:
    ckpt = Checkpoint(step=1, state_address="sha256:" + "aa" * 32)
    with pytest.raises(Exception, match="frozen"):
        ckpt.step = 2  # type: ignore[misc]
