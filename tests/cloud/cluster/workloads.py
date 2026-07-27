"""The workloads the live-cluster tests run.

Each is a ``python -c`` script, run through the *same* ``JobSpec.command`` whether the backend is
``local`` (a subprocess in the repo's venv) or ``cluster`` (the in-pod harness in the workload
image). ``python`` resolves to the right interpreter in both -- the venv is on ``PATH`` in the
image exactly as ``uv run`` puts it on ``PATH`` on a workstation -- so the *command* is not a
variable in the equivalence experiment. Only the substrate is.
"""

from __future__ import annotations

import hashlib

# --- deterministic: y = x * seed, over the ASTRO_MINE_{INPUTS,OUTPUTS,SEED} contract ---------
#
# The same workload the local-backend tests use. Nothing here touches the object store: staging
# inputs in and capturing outputs back out is the *harness's* job on both sides, which is the
# whole point -- the workload cannot tell which backend is running it.
DETERMINISTIC = (
    "import os, pathlib;"
    "i=pathlib.Path(os.environ['ASTRO_MINE_INPUTS']);"
    "o=pathlib.Path(os.environ['ASTRO_MINE_OUTPUTS']);"
    "s=int(os.environ['ASTRO_MINE_SEED']);"
    "x=int((i/'x.txt').read_text());"
    "(o/'y.txt').write_text(str(x*s))"
)


def deterministic_output(x: int, seed: int) -> bytes:
    """What :data:`DETERMINISTIC` writes to ``y.txt`` -- computed independently of any backend."""
    return str(x * seed).encode()


# --- checkpointing: a hash chain that survives losing its pod --------------------------------
#
# The chain starts at the job's *seed* and step n is sha256(state_{n-1} + str(n)). Three
# properties make this the right shape for a chaos test:
#
#  1. it is *deterministic*, so a run that is killed and resumed must reproduce the uninterrupted
#     run's final output byte-for-byte -- which is the actual assertion (cloud.md §8);
#  2. its checkpoints are *content-addressed*, so a resumed pod recovers its progress with no
#     index, no listing and no coordination: it recomputes each step's address and asks the store
#     `exists?`. The newest one present is where it left off. The same trick lets the *test* watch
#     progress from outside the cluster, with no log scraping;
#  3. seeding the chain keeps two runs' checkpoints *disjoint*. Without that, a second chaos test
#     would find the first run's completed checkpoints already in the shared store, "resume" from
#     the final step, and pass without ever running anything.
CHECKPOINTED = """
import hashlib, os, pathlib, time

from astro_mine.cloud.artifacts.addressing import content_address
from astro_mine.cloud.artifacts.s3 import S3ArtifactStore
from astro_mine.cloud.autoscale.checkpoint import Checkpoint, run_checkpointed

steps = int(os.environ["WORKLOAD_STEPS"])
delay = float(os.environ.get("WORKLOAD_STEP_SECONDS", "2"))
initial = os.environ["ASTRO_MINE_SEED"].encode()

store = S3ArtifactStore(
    os.environ["ASTRO_MINE_S3_BUCKET"], endpoint_url=os.environ["ASTRO_MINE_S3_ENDPOINT"]
)


def advance(state, n):
    return hashlib.sha256(state + str(n).encode()).hexdigest().encode()


def step_fn(state, n):
    time.sleep(delay)  # the work; long enough that a pod can be killed in the middle of it
    return advance(state, n)


# Recover the newest checkpoint this run already committed, by recomputing each step's address
# and asking the store whether it is there. No sleeping: replay is pure arithmetic.
resume = None
replay = initial
for n in range(1, steps + 1):
    replay = advance(replay, n)
    address = content_address(replay)
    if not store.exists(address):
        break
    resume = Checkpoint(step=n, state_address=address)

print(f"resuming from {resume}", flush=True)
final, _last = run_checkpointed(
    steps=steps, store=store, step_fn=step_fn, initial=initial, resume=resume
)
pathlib.Path(os.environ["ASTRO_MINE_OUTPUTS"], "state.txt").write_bytes(final)
"""


def chain_state(seed: int, step: int) -> bytes:
    """The checkpointed workload's state after *step* -- the host's copy of the same recurrence.

    Recomputing it here rather than reading it back is what lets a test *predict* a checkpoint's
    content address: to watch for progress from outside the cluster, and to assert the final
    output without trusting the run that produced it.
    """
    state = str(seed).encode()
    for n in range(1, step + 1):
        state = hashlib.sha256(state + str(n).encode()).hexdigest().encode()
    return state


# --- trivial: exit 0, produce nothing (admission tests only care whether the pod is created) ---
NOOP = "print('ok')"
