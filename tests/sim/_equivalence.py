"""The shard-vs-oracle equivalence assertion for the JAX fan-out (both tiers).

**What the fan-out actually guarantees: sharding does not change the physics.** A sharded
rollout must reassemble to the single-process rollout of the same batch — that is what makes
the fan-out's sharder/actor/aggregation trustworthy, and it is the property global-index
seeding exists to provide.

That guarantee is **numerical, not bitwise**, and the distinction is the whole point of this
module. The oracle runs one ``vmap`` over ``n`` envs; the shards run ``vmap`` over ``n/k`` envs
each. A different batch dimension gives XLA a different vectorization and reduction order, so
the last bits of a near-zero quantity can differ — and *whether* they differ depends on the CPU
the job lands on. These tests used to assert exact equality (``aggregated == oracle``) and so
passed locally and on some GitHub runners while failing on others, on a diff of ``5.1e-19`` vs
``5.3e-19``: noise around zero (astro-mine-sim#46).

Bit-identical results across differing batch shapes is **not something XLA promises**, so the
old assertion was never a contract the engine could keep. Asserting it anyway was worse than
merely flaky: this is a *determinism* check (``CX-REPRO``), and a determinism check that cries
wolf trains everyone to hit re-run — which is exactly how a real nondeterminism regression
would get waved through.

The tolerance below is deliberately ~1000x tighter than any genuine sharding bug (a wrong shard
boundary or a mis-seeded RNG stream perturbs a trajectory by O(1), not by 1e-12) and ~1000x
looser than last-bit reduction noise, so it keeps all of the test's teeth and none of its
fragility. If a future change needs a *bitwise* guarantee, it must be asserted within a
**fixed** batch shape (shard-vs-shard), never across one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["assert_shards_match_oracle"]

#: Relative tolerance on each position component. Loose enough to absorb a differing XLA
#: reduction order, far tighter than any real perturbation of the trajectory.
_RTOL = 1e-12
#: Absolute tolerance, for components that are physically zero — where a relative bound is
#: meaningless and the reduction-order noise (~1e-19) actually shows up.
_ATOL = 1e-15


def assert_shards_match_oracle(
    aggregated: Any, oracle: Any, *, what: str = "sharded rollout"
) -> None:
    """Assert a reassembled sharded rollout matches the single-process oracle.

    Both are ``[env][agent][xyz]`` nested float lists. Compared numerically — see the module
    docstring for why bitwise equality is not the contract here.
    """
    got = np.asarray(aggregated, dtype=np.float64)
    want = np.asarray(oracle, dtype=np.float64)
    assert got.shape == want.shape, (
        f"{what}: shape {got.shape} != oracle {want.shape} — the shards did not reassemble "
        "into the same batch, which is a sharding bug, not a numerical one"
    )
    np.testing.assert_allclose(
        got,
        want,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=(
            f"{what} diverged from the in-process oracle by more than reduction-order noise. "
            "This tolerance absorbs XLA's batch-shape-dependent rounding; exceeding it means "
            "sharding actually changed the physics (check shard boundaries and per-env RNG "
            "seeding), not that the tolerance is too tight."
        ),
    )
