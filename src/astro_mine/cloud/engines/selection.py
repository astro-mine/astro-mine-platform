# SPDX-License-Identifier: Apache-2.0
"""Engine selection -- route a workload shape to the right engine.

Realizes ``cloud.md`` §2 principle 3 ("the right engine for the workload shape"): a
tightly-coupled/stateful :class:`~astro_mine.cloud.submission.jobspec.JobSpec`
(``distributed=True``) routes to **Ray**; a trivial one-shot routes to a plain **K8s Job**.
Sweeps and workflows always compile to **Argo** (fan-out / DAG) and so are not routed here.

Backlog: RM-P1-CLOUD-02 -- astro-mine-cloud#13
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_mine.cloud.submission.jobspec import JobSpec

__all__ = ["select_engine"]


def select_engine(job: JobSpec) -> str:
    """Return the engine name for a single *job*: ``ray`` if distributed, else ``k8sjob``."""
    return "ray" if job.distributed else "k8sjob"
