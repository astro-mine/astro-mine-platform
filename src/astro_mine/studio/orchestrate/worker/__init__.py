# SPDX-License-Identifier: Apache-2.0
"""The design-loop worker: one candidate evaluation, as a process (RM-P1-STUDIO-03).

:class:`~.cloud.CloudDispatcher` fans candidates out through ``cloud.submit()``, and Cloud
executes a **job**, not a Python callable: the local backend subprocesses ``job.command``, the
docker backend runs it in a digest-pinned image, and both present one file-based I/O contract
(``ASTRO_MINE_INPUTS`` / ``ASTRO_MINE_OUTPUTS`` / ``ASTRO_MINE_SEED``; cloud.md §3). This module
is the other end of that contract — the argv Studio asks Cloud to run:

    python -m astro_mine.studio.orchestrate.worker

It reads one :class:`EvaluationRequest` (plus the objective in Core's byte-stable wire form)
from the inputs directory, runs the **same** :func:`~.loop.evaluate_candidate` the in-process
:class:`~.jobs.LocalDispatcher` runs, and writes one :class:`EvaluationOutcome` to the outputs
directory. Same loop, same seed, same provenance — a Cloud evaluation and a local one are
byte-identical for the same ``(candidate, objective, seed)``, which is what makes
``CloudDispatcher`` a drop-in ``JobDispatcher`` rather than a second, weaker code path.

**Failures ride in the outcome, not in the exit code.** A Guard rejection is a *result* — the
candidate is infeasible, and ``run_batch`` must record it as a ``FAILED`` job carrying Guard's
reason, exactly as it does locally. But Cloud collects declared outputs only on exit 0, and a
``RunResult`` carries no error payload — so a worker that exited non-zero on rejection would
lose the reason and downgrade the contract. Instead the worker always exits 0 when it *ran*,
and reports feasibility inside :class:`EvaluationOutcome`. A non-zero exit therefore means what
it should: the worker itself broke (bad image, missing dependency, OOM), which
``CloudDispatcher`` surfaces as a :class:`~.cloud.CloudEvaluationError`.

**The sibling clients are an injected seam, here too.** Which
:class:`~.clients.SiblingClients` bundle the worker binds is named by
``ASTRO_MINE_STUDIO_CLIENTS`` (``"module:factory"``), defaulting to the all-local bundle. A
real deployment points it at a factory that binds Sim/Learn/Mind/Allocate/Guard/Bench over the
network; nothing else about the worker changes (studio.md §2 principle 1 — Studio computes
nothing, it sequences).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from astro_mine.core.objective import ObjectiveDocument, from_wire, to_wire

from ..._base import FrozenStudioModel
from ...models import DesignCandidate, EvaluatedCandidate
from ..clients import GuardRejection, SiblingClients
from ..loop import evaluate_candidate

__all__ = [
    "CLIENTS_ENV",
    "DEFAULT_CLIENTS_FACTORY",
    "OBJECTIVE_INPUT",
    "OUTCOME_OUTPUT",
    "REQUEST_INPUT",
    "EvaluationOutcome",
    "EvaluationRequest",
    "load_clients",
    "main",
    "run_request",
]

#: The run-relative filenames of the job's I/O contract. Cloud stages ``inputs`` by name into
#: ``$ASTRO_MINE_INPUTS`` and captures ``outputs`` by name from ``$ASTRO_MINE_OUTPUTS``.
REQUEST_INPUT = "request.json"
OBJECTIVE_INPUT = "objective.pb"
OUTCOME_OUTPUT = "outcome.json"

#: Names the ``SiblingClients`` factory the worker binds, as ``"module:callable"``.
CLIENTS_ENV = "ASTRO_MINE_STUDIO_CLIENTS"
DEFAULT_CLIENTS_FACTORY = "astro_mine.studio.orchestrate.clients:local_clients"


class EvaluationRequest(FrozenStudioModel):
    """One candidate evaluation, as the worker receives it.

    The objective travels *beside* this, as its Core wire form (``OBJECTIVE_INPUT``) — the same
    byte-stable encoding ``cache_key``/``objective_content_hash`` address it by, so the payload
    Cloud content-addresses and the identity Studio caches on are one and the same.
    """

    candidate: DesignCandidate
    seed: int
    max_steps: int


class EvaluationOutcome(FrozenStudioModel):
    """What the worker produces: a scored candidate, or the reason it is infeasible.

    ``evaluated`` and ``error`` are exclusive — exactly one is set. An infeasible candidate is a
    successful *run* with a negative *result* (see the module docstring).
    """

    evaluated: EvaluatedCandidate | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.evaluated is not None


def load_clients(spec: str | None = None) -> SiblingClients:
    """Import and call the ``"module:factory"`` named by *spec* (or ``$ASTRO_MINE_STUDIO_CLIENTS``).

    The default binds the deterministic in-process bundle, so the worker runs with no siblings
    and no network — the local tier MUST work (conventions.md §7).
    """
    target = spec or os.environ.get(CLIENTS_ENV) or DEFAULT_CLIENTS_FACTORY
    module_name, separator, attribute = target.partition(":")
    if not (separator and module_name and attribute):
        raise ValueError(f"{CLIENTS_ENV} must be 'module:factory', got {target!r}")
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory()  # type: ignore[no-any-return]  # the factory is external by construction


def run_request(
    request: EvaluationRequest, objective: ObjectiveDocument, *, clients: SiblingClients
) -> EvaluationOutcome:
    """Run one candidate through the design loop and wrap the result (or Guard's veto)."""
    try:
        evaluated = evaluate_candidate(
            request.candidate,
            objective,
            clients=clients,
            seed=request.seed,
            max_steps=request.max_steps,
            cache=None,  # the cache is the dispatcher's, on the Studio side of the job boundary
        )
    except GuardRejection as rejection:
        return EvaluationOutcome(error=str(rejection))
    return EvaluationOutcome(evaluated=evaluated)


def _container() -> Any:
    """The worker's composition root: bind :class:`SiblingClients` to the factory named by env.

    One of the four places the platform is assembled into an application (conventions.md §3.3),
    and one of only two inside this distribution that may import ``svcs`` — the layering suite
    allows it here by name and nowhere else.

    This root is the clearest case for the container in the tree, because the wiring already
    existed in hand-rolled form: :func:`load_clients` resolves a ``"module:factory"`` string from
    the environment and calls it, which is a one-entry registry with no type on it. Registering it
    as a factory *for* :class:`SiblingClients` is the same indirection with the contract named, and
    it puts the binding where a reader looks for it.

    Built, used, and dropped — no module-level container (§3.3).
    """
    # Imported here rather than at module scope, for the same reason as the Cloud harness: this
    # package is imported for its I/O constants by `orchestrate/cloud.py`, and a composition root
    # should not put its container on the import path of everything that reads a filename off it.
    import svcs

    registry = svcs.Registry()
    registry.register_factory(SiblingClients, load_clients)
    return svcs.Container(registry)


def main() -> int:
    """The ``python -m astro_mine.studio.orchestrate.worker`` entry point."""
    inputs = Path(os.environ["ASTRO_MINE_INPUTS"])
    outputs = Path(os.environ["ASTRO_MINE_OUTPUTS"])

    request = EvaluationRequest.model_validate_json((inputs / REQUEST_INPUT).read_bytes())
    objective = from_wire((inputs / OBJECTIVE_INPUT).read_bytes())
    with _container() as services:
        outcome = run_request(request, objective, clients=services.get(SiblingClients))

    (outputs / OUTCOME_OUTPUT).write_text(outcome.model_dump_json())
    return 0


def encode_objective(objective: ObjectiveDocument) -> bytes:
    """The objective's byte-stable Core wire form — what rides as ``OBJECTIVE_INPUT``."""
    return to_wire(objective)
