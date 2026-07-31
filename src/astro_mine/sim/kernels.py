"""Furnishing the SPICE kernel pool from the CLI and the environment (#80).

Sim resolves body-fixed frames through :mod:`astro_mine.spice` (RFC-0002), and SPICE cannot
answer a geometry query until a kernel pool is furnished. Nothing in the platform furnished one:
``astro_mine.spice.load_metakernel`` existed and no entry point called it, so a Sim-backed anchor
run — the platform's central claim — could not be driven from a shell at all. It failed four
frames deep with ``SPICE(UNKNOWNFRAME)``, and the only way through was for the caller to import
Spice and furnish the pool before invoking the CLI.

This module closes that. It is deliberately small and does three things:

**Resolves a path** — an explicit argument (the CLI's ``--metakernel``) wins over
:data:`METAKERNEL_ENV`, mirroring how ``--registry`` wins over ``$ASTRO_MINE_HUB_REGISTRY``
throughout Sim. Neither is required: **with nothing configured this is a no-op**, because plenty
of legitimate runs need no kernels at all (``astro-mine sim record`` on the shipped reference
scenario is the case that matters — it is the zero-prerequisite path and it must stay that way).

**Furnishes at most once per process.** SPICE's pool is process-global, shared with every other
consumer in the interpreter (Worlds' illumination, Link's LOS). So this never clears the pool —
clearing would silently break a consumer that furnished its own kernels — and it skips a repeat
furnish of the same path rather than stacking duplicates.

**Validates coverage up front, where the epoch window is known.** ``load_metakernel`` accepts a
``coverage`` window and lifts SPICE's mid-query ``SPICE(SPKINSUFFDATA)`` to furnish time
(``spice.md`` §10). That is only useful if the window is in hand, which is why callers furnish
*after* materializing a scenario rather than at construction: a kernel set that stops short of a
30-day lunar episode then fails in the first second rather than ~18,000 ticks in.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_mine.core.units import Epoch, EpochWindow
    from astro_mine.sim.runtime import Scenario

__all__ = [
    "METAKERNEL_ENV",
    "KernelConfigurationError",
    "furnish_metakernel",
    "kernel_help",
    "scenario_epoch_window",
]

#: Env fallback for the SPICE metakernel, so ``--metakernel`` need not be repeated — and so
#: ``astro-mine bench score --runner sim`` can furnish a pool without Bench ever naming SPICE.
#: Bench discovers the ``sim`` runner through an entry point and passes it a content store; it has
#: no vocabulary for kernels and must not grow one (conventions.md §1.1), so the runner reads its
#: own environment, exactly as it already does for ``$ASTRO_MINE_HUB_REGISTRY``.
METAKERNEL_ENV = "ASTRO_MINE_SPICE_METAKERNEL"

#: Paths already furnished in this process, so a second call is a no-op rather than a duplicate
#: ``furnsh``. Keyed by resolved absolute path: two spellings of one file are one kernel.
_furnished: set[Path] = set()


class KernelConfigurationError(Exception):
    """Raised when kernels are needed but the pool cannot be furnished as configured.

    Carries a message naming how to fix it — the flag, the env var, and where kernels come from.
    The CLI prints it and exits non-zero; it never reaches the user as a traceback (CX-LOCAL).
    """


def kernel_help() -> str:
    """The actionable remedy, in the words a user needs to act on it.

    Shared by every path that can fail for want of kernels, so the CLI, the scoring runner, and a
    library caller all name the same two knobs and the same source.
    """
    return (
        f"supply a SPICE metakernel with --metakernel PATH or ${METAKERNEL_ENV}.\n"
        "  Kernels are not shipped with Astro-Mine; obtain SPK/PCK/FK/LSK kernels from NAIF\n"
        "  (https://naif.jpl.nasa.gov/naif/data.html) and list them in a metakernel (.tm)."
    )


def scenario_epoch_window(scenario: Scenario) -> EpochWindow:
    """The half-open epoch window a scenario's episode spans, for coverage validation.

    ``[start_epoch, start_epoch + dt_s * horizon_steps)`` — the interval SPK data has to cover for
    the run to complete. Deriving it here keeps the arithmetic in one place; the anchor's 43,200
    steps of 60 s is a 30-day window, and a kernel set that stops short of it is the failure this
    exists to catch early.
    """
    from astro_mine.core.units import EpochWindow

    span_s = scenario.dt_s * scenario.horizon_steps
    start: Epoch = scenario.start_epoch
    # Copy the start epoch rather than constructing one: `scale` is required and has no default
    # (a scaleless epoch fails loudly by design), and the end of an episode is trivially in the
    # same time scale as its start.
    end = start.model_copy(update={"tdb_seconds": start.tdb_seconds + span_s})
    return EpochWindow(start=start, end=end)


def furnish_metakernel(
    path: str | Path | None = None,
    *,
    scenario: Scenario | None = None,
) -> Path | None:
    """Furnish the SPICE pool from ``path``, else :data:`METAKERNEL_ENV`; return what was used.

    Returns ``None`` when neither is set — the deliberate no-op for runs that need no geometry.
    Raises :class:`KernelConfigurationError`, with the remedy in the message, when a configured
    metakernel is missing, malformed, or does not span ``scenario``'s episode.

    ``scenario`` is read **only when a metakernel is actually being furnished**, and only for its
    epoch window. That laziness is deliberate: with no kernels configured this function must not
    touch the scenario at all, so the no-op path stays a true no-op for any caller — including one
    holding something that is not a full ``Scenario``.

    Furnishing the same resolved path twice in one process is a no-op. The pool is never cleared:
    it is shared with every other Spice consumer in the interpreter, and dropping their kernels to
    tidy ours would be a silent correctness bug in someone else's component.
    """
    from astro_mine.spice import SpiceKernelError, load_metakernel

    raw = path if path is not None else os.environ.get(METAKERNEL_ENV)
    if raw is None or raw == "":
        return None

    resolved = Path(raw).expanduser()
    if resolved in _furnished:
        return resolved

    coverage = scenario_epoch_window(scenario) if scenario is not None else None

    if not resolved.exists():
        raise KernelConfigurationError(
            f"SPICE metakernel not found: {resolved}\n  " + kernel_help()
        )

    try:
        load_metakernel(resolved, coverage=coverage)
    except SpiceKernelError as exc:
        # Spice already fails loudly and says what is wrong — a missing file, or an SPK pool that
        # does not span the window. Keep its diagnosis and add the remedy it has no way to know.
        raise KernelConfigurationError(f"{exc}\n  " + kernel_help()) from exc

    _furnished.add(resolved)
    return resolved
