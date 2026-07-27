"""The Astro-Mine-Sim command line and container entrypoint (RM-P0-SIM-11, RM-P0-CLOUD-01).

Two subcommands, because there are **two scenario schemas** and conflating them is a trap:

- ``astro-mine-sim run <bench-scenario-id>`` runs a **Bench** ``ScenarioSpec`` — a declarative,
  content-pinned *benchmark task* identified by id (e.g. ``lunar-polar-ice-prospecting-v1``). It
  resolves the pinned world/fleet/prospect/link content from a local Hub registry and materializes
  a runnable episode via :func:`~astro_mine.sim.bench.sim_scenario_from_spec`. Needs the ``[bench]``
  and ``[hub]`` extras and a content store (populate one with ``astro-mine-bench fetch``).
- ``astro-mine-sim record --scenario-file <path>`` runs a **Sim** ``Scenario`` — a materialized,
  self-contained *runnable episode* authored as a JSON document. This is the always-available local
  path: no Bench, no Hub, no content resolution.

Both write an MCAP recording and print the run's ``Trace.content_hash`` — the determinism key
(RM-P0-SIM-10) — so a run's reproducibility is checkable from its logs alone.

**Container entrypoint (RM-P0-CLOUD-01).** ``docker/Dockerfile`` runs ``python -m astro_mine.sim``
and Cloud appends ``--scenario <path> --seed N --out run.mcap``. That legacy flat form (leading with
a flag, no subcommand) routes to ``record``, so the container contract keeps working unchanged —
``--scenario`` is a deprecated alias for ``record``'s ``--scenario-file``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from astro_mine.sim.kernels import (
    METAKERNEL_ENV,
    KernelConfigurationError,
    furnish_metakernel,
    kernel_help,
)
from astro_mine.sim.recording import record_episode
from astro_mine.sim.runtime import load_scenario
from astro_mine.spice import SpiceGeometryError

#: Env fallback for the ``run`` content store, so ``--registry`` need not be repeated. Points at a
#: local OCI-layout Hub registry (the workspace convention; ``astro-mine-bench fetch`` fills one).
_REGISTRY_ENV = "ASTRO_MINE_HUB_REGISTRY"


def _record(args: argparse.Namespace) -> int:
    """Run a **Sim** ``Scenario`` JSON document and record it to MCAP; print the determinism key."""
    scenario = load_scenario(args.scenario_file)
    # Furnish *after* loading, so the scenario's own epoch window can be validated against the SPK
    # pool up front (#80; spice.md §10). A self-contained scenario over synthetic geometry needs no
    # kernels at all, and with none configured this is a no-op — the zero-prerequisite path stays
    # zero-prerequisite.
    furnish_metakernel(args.metakernel, scenario=scenario)
    trace = record_episode(scenario, args.out, seed=args.seed)
    print(trace.content_hash)
    return 0


def _run(args: argparse.Namespace) -> int:
    """Resolve a **Bench** ``ScenarioSpec`` id into a runnable Sim episode and record it to MCAP.

    Degrades with an actionable message at each missing layer — the ``[bench]`` loader, a content
    store, the ``[hub]`` client, an unknown id — never a traceback (CX-LOCAL). End-to-end against
    the anchor needs published content fetched into a local registry (gated on the anchor-content
    work). Bench is reached only through ``astro_mine.sim.bench`` (the adapter), so Sim's base
    runtime never imports Bench (conventions.md §1.1; bench.md §2.2).
    """
    import importlib.util

    # A probe, not an import (so the base runtime carries no Bench dependency and the narrow-waist
    # import guard stays green): `find_spec` takes a *string*, it does not import the package.
    if importlib.util.find_spec("astro_mine.bench") is None:
        print(
            "error: `run` executes a Bench ScenarioSpec and needs astro-mine-bench; "
            "install it with `pip install 'astro-mine-sim[bench]'`.",
            file=sys.stderr,
        )
        return 2

    registry = args.registry or os.environ.get(_REGISTRY_ENV)
    if not registry:
        print(
            "error: `run` needs a content store for the scenario's pinned "
            "world/fleet/prospect/link.\n"
            f"  pass --registry PATH or set ${_REGISTRY_ENV} to a local OCI-layout Hub registry;\n"
            "  populate one with `astro-mine-bench fetch <scenario>`.",
            file=sys.stderr,
        )
        return 2

    from astro_mine.sim.bench import materialize_bench_run
    from astro_mine.sim.runtime import open_bundle_store
    from astro_mine.sim.runtime.content import describe_unresolved

    try:
        store = open_bundle_store(registry)
    except ImportError:
        print(
            "error: resolving pinned content needs the Hub client; "
            "install it with `pip install 'astro-mine-sim[hub]'`.",
            file=sys.stderr,
        )
        return 2

    seed = 0 if args.seed is None else args.seed
    try:
        run = materialize_bench_run(args.scenario_id, store=store, seed=seed)
    except KeyError:
        print(
            f"error: unknown scenario id {args.scenario_id!r}; "
            "`astro-mine-bench list` shows the catalog.",
            file=sys.stderr,
        )
        return 2

    # A pin can resolve by digest and still rebuild nothing, because content and code ship
    # separately: `astro-mine-bench fetch` obtains the bundles, but turning a world bundle back
    # into a WorldProvider is astro-mine-worlds' job. Say so rather than recording a blind run
    # and exiting 0 (#67). This is a warning, not a refusal — `record` is the library tier and a
    # partial run is still a legitimate thing to ask for here; the *scoring* path refuses instead.
    unresolved = getattr(run, "unresolved", ())
    if unresolved:
        print(f"warning: {describe_unresolved(unresolved)}", file=sys.stderr)

    # The world resolves body-fixed frames through SPICE, so a pinned world needs a furnished pool.
    # Now that the scenario is materialized its epoch window is known, so a kernel set that stops
    # short of the episode fails here rather than mid-rollout (#80).
    furnish_metakernel(args.metakernel, scenario=run.scenario)

    trace = record_episode(
        run.scenario,
        args.out,
        seed=seed,
        world_provider=run.world_provider,
        # A sealed per-seed field is Prospect's to realize; the resolved bundle carries the pinned
        # one (mirrors SimEpisodeRunner._run, which suppresses the same variance).
        resource_field=run.resource_field,  # type: ignore[arg-type]
        connectivity=run.connectivity,
        content_hashes=run.content_hashes,
        unresolved=unresolved,
    )
    print(trace.content_hash)
    return 0


# --- Per-verb argument sets -------------------------------------------------------------------
#
# Each verb's flags live in one function so they can be attached to *either* parser: this package's
# own `astro-mine-sim <verb>`, and the umbrella's `astro-mine <verb>` (RFC-0011 §3, wired in
# astro_mine.sim.umbrella). Declaring them once is what stops the two surfaces from drifting — the
# `--scenario` container alias in particular must exist on both, since a Cloud workload and a
# laptop are supposed to be the same run (cloud.md §4).


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """`run` — execute a Bench ScenarioSpec by id on real physics."""
    parser.add_argument(
        "scenario_id", help="a Bench ScenarioSpec id (from `astro-mine-bench list`)"
    )
    parser.add_argument("--seed", type=int, default=None, help="episode seed (default: 0)")
    parser.add_argument("--out", type=Path, default=Path("run.mcap"), help="MCAP output path")
    parser.add_argument(
        "--registry",
        default=None,
        metavar="PATH",
        help=f"local OCI-layout Hub registry for pinned content (default: ${_REGISTRY_ENV})",
    )
    _add_metakernel_argument(parser)


def _add_metakernel_argument(parser: argparse.ArgumentParser) -> None:
    """``--metakernel`` — shared by both verbs, because both can touch SPICE-backed geometry.

    Declared once for the same reason the other flags are: this parser is also the umbrella's
    (``astro-mine run`` / ``astro-mine record``, RFC-0011 §3), and a flag that exists on only one
    of the two surfaces is a drift the user pays for.
    """
    parser.add_argument(
        "--metakernel",
        default=None,
        metavar="PATH",
        help=(
            "SPICE metakernel (.tm) furnished before the run, for body-fixed frames "
            f"(default: ${METAKERNEL_ENV}; kernels come from NAIF and are not shipped)"
        ),
    )


def add_record_arguments(parser: argparse.ArgumentParser) -> None:
    """`record` — run a self-contained Sim Scenario document and record it to MCAP."""
    # `--scenario` is the deprecated alias the container entrypoint passes (Dockerfile + Cloud);
    # `--scenario-file` is the canonical, disambiguated flag.
    parser.add_argument(
        "--scenario-file",
        "--scenario",
        dest="scenario_file",
        required=True,
        type=Path,
        metavar="PATH",
        help="path to a Sim Scenario JSON document",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the scenario's seed")
    parser.add_argument("--out", type=Path, default=Path("run.mcap"), help="MCAP output path")
    _add_metakernel_argument(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astro-mine-sim", description=__doc__.splitlines()[0] if __doc__ else None
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser(
        "run", help="run a Bench ScenarioSpec (by id) on real physics and record it"
    )
    add_run_arguments(run)
    run.set_defaults(func=_run)

    record = subcommands.add_parser(
        "record", help="record a Sim Scenario JSON document (a materialized episode) to MCAP"
    )
    add_record_arguments(record)
    record.set_defaults(func=_record)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch a Sim CLI command; return the process exit code."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Container back-compat (RM-P0-CLOUD-01): the Dockerfile ENTRYPOINT `python -m astro_mine.sim`
    # gets `--scenario <path> --seed N --out run.mcap` appended by Cloud. A form that leads with a
    # flag (no subcommand) routes to `record`, so that invocation keeps working unchanged.
    # `-h`/`--help` stay top-level.
    if args_list and args_list[0].startswith("-") and args_list[0] not in ("-h", "--help"):
        args_list = ["record", *args_list]
    parser = _build_parser()
    args = parser.parse_args(args_list)
    try:
        result: int = args.func(args)
    except KernelConfigurationError as exc:
        # A configured-but-unusable pool: missing file, malformed kernel, or coverage short of the
        # episode. The message already carries the remedy (#80).
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SpiceGeometryError as exc:
        # Nothing was configured and geometry was needed anyway — the failure this issue is about.
        # Previously a traceback four frames deep inside the illumination model; now it names the
        # two knobs that fix it (CX-LOCAL).
        print(
            f"error: this run needs SPICE geometry and no kernel pool is furnished.\n  {exc}\n  "
            + kernel_help(),
            file=sys.stderr,
        )
        return 2
    return result


if __name__ == "__main__":  # pragma: no cover  (the container's entrypoint)
    raise SystemExit(main())
